#!/usr/bin/env python3
"""CPU fixtures for pinned draft preparation and parallel acceptance measurement."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load("legacy_eagle_tests", ROOT / "kernels/probes/test-eagle-accept.py")
probe = legacy.probe
preparer = load("prepare_eagle_drafts", ROOT / "kernels/probes/prepare-eagle-drafts.py")


def fixture():
    info, export = legacy.fixture()
    info.update(speculative_draft_model_quantization="fp8", random_seed=2026, disable_cuda_graph=True)
    export["request_count"]["avg"] = 128
    export["input_config"]["phases"][0].update(duration=180, requests=128)
    return info, export


def manifest(root):
    drafts = json.loads((ROOT / "kernels/probes/eagle-drafts.json").read_text())
    (root / "source").mkdir(exist_ok=True)
    (root / "source/eagle-drafts.json").write_text(json.dumps(drafts))
    for name in ("nebius", "meituan"):
        (root / "logs" / name).mkdir(parents=True)
        (root / "results" / name).mkdir(parents=True)
    return drafts


class DraftTests(unittest.TestCase):
    def test_commands_change_only_limits_url_and_precision(self):
        official = probe.aiperf_command()
        command = probe.aiperf_command("meituan", 8100)
        official[official.index("--url") + 1] = "http://127.0.0.1:8100"
        official[official.index("--benchmark-duration") + 1] = "180"
        self.assertEqual(command, official + ["--request-count", "128"])
        a, b = [probe.server_command(Path("/runtime"), name, port) for name, port in (("nebius", 8000), ("meituan", 8100))]
        for command in (a, b):
            self.assertEqual(command[command.index("--speculative-draft-model-quantization") + 1], "fp8")
            self.assertIn("--disable-cuda-graph", command)
        for flag in ("--port", "--nccl-port", "--dist-init-addr"):
            self.assertNotEqual(a[a.index(flag) + 1], b[b.index(flag) + 1])

    def test_exact_count_precision_and_phase_validation(self):
        info, export = fixture()
        self.assertEqual(probe.collect(info, "", export, 120, "nebius")["requests"], 128)
        for key, value in (("speculative_draft_model_quantization", None), ("random_seed", 1), ("disable_cuda_graph", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.collect(dict(info, **{key: value}), "", export, 120, "nebius")
        for change in ("count", "phase", "dataset"):
            wrong = copy.deepcopy(export)
            if change == "count":
                wrong["request_count"]["avg"] = 127
            elif change == "phase":
                wrong["input_config"]["phases"][0]["requests"] = 127
            else:
                wrong["input_config"]["datasets"][0]["entries"] = 128
            with self.subTest(change=change), self.assertRaises(ValueError):
                probe.completed_requests(wrong, "nebius")

    def test_per_request_and_batch_counts(self):
        log = "EAGLE_POSITION histogram=[1, 2, 3, 4]\nEAGLE_POSITION histogram=[0, 2, 0, 0]\n"
        log += "EAGLE_REQUEST rid=a verify_steps=100 accepted_draft_tokens=120\nEAGLE_REQUEST rid=b verify_steps=80 accepted_draft_tokens=120\n"
        positions = probe.position_acceptance(log)
        self.assertEqual(positions["verify_batches"], 2)
        self.assertEqual(positions["verification_requests"], 12)
        requests = probe.request_acceptance(log, 2)
        self.assertEqual([r["accept_length_including_bonus"] for r in requests], [2.2, 2.5])
        self.assertEqual(sum(r["verify_steps"] for r in requests), 180)
        for bad in ("", log + log, "EAGLE_REQUEST rid=a verify_steps=0 accepted_draft_tokens=0", "EAGLE_REQUEST rid=a verify_steps=1 accepted_draft_tokens=4"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                probe.request_acceptance(bad, 2)

    def test_prepare_verifies_bytes_and_never_changes_weights(self):
        for name in ("nebius", "meituan"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                original = json.dumps({"architectures": ["LlamaForCausalLM" if name == "nebius" else "LlamaForCausalLMEagle3"], "dtype": "bfloat16"}).encode()
                header = b'{"fc.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}'
                weights = struct.pack("<Q", len(header)) + header + b"\0\0"
                (root / "config.json").write_bytes(original)
                (root / "model.safetensors").write_bytes(weights)
                (root / "README.md").write_text("fixture")
                descriptor = next(d for d in json.loads((ROOT / "kernels/probes/eagle-drafts.json").read_text()) if d["name"] == name)
                descriptor.update(weight_bytes=len(weights), weight_sha256=hashlib.sha256(weights).hexdigest(), header_sha256=hashlib.sha256(header).hexdigest(), original_config_sha256=hashlib.sha256(original).hexdigest())
                preparer.prepare(root, root / "evidence", descriptor)
                self.assertEqual((root / "model.safetensors").read_bytes(), weights)
                runtime = json.loads((root / "config.json").read_text())
                self.assertEqual(runtime["architectures"], ["LlamaForCausalLMEagle3"])
                self.assertEqual((root / "evidence/draft-original-config.json").read_bytes(), original)
                if name == "meituan":
                    self.assertEqual((root / "config.json").read_bytes(), original)
                provenance = json.loads((root / "evidence/draft-provenance.json").read_text())
                self.assertEqual(provenance["draft_precision"], "fp8")
                self.assertEqual(provenance["source_precision"], "BF16")
                for key, value in (("weight_bytes", 1), ("weight_sha256", "bad"), ("header_sha256", "bad"), ("original_config_sha256", "bad")):
                    (root / "config.json").write_bytes(original)
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        preparer.prepare(root, root / "bad", dict(descriptor, **{key: value}))
                self.assertFalse((root / "bad").exists())

    def test_redhat_is_explicitly_excluded(self):
        redhat = next(d for d in json.loads((ROOT / "kernels/probes/eagle-drafts.json").read_text()) if d["name"] == "redhat")
        self.assertFalse(redhat["enabled"])
        self.assertIn("norm_before_residual=true", redhat["reason"])
        with self.assertRaises(ValueError):
            preparer.prepare(Path("/unused"), Path("/unused"), redhat)

    def test_completed_request_instrumentation_uses_cpu_only_counters(self):
        instrument = load("instrument_eagle_drafts", ROOT / "kernels/probes/instrument-eagle-accept.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processor.py"
            path.write_text("def settle(self, result):\n        predict_tokens = []\n        # In adaptive spec-v2\n        return result.num_correct_drafts_per_req_cpu\n\ndef finish(self, req):\n        if req.finished():\n            # isinstance narrowing: create_worker may also return plain\n            pass\n")
            instrument.instrument(path)
            module = load("request_instrument_fixture", path)
            calls = []
            module.logger = SimpleNamespace(info=lambda *args: calls.append(args))
            req = SimpleNamespace(finished=lambda: True, spec_verify_ct=10, spec_num_correct_drafts=12, rid="a")
            owner = SimpleNamespace(metrics_reporter=SimpleNamespace(is_stats_logging_rank=True))
            module.finish(owner, req)
            self.assertEqual(calls[0][1:], ("a", 10, 12))
            owner.metrics_reporter.is_stats_logging_rank = False
            module.finish(owner, req)
            self.assertEqual(len(calls), 1)

    def test_parallel_workers_real_cpu_processes_and_failure_cleanup(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest(root)
                python = root / "fake-python"
                python.write_text("""#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
args = sys.argv
root = Path(args[args.index('--firmus') - 2])
name = args[args.index('--draft') + 1]
(root/'results'/name/'launch.json').write_text(json.dumps({'gpus':os.environ['CUDA_VISIBLE_DEVICES'],'port':args[args.index('--port')+1]}))
if os.environ.get('FAIL_DRAFT') == name:
    sys.exit(7)
time.sleep(0.1)
descriptor = next(d for d in json.loads((root/'source/eagle-drafts.json').read_text()) if d['name']==name)
result = {'draft':descriptor,'revision':descriptor['revision'],'draft_precision':'fp8','accept_length':2.2 if name=='nebius' else 2.4,'accepted_draft':1.2 if name=='nebius' else 1.4,'requests':128,'verify_steps':11500}
(root/'results'/name/'eagle-acceptance.json').write_text(json.dumps(result))
""")
                python.chmod(0o755)
                children = []
                popen = subprocess.Popen

                def track(*args, factory=popen, registered=children, **kwargs):
                    child = factory(*args, **kwargs)
                    registered.append(child)
                    return child

                with patch.object(probe, "PYTHON", str(python)), patch.object(probe.subprocess, "Popen", track), patch.dict(os.environ, FAIL_DRAFT="meituan" if failed else ""), contextlib.redirect_stdout(io.StringIO()):
                    if failed:
                        with self.assertRaises(RuntimeError):
                            probe.compare_drafts(root, time.time())
                    else:
                        result = probe.compare_drafts(root, time.time())
                        self.assertAlmostEqual(result["drafts"][1]["relative_to_nebius"], 2.4 / 2.2)
                        self.assertEqual(result["excluded"][0]["name"], "redhat")
                self.assertTrue(all(child.poll() is not None for child in children))
                self.assertEqual((root / "results/eagle-acceptance.json").exists(), not failed)
                for name, gpus, port in (("nebius", "0,1,2,3", "8000"), ("meituan", "4,5,6,7", "8100")):
                    launch = json.loads((root / "results" / name / "launch.json").read_text())
                    self.assertEqual(launch, {"gpus": gpus, "port": port})

    def test_worker_collects_isolated_evidence_and_skips_english(self):
        info, export = fixture()
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest(root)
                artifacts = root / "results/meituan/artifacts/qwen"
                artifacts.mkdir(parents=True)
                (artifacts / "profile_export_aiperf.json").write_text(json.dumps(export))
                (artifacts / "profile_export.jsonl").write_text("{}\n")
                children = []

                class Child:
                    def __init__(self, command, failed=failure, runtime=root, registered=children, **kwargs):
                        self.server = "sglang.launch_server" in command
                        self.returncode = 7 if failed else 0
                        log = runtime / "logs/meituan/server.log"
                        with log.open("a") as stream:
                            if self.server:
                                stream.write("The server is fired up and ready to roll!\n")
                            else:
                                stream.write("EAGLE_POSITION histogram=[1, 2, 3, 4]\n")
                                for n in range(128):
                                    stream.write(f"EAGLE_REQUEST rid={n} verify_steps=100 accepted_draft_tokens=120\n")
                        registered.append(self)

                    def poll(self):
                        return None if self.server else self.returncode

                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"))
                    stack.enter_context(patch.object(probe, "version", return_value="0.12.0"))
                    stack.enter_context(patch.object(probe.subprocess, "Popen", Child))
                    stack.enter_context(patch.object(probe.signal, "signal"))
                    stack.enter_context(patch.object(probe.signal, "setitimer"))
                    stack.enter_context(patch.object(probe.urllib.request, "urlopen")).return_value.__enter__.return_value.status = 200
                    stack.enter_context(patch.object(probe, "get_json", return_value=info))
                    english = stack.enter_context(patch.object(probe, "collect_english"))
                    stopped = stack.enter_context(patch.object(probe, "stop"))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    if failure:
                        with self.assertRaises(RuntimeError):
                            probe.execute(root, time.time(), firmus=True, draft="meituan", port=8100)
                    else:
                        result = probe.execute(root, time.time(), firmus=True, draft="meituan", port=8100)
                        self.assertEqual(result["verify_steps"], 12800)
                        self.assertEqual(len(result["per_request"]), 128)
                        self.assertEqual(result["revision"], json.loads((root / "source/eagle-drafts.json").read_text())[1]["revision"])
                    english.assert_not_called()
                    self.assertEqual([c.args[0] for c in stopped.call_args_list], list(reversed(children)))
                self.assertEqual((root / "results/meituan/eagle-acceptance.json").exists(), not failure)
                self.assertFalse((root / "results/nebius/eagle-acceptance.json").exists())


if __name__ == "__main__":
    unittest.main()
