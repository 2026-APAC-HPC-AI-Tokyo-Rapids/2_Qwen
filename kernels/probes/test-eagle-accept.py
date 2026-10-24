#!/usr/bin/env python3
"""CPU checks for prompt parity, acceptance collection and process cleanup."""

import contextlib
import copy
import importlib.util
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load("eagle_metrics", ROOT / "scripts/collect-eagle-acceptance.py")
probe = load("eagle_accept", ROOT / "kernels/probes/eagle-accept.py")


def fixture():
    info = {"speculative_algorithm": "EAGLE3", "speculative_num_steps": 3, "speculative_eagle_topk": 1, "speculative_num_draft_tokens": 4, "tp_size": 4, "dp_size": 1, "enable_dp_attention": False, "disaggregation_mode": "null", "moe_a2a_backend": "none", "dtype": "auto", "context_length": 8192, "kv_cache_dtype": "fp8_e4m3", "internal_states": [{"avg_spec_accept_length": 2.2}, {"avg_spec_accept_length": 2.4}]}
    export = {"aiperf_version": "0.12.0", "was_cancelled": False, "error_summary": [], "request_count": {"avg": 32.0}, "input_config": {"phases": [{"name": "profiling", "kind": "profiling", "type": "concurrency", "duration": 20.0, "concurrency": 32, "grace_period": 30.0}], "tokenizer": {"name": probe.TOKENIZER}, "datasets": [{"name": "main", "type": "synthetic", "entries": 12800, "random_seed": 2026, "prompts": {"isl": {"mean": 4000.0, "stddev": 0.0}, "osl": {"mean": 200.0, "stddev": 0.0}, "cache_bust": {"target": "first_turn_prefix"}}}], "endpoint": {"extra": {"min_tokens": 200, "max_tokens": 200, "ignore_eos": True, "temperature": 0.0, "repetition_penalty": 1.0}}}}
    return info, export


class ProbeTests(unittest.TestCase):
    def test_command_matches_official_except_timings_and_concurrency(self):
        script = (ROOT / "platforms/firmus/framework-comparison/official-aiperf.sh").read_text()
        command = next(line for line in script.splitlines() if line.startswith("command=("))[9:-1]
        command = command.replace("${N}", "32").replace('"${timings[@]}"', "--benchmark-duration 20 --benchmark-grace-period 30")
        self.assertEqual(probe.aiperf_command(), shlex.split(command))
        server = probe.server_command(Path("/runtime"))
        for flag in ("--enable-dp-attention", "--disaggregation-mode", "--enable-dp-lm-head"):
            self.assertNotIn(flag, server)
        self.assertIn("--disable-cuda-graph", server)
        self.assertEqual(server[server.index("--moe-a2a-backend") + 1], "none")

    def test_all_states_and_bonus_and_logs(self):
        info, export = fixture()
        log = "Decode batch accept len: 2.10, accept rate: 0.37\nDecode batch accept len: 2.50, accept rate: 0.50\n"
        result = probe.collect(info, log, export, 120.5)
        self.assertAlmostEqual(result["accept_length"], 2.3)
        self.assertAlmostEqual(result["accepted_draft"], 1.3)
        self.assertEqual(result["requests"], 32)
        self.assertEqual(result["log_samples"]["accept_length"]["count"], 2)

    def test_reject_incomplete_states_and_wrong_route(self):
        info, export = fixture()
        for states in ([], [{"avg_spec_accept_length": None}], [{"avg_spec_accept_length": float("nan")}], [{"avg_spec_accept_length": 2.2}, {}], [{"avg_spec_accept_length": 4.5}]):
            candidate = dict(info, internal_states=states)
            with self.subTest(states=states), self.assertRaises(ValueError):
                probe.collect(candidate, "", export, 1)
        with self.assertRaises(ValueError):
            probe.collect(dict(info, disaggregation_mode="decode"), "", export, 1)

    def test_reject_failed_or_wrong_workload(self):
        _, export = fixture()
        for updates in ({"was_cancelled": True}, {"error_summary": [{"error": "failed"}]}, {"request_count": {"avg": 0}}, {"error_request_count": {"avg": 1}}, {"osl_mismatch_count": {"avg": 1}}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                probe.completed_requests(dict(export, **updates))
        wrong = copy.deepcopy(export)
        wrong["input_config"]["datasets"][0]["random_seed"] = 1
        with self.assertRaises(ValueError):
            probe.completed_requests(wrong)
        wrong = copy.deepcopy(export)
        wrong["input_config"]["phases"].append({"kind": "warmup"})
        with self.assertRaises(ValueError):
            probe.completed_requests(wrong)

    def test_execute_saves_results_and_always_stops_both_processes(self):
        info, export = fixture()
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "logs").mkdir()
                artifact = root / "results/artifacts/qwen"
                artifact.mkdir(parents=True)
                (artifact / "profile_export_aiperf.json").write_text(json.dumps(export))
                (artifact / "profile_export.jsonl").write_text('{}\n')
                children = []

                class Child:
                    def __init__(self, command, failure=fail, registered=children, **kwargs):
                        self.returncode = 1 if failure else 0
                        self.server = "sglang.launch_server" in command
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
                    stopped = stack.enter_context(patch.object(probe, "stop"))
                    output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    if fail:
                        with self.assertRaises(RuntimeError):
                            probe.execute(root, time.time())
                        self.assertFalse((root / "results/eagle-acceptance.json").exists())
                        self.assertNotIn("PASS ", output.getvalue())
                    else:
                        self.assertEqual(probe.execute(root, time.time())["requests"], 32)
                        self.assertTrue((root / "results/server-info.json").is_file())
                        self.assertTrue((root / "results/eagle-acceptance.json").is_file())
                    self.assertEqual([call.args[0] for call in stopped.call_args_list], list(reversed(children)))

    def test_stop_real_cpu_process_group(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            probe.stop(child)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


if __name__ == "__main__":
    unittest.main()
