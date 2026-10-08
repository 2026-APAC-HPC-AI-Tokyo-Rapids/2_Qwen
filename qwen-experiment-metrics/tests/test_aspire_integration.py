"""CPU fixtures for the candidate port; Enroot, PBS, and GPUs are not invoked."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).parents[2]
FIXTURE = Path(__file__).parent / "fixtures/aspire-integration-argv.json"


def capture_workers(source: str, platform: str, nodes: int, tmp_path: Path, profile: bool = False) -> list[dict]:
    """Execute the real worker function and topology with a recording Enroot stub."""
    root = tmp_path / f"{platform}-{nodes}-{profile}"
    root.mkdir(parents=True, exist_ok=True)
    output = root / "output"
    for directory in ("logs", "source", "scripts"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    (output / "source/aspire-serving.json").write_bytes((REPOSITORY / "configs/aspire-serving.json").read_bytes())
    record = root / "commands.jsonl"
    record.write_text("")
    stub = root / "enroot"
    stub.write_text(f"#!{sys.executable}\nimport json, os, sys\nfd=os.open(os.environ['ARGV_RECORD'], os.O_WRONLY|os.O_APPEND)\nos.write(fd, (json.dumps({{'argv':sys.argv[1:], 'profiler':os.environ.get('QWEN_PROFILER_INVOCATION_PATH')}})+'\\n').encode())\nos.close(fd)\n")
    stub.chmod(0o755)
    profiler = output / "scripts/exec-profiler.sh"
    profiler.write_text('#!/bin/bash\nexec "$@"\n')
    profiler.chmod(0o755)
    worker = source.split("  start_worker() {", 1)[1].split("\n  }\n", 1)[0] + "\n  }\n"
    topology = source.split("\n  case \"${QWEN_PLATFORM}\"", 1)[1].split("\n  esac", 1)[0]
    harness = 'set -euo pipefail\nrestore_worker_cache() { mkdir -p "$2"; }\npids=()\ncache_workers=()\nstart_worker() {' + worker + '\nfor ((node_rank=0;node_rank<NODE_COUNT;node_rank++)); do\ncase "${QWEN_PLATFORM}"' + topology + '\nesac\ndone\nfor pid in "${pids[@]}"; do wait "$pid"; done\n'
    environment = {**os.environ, "PATH": f"{root}:{os.environ['PATH']}", "ARGV_RECORD": str(record), "QWEN_PLATFORM": platform, "NODE_COUNT": str(nodes), "GPUS_PER_NODE": "4" if platform == "genkai" else "8", "PROFILE": "true" if profile else "false", "OUTPUT_DIR": str(output), "node_tmp": str(root / "node"), "cache_root": str(root / "cache"), "MODEL_PATH": "/fixture/model", "CONTAINER_IMAGE": "/fixture/image.sqsh", "ucx_devices": "mlx5_0:1,mlx5_1:1", "node_ip": "192.0.2.1"}
    subprocess.run(["bash", "-c", harness], env=environment, check=True, capture_output=True, text=True)
    result = [json.loads(line.replace(str(root), "/fixture")) for line in record.read_text().splitlines()]
    return sorted(result, key=lambda item: item["argv"][item["argv"].index("--port") + 1])


def capture_router(source: str, platform: str, tmp_path: Path) -> list[str]:
    root = tmp_path / platform
    (root / "source").mkdir(parents=True)
    (root / "source/aspire-serving.json").write_bytes((REPOSITORY / "configs/aspire-serving.json").read_bytes())
    (root / "source/worker-endpoints.tsv").write_text("prefill\t192.0.2.1\t30100\t9000\nprefill\t192.0.2.1\t30101\t9001\ndecode\t192.0.2.2\t30200\tunused\ndecode\t192.0.2.2\t30201\tunused\n")
    function = source.split("run_router() {", 1)[1].split("\nrun_aiperf()", 1)[0]
    harness = "set -euo pipefail\nenroot() { printf '%s\\0' \"$@\" >\"${OUTPUT_DIR}/argv\"; }\nrun_router() {" + function + "\nrun_router\n"
    subprocess.run(["bash", "-c", harness], env={**os.environ, "QWEN_PLATFORM": platform, "OUTPUT_DIR": str(root), "MODEL_PATH": "/fixture/model", "CONTAINER_IMAGE": "/fixture/image.sqsh"}, check=True)
    return root.joinpath("argv").read_text().rstrip("\0").split("\0")


@pytest.mark.parametrize("platform,nodes,profile", [("aspire2ap", 2, False), ("firmus", 2, False), ("firmus", 2, True), ("genkai", 2, False), ("genkai", 4, False), ("genkai", 4, True)])
def test_worker_argv_matches_archived_base_with_graph_candidate(platform: str, nodes: int, profile: bool, tmp_path: Path) -> None:
    source = (REPOSITORY / "submission/run/submit.sbatch").read_text()
    expected = json.loads(FIXTURE.read_text())["workers"][f"{platform}-{nodes}-{profile}"]
    if platform == "aspire2ap":
        graph_bs = json.loads((REPOSITORY / "configs/aspire-serving.json").read_text())["decode_cuda_graph_bs"]
        for worker in expected:
            if worker["argv"][worker["argv"].index("--disaggregation-mode") + 1] == "decode":
                worker["argv"] += ["--cuda-graph-bs-decode", *map(str, graph_bs)]
            else:
                worker["argv"] += ["--enable-symm-mem"]
                image_index = worker["argv"].index("/fixture/image.sqsh")
                worker["argv"][image_index:image_index] = ["-e", "SGLANG_PREFILL_FUSED_FP8_SHUFFLE=1", "-e", "SGLANG_PREFILL_TILED_ROUTING=1"]
                if profile:
                    image_index = worker["argv"].index("/fixture/image.sqsh")
                    worker["argv"][image_index:image_index] = ["-e", "NCCL_DEBUG=INFO", "-e", "NCCL_DEBUG_SUBSYS=TUNING"]
    assert capture_workers(source, platform, nodes, tmp_path, profile) == expected


def test_graph_candidate_preserves_archived_buckets_and_client_condition() -> None:
    config = json.loads((REPOSITORY / "configs/aspire-serving.json").read_text())
    # Audited decode default at SGLang fc275674; only two missing buckets are added.
    archived_bs = [*range(1, 9), *range(10, 33, 2), *range(40, 65, 4), *range(72, 257, 8), *range(272, 513, 16)]
    assert config["decode_cuda_graph_bs"] == sorted([*archived_bs, 13, 15])
    assert config["concurrency"] == 48


@pytest.mark.parametrize("platform", ["aspire2ap", "firmus", "genkai"])
def test_router_argv_matches_candidate_or_main(platform: str, tmp_path: Path) -> None:
    source = (REPOSITORY / "submission/run/submit.sbatch").read_text()
    assert capture_router(source, platform, tmp_path) == json.loads(FIXTURE.read_text())["routers"][platform]


def test_candidate_flags_agree_with_archived_server_info() -> None:
    fixture = json.loads(FIXTURE.read_text())
    workers = fixture["workers"]["aspire2ap-2-False"]
    for worker in workers:
        argv = worker["argv"]
        port = argv[argv.index("--port") + 1]
        observed = fixture["archived_server_info"][port]
        for flag, key in {"--tp-size": "tp_size", "--ep-size": "ep_size", "--kv-cache-dtype": "kv_cache_dtype", "--nccl-port": "nccl_port", "--max-total-tokens": "max_total_tokens", "--speculative-algorithm": "speculative_algorithm", "--speculative-num-steps": "speculative_num_steps", "--speculative-eagle-topk": "speculative_eagle_topk", "--speculative-num-draft-tokens": "speculative_num_draft_tokens", "--disaggregation-mode": "disaggregation_mode"}.items():
            assert str(observed[key]) == argv[argv.index(flag) + 1]
        assert observed["moe_runner_backend"] == ("cutlass" if "--moe-runner-backend" in argv else "auto")
        assert observed["chunked_prefill_size"] == 8192
        assert observed["speculative_draft_model_quantization"] is None


def test_node_model_stage_resolves_shards_and_excludes_download_cache(tmp_path: Path) -> None:
    source = tmp_path / "model"
    source.mkdir()
    (source / ".cache").mkdir()
    (source / ".cache/download").write_text("unused")
    weights = source / "model-1.safetensors"
    weights.write_bytes(b"original weights")
    (source / "README.md").symlink_to(weights)
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "model-1.safetensors"}}))
    destination = tmp_path / "node-model"
    log = tmp_path / "copy.log"
    subprocess.run(["bash", str(REPOSITORY / "scripts/stage-qwen-node-model.sh"), str(source), str(destination), str(log)], check=True)
    assert (destination / "model-1.safetensors").read_bytes() == weights.read_bytes()
    assert not (destination / "README.md").is_symlink()
    assert (destination / "README.md").read_bytes() == weights.read_bytes()
    assert not (destination / ".cache").exists()
    assert "copy_verified=true" in log.read_text()


@pytest.mark.parametrize("wrong_dtype", [False, True])
def test_eagle_preparation_changes_only_config_and_records_weights(tmp_path: Path, wrong_dtype: bool) -> None:
    spec = importlib.util.spec_from_file_location("prepare_eagle", REPOSITORY / "research/simulation/prepare_eagle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"architectures": ["LlamaForCausalLM"], "hidden_size": 4096, "draft_vocab_size": 64000}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "README.md").write_text("pinned draft fixture")
    tensors = {"fc.weight": {"shape": [4096, 12288], "dtype": "FP8" if wrong_dtype else "BF16"}, "lm_head.weight": {"shape": [64000, 4096], "dtype": "BF16"}, "d2t": {"shape": [64000], "dtype": "I32"}, "t2d": {"shape": [151936], "dtype": "I32"}}
    header = json.dumps(tensors).encode()
    weights = struct.pack("<Q", len(header)) + header + b"fixture payload"
    (tmp_path / "model.safetensors").write_bytes(weights)
    if wrong_dtype:
        with pytest.raises(ValueError, match="retain original BF16"):
            module.prepare(tmp_path, tmp_path / "evidence")
        assert json.loads((tmp_path / "config.json").read_text()) == config
    else:
        module.prepare(tmp_path, tmp_path / "evidence")
        assert json.loads((tmp_path / "config.json").read_text()) == {**config, "architectures": ["LlamaForCausalLMEagle3"], "torch_dtype": "bfloat16"}
        provenance = json.loads((tmp_path / "evidence/draft-provenance.json").read_text())
        assert provenance["weight_sha256"] == hashlib.sha256(weights).hexdigest()
        assert json.loads((tmp_path / "evidence/draft-original-config.json").read_text()) == config
    assert (tmp_path / "model.safetensors").read_bytes() == weights


@pytest.mark.parametrize("state,status,success,expected", [("Q", 0, False, 0), ("R", 0, False, 0), ("F", 0, True, 0), ("F", 7, True, 1), ("F", 0, False, 1)])
def test_pbs_checker_requires_terminal_success_and_artifacts(tmp_path: Path, state: str, status: int, success: bool, expected: int) -> None:
    for directory in ("logs", "results"):
        (tmp_path / directory).mkdir()
    (tmp_path / "logs/pbs.log").write_text("fixture job log")
    for name in ("profile_export_aiperf.json", "score.json"):
        (tmp_path / "results" / name).write_text("{}")
    if success:
        (tmp_path / "run-success").touch()
    source = (REPOSITORY / "platforms/aspire2ap/check-qwen-job.sh").read_text()
    accounting = f"job_state = {state}\nExit_status = {status}\n"
    source = source.replace('/opt/pbs/bin/qstat -x -f "${PBS_JOB_ID}"', f"printf %s {shlex.quote(accounting)}")
    process = subprocess.run(["bash", "-c", source], env={**os.environ, "PBS_JOB_ID": "123.fixture", "RUNTIME_DIR": str(tmp_path)}, capture_output=True, text=True, check=False)
    assert process.returncode == expected, process.stderr
    assert ("scheduler_job_pending=" in process.stdout) == (state != "F")


def test_aiperf_argv_matches_archived_candidate(tmp_path: Path) -> None:
    source = (REPOSITORY / "submission/run/submit.sbatch").read_text()
    command = next(line for line in source.splitlines() if line.startswith("  command=("))
    recording = command + "\nprintf '%s\\0' \"${command[@]}\""
    process = subprocess.run(["bash", "-c", recording], env={**os.environ, "CONCURRENCY": "40"}, check=True, capture_output=True, text=True)
    assert process.stdout.rstrip("\0").split("\0") == json.loads(FIXTURE.read_text())["aiperf_argv"]


@pytest.mark.parametrize("platform,parser_status", [("aspire2ap", 0), ("aspire2ap", 3), ("firmus", 0), ("firmus", 3), ("genkai", 3)])
def test_parser_status_survives_post_benchmark_evidence(tmp_path: Path, platform: str, parser_status: int) -> None:
    for directory in ("source", "results/artifacts/run"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "source/worker-endpoints.tsv").write_text("prefill\t192.0.2.1\t30100\t9000\ndecode\t192.0.2.2\t30200\tunused\n")
    (tmp_path / "results/artifacts/run/profile_export_aiperf.json").write_text("{}")
    source = (REPOSITORY / "submission/run/submit.sbatch").read_text()
    function = source.split("run_aiperf() {", 1)[1].split('\nif [[ "${mode}" == node ]]', 1)[0]
    stubs = '''set -euo pipefail
enroot() {
  case " $* " in
    *" --require-eligible "*) return "${PARSER_STATUS}" ;;
    *" /calibration/scripts/greedy_probe.py "*) printf 'greedy\n' >"${OUTPUT_DIR}/greedy-invoked" ;;
  esac
}
curl() { printf '{}\n'; }
'''
    process = subprocess.run(["bash", "-c", stubs + "run_aiperf() {" + function + "\nrun_aiperf\n"], env={**os.environ, "QWEN_PLATFORM": platform, "OUTPUT_DIR": str(tmp_path), "CONCURRENCY": "40" if platform == "aspire2ap" else "16", "CONTAINER_IMAGE": "/fixture/image.sqsh", "PARSER_STATUS": str(parser_status)}, capture_output=True, text=True, check=False)
    assert process.returncode == parser_status, process.stderr
    assert (tmp_path / "greedy-invoked").exists() == (platform == "aspire2ap")
    assert len(list((tmp_path / "results").glob("server-info-*.json"))) == (2 if platform == "aspire2ap" else 0)


@pytest.mark.parametrize("missing_logprobs", [False, True])
def test_greedy_probe_records_fixed_requests_and_requires_logprobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_logprobs: bool) -> None:
    import io

    spec = importlib.util.spec_from_file_location("greedy_probe", REPOSITORY / "research/simulation/greedy_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    requests = []

    def response(request, timeout):
        assert request.full_url == "http://127.0.0.1:8000/v1/chat/completions"
        assert timeout == 120
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"logprobs": {"content": [] if missing_logprobs else [{"token": "fixture", "logprob": -0.1}]}}]}).encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", response)
    monkeypatch.setattr(sys, "argv", ["greedy_probe.py", str(tmp_path)])
    if missing_logprobs:
        with pytest.raises(ValueError, match="token logprobs required"):
            module.main()
        assert not (tmp_path / "greedy-parity.json").exists()
    else:
        module.main()
        result = json.loads((tmp_path / "greedy-parity.json").read_text())
        assert len(result["responses"]) == len(requests) == 8
        assert result["phase"] == "after official benchmark"
        assert not result["training_data"]
    for request in requests:
        assert request["temperature"] == 0
        assert request["min_tokens"] == request["max_tokens"] == 64
        assert request["ignore_eos"] and request["logprobs"]
        assert not request["stream"]
