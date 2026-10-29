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
ASPIRE_HCAS = "mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1"


@pytest.mark.parametrize("corruption", [None, "layer_count", "duplicate_expert", "hash", "replica_owner", "overlay_mount", "overlay_hash", "overlay_image"])
def test_runtime_mount_preflight_rejects_bad_placement(tmp_path: Path, corruption: str | None) -> None:
    for directory in ("runtime/source", "model", "draft", "runtime-home"):
        (tmp_path / directory).mkdir(parents=True)
    config = json.loads((REPOSITORY / "configs/aspire-serving.json").read_text())
    mapping = json.loads((REPOSITORY / config["prefill_expert_location"]).read_text())["physical_to_logical_map"]
    if corruption == "layer_count":
        mapping.pop()
    elif corruption == "duplicate_expert":
        mapping[0][4] = mapping[0][0]
    elif corruption == "replica_owner":
        # Preserve global counts but move a hot replica to the wrong GPU.
        mapping[0][0], mapping[0][39] = mapping[0][39], mapping[0][0]
    placement = json.dumps({"physical_to_logical_map": mapping}).encode()
    (tmp_path / "runtime/source/prefill-expert-location.json").write_bytes(placement)
    for name, prefix, mount in (("sglang", "python/sglang", "opt/sglang/source/python/sglang"), ("deep-gemm", "deep_gemm", "opt/sglang/venv/lib/python3.12/site-packages/deep_gemm")):
        path = f"patches/{name}/fixture.patch"
        config[f"{name.replace('-', '_')}_overlay_patches"] = [path]
        content = f"+++ b/{prefix}/kernel.py\n".encode()
        patches = tmp_path / f"runtime/source/{name}-overlay"
        patches.mkdir()
        (patches / "fixture.patch").write_bytes(content)
        record = {"base_image_sha256": "wrong" if corruption == "overlay_image" else config["container_sha256"], "mount": "/" + mount, "patches": [{"path": path, "sha256": "wrong" if corruption == "overlay_hash" else hashlib.sha256(content).hexdigest()}]}
        (tmp_path / f"runtime/source/{name}-overlay.json").write_text(json.dumps(record))
        overlay = tmp_path / f"runtime/overlay/{name}/{prefix}"
        overlay.mkdir(parents=True)
        (overlay / "kernel.py").write_text("patched")
        mounted = tmp_path / mount
        mounted.mkdir(parents=True)
        (mounted / "kernel.py").write_text("old" if corruption == "overlay_mount" else "patched")
    (tmp_path / "runtime/source/aspire-serving.json").write_text(json.dumps(config))
    (tmp_path / "model/config.json").write_text('{"num_hidden_layers":94,"num_experts":128}')
    (tmp_path / "draft/config.json").write_text('{"architectures":["LlamaForCausalLMEagle3"],"torch_dtype":"bfloat16"}')
    (tmp_path / "draft/model.safetensors").write_bytes(b"fixture")
    expected = "wrong-hash" if corruption == "hash" else hashlib.sha256(placement).hexdigest()
    source = (REPOSITORY / "scripts/check-aspire-runtime-mount.sh").read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    source = source.replace("from pathlib import Path", "from pathlib import Path as RealPath\nPath = lambda value: RealPath(sys.argv[2]) / value.lstrip('/')")
    result = subprocess.run([sys.executable, "-c", source, expected, str(tmp_path)], capture_output=True, text=True, check=False)
    if corruption is None:
        assert result.returncode == 0, result.stderr
        assert "aspire_runtime_mount_ok" in result.stdout
    else:
        assert result.returncode != 0
        assert "AssertionError" in result.stderr


def capture_workers(source: str, platform: str, nodes: int, tmp_path: Path, profile: bool = False) -> list[dict]:
    """Execute the real worker function and topology with a recording Enroot stub."""
    root = tmp_path / f"{platform}-{nodes}-{profile}"
    root.mkdir(parents=True, exist_ok=True)
    output = root / "output"
    for directory in ("logs", "source", "scripts"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    (output / "source/aspire-serving.json").write_bytes((REPOSITORY / "configs/aspire-serving.json").read_bytes())
    if platform == "aspire2ap":
        # Execute the workflow's real staging commands, then exercise the container mount below.
        lines = (REPOSITORY / ".github/workflows/aspire2ap.yml").read_text().splitlines()
        start = next(i for i, line in enumerate(lines) if line.strip().startswith('prefill_location='))
        subprocess.run(["bash", "-c", "set -euo pipefail\n" + "\n".join(lines[start:start + 2])], cwd=REPOSITORY, env={**os.environ, "GITHUB_WORKSPACE": str(REPOSITORY), "runtime_dir": str(output)}, check=True)
        config = json.loads((output / "source/aspire-serving.json").read_text())
        for directory in ("sglang/python/sglang", "deep-gemm/deep_gemm"):
            (output / "overlay" / directory).mkdir(parents=True)
        assert (output / "source/prefill-expert-location.json").read_bytes() == (REPOSITORY / config["prefill_expert_location"]).read_bytes()
    record = root / "commands.jsonl"
    record.write_text("")
    stub = root / "enroot"
    stub.write_text(f"#!{sys.executable}\nimport json, os, sys\nfd=os.open(os.environ['ARGV_RECORD'], os.O_WRONLY|os.O_APPEND)\nos.write(fd, (json.dumps({{'argv':sys.argv[1:], 'profiler':os.environ.get('QWEN_PROFILER_INVOCATION_PATH')}})+'\\n').encode())\nos.close(fd)\n")
    stub.chmod(0o755)
    profiler = output / "scripts/exec-profiler.sh"
    profiler.write_text('#!/bin/bash\nexec "$@"\n')
    profiler.chmod(0o755)
    worker = source.split("  start_worker() {", 1)[1].split("\n  case \"${QWEN_PLATFORM}\"", 1)[0]
    topology = source.split("\n  case \"${QWEN_PLATFORM}\"", 1)[1].split("\n  esac", 1)[0]
    harness = 'set -euo pipefail\nrestore_worker_cache() { mkdir -p "$2"; }\npids=()\ncache_workers=()\nstart_worker() {' + worker + '\nfor ((node_rank=0;node_rank<NODE_COUNT;node_rank++)); do\ncase "${QWEN_PLATFORM}"' + topology + '\nesac\ndone\nfor pid in "${pids[@]}"; do wait "$pid"; done\n'
    environment = {**os.environ, "PATH": f"{root}:{os.environ['PATH']}", "ARGV_RECORD": str(record), "QWEN_PLATFORM": platform, "NODE_COUNT": str(nodes), "GPUS_PER_NODE": "4" if platform == "genkai" else "8", "PROFILE": "true" if profile else "false", "OUTPUT_DIR": str(output), "node_tmp": str(root / "node"), "cache_root": str(root / "cache"), "MODEL_PATH": "/fixture/model", "CONTAINER_IMAGE": "/fixture/image.sqsh", "ucx_devices": "mlx5_0:1,mlx5_1:1", "node_ip": "192.0.2.1"}
    if platform == "aspire2ap":
        environment["ucx_devices"] = ASPIRE_HCAS
    subprocess.run(["bash", "-c", harness], env=environment, check=True, capture_output=True, text=True)
    result = [json.loads(line.replace(str(root), "/fixture")) for line in record.read_text().splitlines()]
    return sorted(result, key=lambda item: item["argv"][item["argv"].index("--port") + 1])


def capture_router(source: str, platform: str, tmp_path: Path) -> list[str]:
    root = tmp_path / platform
    (root / "source").mkdir(parents=True)
    (root / "source/aspire-serving.json").write_bytes((REPOSITORY / "configs/aspire-serving.json").read_bytes())
    endpoints = "prefill\t192.0.2.1\t30100\t9000\nprefill\t192.0.2.1\t30101\t9001\ndecode\t192.0.2.2\t30200\tunused\ndecode\t192.0.2.2\t30201\tunused\n"
    if platform == "aspire2ap":
        endpoints = "prefill\t192.0.2.1\t30100\t9000\nprefill\t192.0.2.1\t30101\t9001\nprefill\t192.0.2.2\t30102\t9002\ndecode\t192.0.2.2\t30200\tunused\n"
    (root / "source/worker-endpoints.tsv").write_text(endpoints)
    function = source.split("run_router() {", 1)[1].split("\nrun_aiperf()", 1)[0]
    harness = "set -euo pipefail\nenroot() { printf '%s\\0' \"$@\" >\"${OUTPUT_DIR}/argv\"; }\nrun_router() {" + function + "\nrun_router\n"
    subprocess.run(["bash", "-c", harness], env={**os.environ, "QWEN_PLATFORM": platform, "OUTPUT_DIR": str(root), "MODEL_PATH": "/fixture/model", "CONTAINER_IMAGE": "/fixture/image.sqsh"}, check=True)
    return root.joinpath("argv").read_text().rstrip("\0").split("\0")


@pytest.mark.parametrize("platform,nodes,profile", [("firmus", 2, False), ("firmus", 2, True), ("genkai", 2, False), ("genkai", 4, False), ("genkai", 4, True)])
def test_worker_argv_matches_validated_candidate_or_main(platform: str, nodes: int, profile: bool, tmp_path: Path) -> None:
    source = (REPOSITORY / "platforms/shared/qwen-pd.sbatch").read_text()
    expected = json.loads(FIXTURE.read_text())["workers"][f"{platform}-{nodes}-{profile}"]
    assert capture_workers(source, platform, nodes, tmp_path, profile) == expected


@pytest.mark.parametrize("platform", ["aspire2ap", "firmus", "genkai"])
def test_router_argv_matches_candidate_or_main(platform: str, tmp_path: Path) -> None:
    source = (REPOSITORY / "platforms/shared/qwen-pd.sbatch").read_text()
    expected = json.loads(FIXTURE.read_text())["routers"][platform]
    if platform == "aspire2ap":
        start = expected.index("/opt/sglang/venv/bin/python")
        expected[start:start + 3] = ["/opt/apac/bin/sglang-router", "launch"]
        index = expected.index("--pd-disaggregation") + 1
        expected[index:index] = ["--assignment-mode", "min_load"]
        expected[expected.index("--prefill-policy") + 1] = "manual"
        start = expected.index("--decode")
        end = expected.index("--host")
        expected[start:end] = ["--prefill", "http://192.0.2.2:30102", "9002", "--decode", "http://192.0.2.2:30200"]
    assert capture_router(source, platform, tmp_path) == expected


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
    source = (REPOSITORY / "platforms/shared/qwen-pd.sbatch").read_text()
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
    source = (REPOSITORY / "platforms/shared/qwen-pd.sbatch").read_text()
    function = source.split("run_aiperf() {", 1)[1].split('\nif [[ "${mode}" == kernels ]]', 1)[0]
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


@pytest.mark.parametrize("profile", [False, True])
def test_aspire_megamoe_uses_two_prefill_and_two_decode_workers(tmp_path: Path, profile: bool) -> None:
    source = (REPOSITORY / "platforms/shared/qwen-pd.sbatch").read_text()
    config = json.loads((REPOSITORY / "configs/aspire-serving.json").read_text())
    assert config["concurrency"] == 104
    assert [(w["node"], w["role"], w["gpus"]) for w in config["workers"]] == [(0, "prefill", "0,1,2,3"), (0, "prefill", "4,5,6,7"), (1, "decode", "0,1,2,3"), (1, "decode", "4,5,6,7")]
    assert config["prefill_load_balance_method"] == "total_requests" and config["decode_load_balance_method"] == "total_requests"
    workers = capture_workers(source, "aspire2ap", 2, tmp_path, profile)
    assert len(workers) == 4
    addresses = []
    for record in workers:
        argv = record["argv"]
        role = argv[argv.index("--disaggregation-mode") + 1]
        expected = {"--model-path": "/model", "--served-model-name": "qwen", "--context-length": "8192", "--host": "0.0.0.0", "--disaggregation-transfer-backend": "nixl", "--speculative-draft-model-path": "/draft", "--speculative-eagle-topk": "1", "--tp-size": "4", "--ep-size": "4", "--dp-size": "4", "--moe-a2a-backend": "megamoe", "--moe-runner-backend": "deep_gemm", "--attention-backend": "fa3", "--kv-cache-dtype": "fp8_e4m3", "--page-size": "64", "--chunked-prefill-size": "16384" if role == "prefill" else "8192", "--max-running-requests": "104" if role == "prefill" else "80", "--max-total-tokens": "16384" if role == "prefill" else "106496", "--mem-fraction-static": "0.9" if role == "prefill" else "0.96", "--speculative-algorithm": "EAGLE3", "--speculative-num-steps": "3", "--speculative-num-draft-tokens": "4", "--load-balance-method": "total_requests", "--load-snapshot-publish-interval": "1"}
        for flag, value in expected.items():
            assert argv.count(flag) == 1
            assert argv[argv.index(flag) + 1] == value
        assert "--trust-remote-code" in argv
        assert "--enable-dp-attention" in argv
        assert "UCX_ADAPTIVE_PROGRESS=n" in argv
        assert "UCX_LOG_LEVEL=info" in argv
        assert "UCX_POSIX_USE_PROC_LINK=n" in argv
        assert "HF_HUB_OFFLINE=1" in argv and "TRANSFORMERS_OFFLINE=1" in argv
        assert "SGLANG_HOST_IP=192.0.2.1" in argv
        assert f"UCX_NET_DEVICES={ASPIRE_HCAS}" in argv
        assert f"NCCL_IB_HCA=={ASPIRE_HCAS}" in argv
        assert "SGLANG_DISAGG_STAGING_BUFFER=0" in argv
        assert "SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1" in argv
        assert "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096" in argv
        assert ("--disable-cuda-graph" in argv) == (role == "prefill")
        assert ("--enable-dp-lm-head" in argv) == (role == "decode")
        # Explicit read-only mounts must retain bind and target creation (Enroot fstab semantics).
        placement_mount = "/fixture/output:/runtime:none:bind,ro,x-create=dir"
        assert placement_mount in argv
        assert ("--init-expert-location" in argv) == (role == "prefill")
        if role == "prefill":
            assert argv[argv.index("--init-expert-location") + 1] == "/runtime/source/prefill-expert-location.json"
            assert argv[argv.index("--deepep-mode") + 1] == "normal"
        for variable in ("DG_SM90_PREFILL_2WG=1", "SGLANG_PREFILL_REQUEST_ALIGNED_CHUNKS=1", "SGLANG_DP_TOTAL_REQUESTS_ROTATE_TIES=1"):
            assert (variable in argv) == (role == "prefill")
        assert "SGLANG_DECODE_ASYNC_ADMISSION=1" in argv
        assert "SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY=1" in argv
        assert ("DG_SM90_DECODE_INDEPENDENT_WG=1" in argv) == (role == "decode")
        assert f"/runtime/source/{role}-kernels.env" in argv
        assert f"/runtime/source/{'decode' if role == 'prefill' else 'prefill'}-kernels.env" not in argv
        assert "/fixture/output/overlay/sglang/python/sglang:/opt/sglang/source/python/sglang:none:bind,ro,x-create=dir" in argv
        assert "/fixture/output/overlay/deep-gemm/deep_gemm:/opt/sglang/venv/lib/python3.12/site-packages/deep_gemm:none:bind,ro,x-create=dir" in argv
        if role == "prefill":
            assert argv[argv.index("--ep-num-redundant-experts") + 1] == "12"
            assert argv[argv.index("--optimistic-prefill-attempts") + 1] == "2"
        else:
            assert "--ep-num-redundant-experts" not in argv
            assert "--optimistic-prefill-attempts" not in argv
        if role == "decode":
            start = argv.index("--cuda-graph-bs-decode") + 1
            assert argv[start:start + len(config["decode_cuda_graph_bs"])] == [str(n) for n in config["decode_cuda_graph_bs"]]
            assert argv[argv.index("--moe-dense-tp-size") + 1] == "1"
            assert argv[argv.index("--disaggregation-decode-extra-slots") + 1] == "2"
            assert argv[argv.index("--disaggregation-decode-polling-interval") + 1] == "1"
        addresses.append(int(argv[argv.index("--dist-init-addr") + 1].split(":")[1]))
        assert bool(record["profiler"]) == profile
    # Separate DP endpoint ranges; adjacent HTTP ports otherwise collide.
    assert addresses == [32000, 32100, 33000, 33100]
    firmus = json.loads((REPOSITORY / "configs/firmus-framework.json").read_text())
    assert config["container_sha256"] == firmus["images"]["sglang"]["sha256"] == "1f247668933c7252d3dfbe8f6c23897d354f095d56842788d4415a85d7dc5e55"
    assert config["prefill_expert_location"] == "configs/firmus-prefill-replica4-expert-location.json"
    # Firmus-only: Prefill start alignment and the in-job selected Decode-step candidates.
    assert firmus["sglang_overlay_patches"] == config["sglang_overlay_patches"] + ["patches/sglang/prefill-start-align.patch"] + ["patches/sglang/decode-norm-quant-wide.patch", "patches/sglang/decode-dense-swapab.patch", "patches/sglang/verify-penalty-broadcast.patch"]
    assert len(config["deep_gemm_overlay_patches"]) == 7
    assert firmus["deep_gemm_overlay_patches"] == config["deep_gemm_overlay_patches"] + ["patches/deep-gemm-bench/fc2-address/fc2-output-address.patch"]
    workflow = (REPOSITORY / ".github/workflows/aspire2ap.yml").read_text()
    assert workflow.index("jq -er .container_sha256 configs/aspire-serving.json") < workflow.index('PROVISION_OUTPUT="${GITHUB_OUTPUT}" scripts/provision-qwen-pd.sh')


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_aspire_effective_settings_reject_wrong_placement_and_clipped_kv_pool(role: str) -> None:
    import runpy

    check = runpy.run_path(str(REPOSITORY / "scripts/check-firmus-server-info.py"))["check"]
    config = json.loads((REPOSITORY / "configs/aspire-serving.json").read_text())
    # Reuse the real endpoint schema; these candidate values are synthetic, not an H100 run.
    info = json.loads((REPOSITORY / "qwen-experiment-metrics/tests/fixtures/firmus-server-info-38369214512.json").read_text())[role]
    capacity = config[f"{role}_max_total_tokens"]
    for state in [info, *info["internal_states"]]:
        state.update(max_total_tokens=capacity, max_running_requests=config["concurrency"] if role == "prefill" else config["decode_max_running_requests"], mem_fraction_static=0.90 if role == "prefill" else config["decode_mem_fraction_static"], chunked_prefill_size=4096 if role == "prefill" else 2048, moe_a2a_backend="megamoe", moe_runner_backend="deep_gemm", disaggregation_decode_extra_slots=2, load_balance_method=config[f"{role}_load_balance_method"], disable_cuda_graph=role == "prefill", effective_max_running_requests_per_dp=config["decode_max_running_requests"] // config["dp_size"], memory_usage={"token_capacity": capacity})
        state.update(init_expert_location="/runtime/source/prefill-expert-location.json" if role == "prefill" else "trivial", ep_dispatch_algorithm="static" if role == "prefill" else None)
        state.update(ep_num_redundant_experts=12 if role == "prefill" else 0, enable_eplb=False, optimistic_prefill_attempts=2 if role == "prefill" else 0)
    info["cuda_graph_bs_decode"] = config["decode_cuda_graph_bs"]
    check(config, info, role)
    for state in [info, *info["internal_states"]]:
        for key, invalid in (("init_expert_location", "trivial" if role == "prefill" else "/runtime/source/prefill-expert-location.json"), ("ep_dispatch_algorithm", None if role == "prefill" else "static"), ("ep_num_redundant_experts", 0 if role == "prefill" else 12), ("enable_eplb", True)):
            original = state[key]
            state[key] = invalid
            with pytest.raises(ValueError, match=key):
                check(config, info, role)
            state[key] = original
    for state in info["internal_states"]:
        state["memory_usage"]["token_capacity"] -= 64
        with pytest.raises(ValueError, match="actual KV capacity"):
            check(config, info, role)
        state["memory_usage"]["token_capacity"] += 64


@pytest.mark.parametrize("fault", [None, "missing", "down", "ethernet"])
def test_aspire_excludes_active_storage_nics_and_requires_compute_fabric(tmp_path: Path, fault: str | None) -> None:
    for index in range(12):
        port = tmp_path / f"mlx5_{index}/ports/1"
        port.mkdir(parents=True)
        (port / "state").write_text("4: ACTIVE\n")
        (port / "link_layer").write_text("InfiniBand\n")
    port = tmp_path / "mlx5_3/ports/1"
    if fault == "missing":
        (port / "state").unlink()
    elif fault == "down":
        (port / "state").write_text("1: DOWN\n")
    elif fault == "ethernet":
        (port / "link_layer").write_text("Ethernet\n")
    source = (REPOSITORY / "platforms/shared/qwen-pd.sbatch").read_text()
    selection = source.split("  hca_devices=()", 1)[1].split("  printf 'hostname=", 1)[0]
    selection = selection.replace("/sys/class/infiniband", str(tmp_path))
    process = subprocess.run(["bash", "-c", "set -euo pipefail\nhca_devices=()\n" + selection + '\nprintf "%s" "$ucx_devices"'], env={**os.environ, "QWEN_PLATFORM": "aspire2ap"}, capture_output=True, text=True, check=False)
    if fault:
        assert process.returncode != 0
        assert process.stdout == ""
    else:
        assert process.returncode == 0, process.stderr
        assert process.stdout == ASPIRE_HCAS


def test_decode_kv_pool_covers_full_batch_growth_and_transfer_slots() -> None:
    config = json.loads((REPOSITORY / "configs/aspire-serving.json").read_text())
    running = config["decode_max_running_requests"] // config["dp_size"]
    # Official ~4K input plus page-rounded 200-token output; retain native 512-token reserve.
    required = running * (4096 + 256 + 512) + config["decode_extra_slots"] * (4096 + 512)
    assert config["decode_max_total_tokens"] >= required
    assert max(config["decode_cuda_graph_bs"]) == running


def test_aspire_memory_override_does_not_change_other_platforms(tmp_path: Path) -> None:
    source = (REPOSITORY / "platforms/shared/qwen-pd.sbatch").read_text()
    # Use a distinctive value to prove this setting remains inside the ASPIRE branch.
    source = source.replace('jq -er .prefill_mem_fraction_static "${config}"', 'printf 0.901')
    for platform in ("aspire2ap", "firmus", "genkai"):
        for record in capture_workers(source, platform, 2, tmp_path):
            argv = record["argv"]
            role = argv[argv.index("--disaggregation-mode") + 1]
            expected = "0.901" if platform == "aspire2ap" and role == "prefill" else "0.96" if platform == "aspire2ap" else "0.90"
            assert argv[argv.index("--mem-fraction-static") + 1] == expected


@pytest.mark.parametrize("gate_status", [0, 23])
def test_aspire_in_job_screens_cover_all_prefill_gpus_and_propagate_gate_failure(tmp_path: Path, gate_status: int) -> None:
    import runpy

    root = tmp_path / "run"
    for directory in ("scripts", "source", "logs"):
        (root / directory).mkdir(parents=True)
    (root / "source/aspire-serving.json").write_bytes((REPOSITORY / "configs/aspire-serving.json").read_bytes())
    (root / "source/prefill-candidates.json").write_bytes((REPOSITORY / "kernels/fa3-verify/prefill-candidates.json").read_bytes())
    (root / "scripts/select-firmus-prefill-fa3.py").write_bytes((REPOSITORY / "scripts/select-firmus-prefill-fa3.py").read_bytes())
    selector = runpy.run_path(str(root / "scripts/select-firmus-prefill-fa3.py"))
    ranks = [{"rank": rank, "lengths": {str(length): {name: {"numeric_passed": True, "control": [350.0] * 20, "candidate": [344.0] * 20} for name in selector["NAMES"][1:]} for length in selector["LENGTHS"]}} for rank in range(4)]
    (tmp_path / "fragment.json").write_text(json.dumps({"ranks": ranks}))
    stub = tmp_path / "enroot"
    stub.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ["SCREEN_ROOT"])
record = {{"argv": args}}
fd = os.open(os.environ["SCREEN_RECORD"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
os.write(fd, (json.dumps(record) + "\\n").encode())
os.close(fd)
if "/runtime/scripts/select-firmus-prefill-kernels.sh" in args:
    out = root / "results/prefill-selection" / args[-1]
    out.mkdir(parents=True)
    (out / "fa3.json").write_text(Path(os.environ["SCREEN_FRAGMENT"]).read_text())
    (out / "status.txt").write_text("fa3=0\\n")
else:
    assert args[-2:] == ["--dense-selected", "--fa3-selected"]
    sys.exit(int(os.environ["GATE_STATUS"]))
''')
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "CONTAINER_IMAGE": "/fixture/image.sqsh", "SCREEN_ROOT": str(root), "SCREEN_RECORD": str(tmp_path / "screens.jsonl"), "SCREEN_FRAGMENT": str(tmp_path / "fragment.json"), "GATE_STATUS": str(gate_status)}
    command = ["bash", str(REPOSITORY / "scripts/select-aspire-kernels.sh"), str(root)]
    subprocess.run([*command, "0"], env=env, check=True)
    selection = json.loads((root / "results/prefill-selection/selection.json").read_text())
    assert selection["rule"]["gpus"] == 8
    assert selection["winner"] == "regs"
    assert (root / "source/prefill-kernels.env").read_text() == "SGLANG_PREFILL_FA3_VERIFY=regs\n"
    result = subprocess.run([*command, "1"], env=env, check=False)
    assert result.returncode == gate_status
    assert (root / "source/decode-kernels.env").exists() == (gate_status == 0)
    if gate_status == 0:
        assert (root / "source/decode-kernels.env").read_bytes() == b""
    commands = [json.loads(line)["argv"] for line in (tmp_path / "screens.jsonl").read_text().splitlines()]
    prefill = [args for args in commands if "/runtime/scripts/select-firmus-prefill-kernels.sh" in args]
    assert {next(a for a in args if a.startswith("CUDA_VISIBLE_DEVICES=")) for args in prefill} == {"CUDA_VISIBLE_DEVICES=0,1,2,3", "CUDA_VISIBLE_DEVICES=4,5,6,7"}
    assert {args[-1] for args in prefill} == {"prefill-0", "prefill-1"}
    for args in commands:
        assert "SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY=1" in args
        assert any(":/runtime:none:bind,rw" in arg for arg in args)
    submission = (REPOSITORY / "platforms/shared/qwen-pd.sbatch").read_text()
    barrier = submission.index('"${script_path}" kernels')
    assert barrier < submission.index('"${script_path}" node')
    assert "#PBS -l walltime=00:30:00" in (REPOSITORY / "platforms/aspire2ap/run-qwen-pd.pbs").read_text()
    provision = (REPOSITORY / "scripts/provision-qwen-pd.sh").read_text()
    for filename in ("select-aspire-kernels.sh", "select-firmus-prefill-kernels.sh", "select-firmus-prefill-fa3.py", "check-firmus-megamoe.py", "check-firmus-decode-kernels.py", "firmus_megamoe_replicas.py", "prefill-candidates.json", "firmus-framework.json"):
        assert filename in provision


@pytest.mark.parametrize("require_build", [False, True])
def test_aspire_container_pin_selects_staged_build_without_history_search(tmp_path: Path, require_build: bool) -> None:
    content = b"pinned image fixture"
    config = json.loads((REPOSITORY / "configs/aspire-serving.json").read_text())
    assert config["container_build_run_id"] == 40197369083
    assert config["container_size_bytes"] == 12272513024
    config.update(container_sha256=hashlib.sha256(content).hexdigest(), container_size_bytes=len(content))
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "40197369083.sqsh").write_bytes(content)
    (tmp_path / "40197369083.container-sidecar.json").write_text(json.dumps({"schema_version": "container-sidecar/v1", "container_artifact": {"artifact_relative_path": "sglang.sqsh", "sha256": config["container_sha256"], "size_bytes": len(content)}}))
    # Any history query would fail this fixture. Pin selection never invokes gh.
    (tmp_path / "gh").write_text("#!/bin/bash\nexit 99\n")
    (tmp_path / "gh").chmod(0o755)
    output = tmp_path / "output"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "CONTAINER_ARCHIVE_DIR": str(tmp_path), "GITHUB_OUTPUT": str(output), "GITHUB_RUN_ID": "123", "GITHUB_SHA": "unused", "REQUIRE_BUILD": str(require_build).lower()}
    subprocess.run(["bash", str(REPOSITORY / "scripts/select-qwen-container.sh"), str(tmp_path / "config.json")], env=env, check=True)
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["action"] == ("build" if require_build else "reuse")
    if not require_build:
        assert values["source_run_id"] == "40197369083"
        assert values["container_sha256"] == config["container_sha256"]
    container = (REPOSITORY / ".github/workflows/aws-container.yml").read_text()
    assert "run: scripts/select-qwen-container.sh configs/aspire-serving.json" in container
    for key in ("sha256", "size"):
        output_expression = container.split(f"      aspire2ap_container_{key}:\n", 1)[1].splitlines()[0]
        assert "jobs.select-genkai" not in output_expression and "jobs.select-firmus" not in output_expression
    entry = (REPOSITORY / ".github/workflows/qwen.yml").read_text()
    for cluster in ("genkai", "firmus"):
        section = next(line for line in entry.split(f"\n  {cluster}:\n", 1)[1].splitlines() if line.startswith("    if:"))
        assert f"contains(github.event.head_commit.message, '[cluster:{cluster}]')" in section
        assert "[cluster:aspire2ap]" not in section


def test_aspire_worker_sources_selection_in_only_its_own_role(tmp_path: Path) -> None:
    source = (REPOSITORY / "platforms/shared/qwen-pd.sbatch").read_text()
    selected = tmp_path / "prefill-kernels.env"
    selected.write_text("SGLANG_PREFILL_FA3_VERIFY=regs\n")
    empty = tmp_path / "decode-kernels.env"
    empty.touch()
    for record in capture_workers(source, "aspire2ap", 2, tmp_path):
        argv = record["argv"]
        role = argv[argv.index("--disaggregation-mode") + 1]
        start = argv.index("-c")
        wrapper = argv[start + 1]
        assert argv[start + 3] == f"/runtime/source/{role}-kernels.env"
        env = {key: value for key, value in os.environ.items() if key != "SGLANG_PREFILL_FA3_VERIFY"}
        result = subprocess.run(["bash", "-c", wrapper, "bash", str(selected if role == "prefill" else empty), sys.executable, "-c", "import os; print(os.environ.get('SGLANG_PREFILL_FA3_VERIFY', 'stock'))"], env=env, check=True, text=True, capture_output=True)
        assert result.stdout.strip() == ("regs" if role == "prefill" else "stock")
