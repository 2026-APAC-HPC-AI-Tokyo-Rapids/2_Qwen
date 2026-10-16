from __future__ import annotations

import copy
import json
import os
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PARSE = runpy.run_path(str(ROOT / "scripts/parse-firmus-reference-result.py"))["parse_reference"]


def fixture() -> dict:
    return json.loads((ROOT / "qwen-experiment-metrics/tests/fixtures/firmus-reference.json").read_text())


def test_real_reference_run_excludes_warmup_and_checks_joint_sla() -> None:
    raw = fixture()
    result = PARSE(raw["export"], raw["records"], 8)
    assert result["request_count"] == 85
    assert result["good_request_count"] == 78
    assert result["good_request_fraction"] == pytest.approx(78 / 85)
    assert result["goodput_requests_per_second"] == pytest.approx(78 / 62.862709504)
    assert result["official_submission"] is False
    assert result["official_validation"]["valid"] is False


@pytest.mark.parametrize("corruption", ["duration", "phase", "records", "tokenizer", "errors", "tokens", "latency"])
def test_rejects_incomplete_or_changed_reference_workload(corruption: str) -> None:
    raw = copy.deepcopy(fixture())
    export, records = raw["export"], raw["records"]
    if corruption == "duration":
        export["benchmark_duration"]["avg"] = 20
    elif corruption == "phase":
        export["input_config"]["phases"][0]["duration"] = 60
    elif corruption == "records":
        records.pop()
    elif corruption == "tokenizer":
        export["input_config"]["tokenizer"]["name"] = "unverified"
    elif corruption == "errors":
        export["error_summary"] = ["request failed"]
    else:
        measured = next(row for row in records if row["metadata"]["benchmark_phase"] == "profiling")
        if corruption == "tokens":
            measured["metrics"]["output_sequence_length"]["value"] = 100
        else:
            measured["metrics"]["time_to_first_token"]["value"] = float("nan")
    with pytest.raises(ValueError):
        PARSE(export, records, 8)


def test_benchmark_completion_does_not_claim_sla_success() -> None:
    raw = fixture()
    for row in raw["records"]:
        row["metrics"]["time_to_first_token"]["value"] = 4000
    result = PARSE(raw["export"], raw["records"], 8)
    assert result["reference_validation"]["valid"] is True
    assert result["eligible"] is False
    assert result["score_output_tokens_per_second"] == 0


def test_sglang_launch_matches_full_h200_tp2_conditions(tmp_path: Path) -> None:
    source = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    config["concurrency"] = 64  # The retained SGLang control launcher uses its original c64 layout.
    config["speculation"].update(steps=3, draft_tokens=4)
    assert config["workload"] == "official"
    (tmp_path / "logs").mkdir()
    (tmp_path / "source").mkdir()
    (tmp_path / "source/firmus-framework.json").write_text(json.dumps(config))
    record = tmp_path / "commands.jsonl"
    stub = tmp_path / "srun"
    stub.write_text(f"#!{sys.executable}\nimport json,os,sys\nfd=os.open(os.environ['RECORD'],os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)\nos.write(fd,(json.dumps(sys.argv[1:])+'\\n').encode())\nos.close(fd)\n")
    stub.chmod(0o755)
    launches = "srun --exact" + source.split("srun --exact", 1)[1].split("\nwait_http", 1)[0]
    router = next(line for line in source.splitlines() if line.startswith("srun ") and "sglang_router.launch_router" in line)
    client = next(line for line in source.splitlines() if line.startswith("srun ") and "official-aiperf.sh" in line)
    configuration = 'speculation=(' + source.split('speculation=(', 1)[1].split('\nART=', 1)[0]
    harness = 'set -euo pipefail\nmounts() { printf /fixture; }\nworker_pids=()\n' + configuration + '\n' + launches + '\nfor pid in "${worker_pids[@]}"; do wait "$pid"; done\n' + router + '\nwait "$!"\n' + client
    environment = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "RECORD": str(record), "ROOT": str(tmp_path), "ART": str(tmp_path / "logs"), "HEAD": "node0", "TAIL": "node1", "HEAD_IP": "192.0.2.1", "TAIL_IP": "192.0.2.2", "SGLANG": "/image", "AIPERF": "/image", "MODEL": "/model", "N": "64", "R_PORT": "8000"}
    subprocess.run(["bash", "-c", harness], env=environment, check=True)
    commands = [json.loads(line) for line in record.read_text().splitlines()]
    workers = [argv for argv in commands if "--exact" in argv]
    assert len(workers) == 6
    assert sum(2 if "--gres=gpu:2" in argv else 4 for argv in workers) == 16
    archived = json.loads((ROOT / "qwen-experiment-metrics/tests/fixtures/aspire-integration-argv.json").read_text())["archived_server_info"]
    worker = source.split("<<'EOF'\n", 1)[1].split("\nEOF", 1)[0]
    assert "jq" not in worker  # Not installed in the pinned image; parse on the host.
    launch = worker.split("UCX_IB=", 1)[0] + 'printf "%s\\0" "${SGLANG_PREFILL_FUSED_FP8_SHUFFLE:-0}" "${SGLANG_PREFILL_TILED_ROUTING:-0}" "$SGLANG_HOST_IP"\n' + "printf '%s\\0' " + worker.split("\nexec ", 1)[1]
    observed = {}
    for argv in workers:
        args = argv[argv.index("bash") + 2:]
        probe = subprocess.run(["bash", "-c", launch, "fixture", *args], env={**os.environ, "SLURM_SUBMIT_DIR": str(tmp_path)}, check=True, capture_output=True)
        shuffle, routing, host_ip, *flags = probe.stdout.decode().rstrip("\0").split("\0")
        name, port = args[:2]
        prefill = name.startswith("prefill")
        expected_tp = 2 if prefill else 4
        assert f"--gres=gpu:{expected_tp}" in argv
        assert f"--cpus-per-task={16 if prefill else 32}" in argv
        assert flags[flags.index("--tp-size") + 1] == str(expected_tp)
        fixture_port = "30100" if prefill else "30200"
        assert host_ip == ("192.0.2.1" if name.startswith("prefill") else "192.0.2.2")
        observed[name] = argv[argv.index("-w") + 1]
        for flag, key in {"--ep-size": "ep_size", "--kv-cache-dtype": "kv_cache_dtype", "--nccl-port": "nccl_port", "--max-total-tokens": "max_total_tokens"}.items():
            expected = str(int(port) + 1000) if key == "nccl_port" else str(archived[fixture_port][key])
            assert flags[flags.index(flag) + 1] == expected
        assert flags[flags.index("--chunked-prefill-size") + 1] == "8192"
        assert flags[flags.index("--attention-backend") + 1] == "fa3"
        assert flags[flags.index("--max-running-requests") + 1] == ("64" if name.startswith("prefill") else "256")
        assert flags[flags.index("--speculative-algorithm") + 1] == "EAGLE3"
        assert flags[flags.index("--speculative-draft-model-path") + 1] == "/draft"
        assert flags[flags.index("--speculative-num-steps") + 1] == "3"
        assert flags[flags.index("--speculative-eagle-topk") + 1] == "1"
        assert flags[flags.index("--speculative-num-draft-tokens") + 1] == "4"
        assert "--speculative-draft-model-quantization" not in flags
        assert "--container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER" in argv
        if name.startswith("prefill"):
            assert flags[flags.index("--moe-runner-backend") + 1] == "cutlass"
            assert "--enable-symm-mem" in flags and (shuffle, routing) == ("1", "1")
        else:
            start = flags.index("--cuda-graph-bs-decode") + 1
            assert [int(value) for value in flags[start:start + len(config["decode_cuda_graph_bs"])]] == config["decode_cuda_graph_bs"]
            assert "--enable-symm-mem" not in flags and (shuffle, routing) == ("0", "0")
    assert observed == {"prefill-0": "node0", "prefill-1": "node0", "prefill-2": "node0", "prefill-3": "node0", "decode-0": "node1", "decode-1": "node1"}
    router_argv = next(argv for argv in commands if "sglang_router.launch_router" in argv)
    assert router_argv.count("--prefill") == 4 and router_argv.count("--decode") == 2
    assert router_argv[router_argv.index("--prefill-policy") + 1] == "round_robin"
    client_argv = next(argv for argv in commands if str(tmp_path / "scripts/official-aiperf.sh") in argv)
    assert client_argv[-2:] == [str(tmp_path), "64"]


def test_vllm_launch_matches_komori_p3_d1_dp4_ep4(tmp_path: Path) -> None:
    source = (ROOT / "platforms/firmus/framework-comparison/vllm.sbatch").read_text()
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    assert config["workload"] == "official"
    (tmp_path / "logs").mkdir()
    (tmp_path / "source").mkdir()
    (tmp_path / "source/firmus-framework.json").write_text(json.dumps(config))
    record = tmp_path / "commands.jsonl"
    stub = tmp_path / "srun"
    stub.write_text(f"#!{sys.executable}\nimport json,os,sys\nfd=os.open(os.environ['RECORD'],os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)\nos.write(fd,(json.dumps(sys.argv[1:])+'\\n').encode())\nos.close(fd)\n")
    stub.chmod(0o755)
    launches = "srun --exact" + source.split("srun --exact", 1)[1].split("\nwait_http", 1)[0]
    router = next(line for line in source.splitlines() if line.startswith("srun ") and "vllm-proxy.py" in line)
    client = next(line for line in source.splitlines() if line.startswith("srun ") and "official-aiperf.sh" in line)
    configuration = 'speculation=(' + source.split('speculation=(', 1)[1].split('\nART=', 1)[0]
    harness = 'set -euo pipefail\nmounts() { printf /fixture; }\nworker_pids=()\n' + configuration + '\n' + launches + '\nfor pid in "${worker_pids[@]}"; do wait "$pid"; done\n' + router + '\nwait "$!"\n' + client
    environment = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "RECORD": str(record), "ROOT": str(tmp_path), "ART": str(tmp_path / "logs"), "HEAD": "node0", "TAIL": "node1", "HEAD_IP": "192.0.2.1", "TAIL_IP": "192.0.2.2", "VLLM": "/image", "AIPERF": "/image", "MODEL": "/model", "N": str(config["concurrency"]), "R_PORT": "8000"}
    subprocess.run(["bash", "-c", harness], env=environment, check=True)
    commands = [json.loads(line) for line in record.read_text().splitlines()]
    workers = [argv for argv in commands if "--exact" in argv]
    assert len(workers) == 4
    assert sum(2 if "--gres=gpu:2" in argv else 4 for argv in workers) == 16
    observed = {argv[argv.index("bash") + 2]: argv[argv.index("-w") + 1] for argv in workers}
    assert observed == {"prefill-0": "node0", "prefill-1": "node0", "prefill-2": "node1", "decode-0": "node1"}
    for argv in workers:
        assert "--gres=gpu:4" in argv
        assert argv[argv.index("--moe-backend") + 1] == "apac_sm90_megamoe"
        assert "--enable-expert-parallel" in argv
        for option in ("--data-parallel-size", "--data-parallel-size-local"):
            assert argv[argv.index(option) + 1] == "4"
    for argv in workers:
        if any(arg.startswith("prefill-") for arg in argv):
            assert argv[argv.index("--max-num-seqs") + 1] == "48"
    decoder = next(argv for argv in workers if "decode-0" in argv)
    assert "--no-enable-prefix-caching" in decoder
    for option, value in [("--data-parallel-size", "4"), ("--data-parallel-size-local", "4"), ("--data-parallel-backend", "mp"), ("--all2all-backend", "allgather_reducescatter"), ("--max-num-seqs", "32")]:
        assert decoder[decoder.index(option) + 1] == value
    router_argv = next(argv for argv in commands if any(arg.endswith("/vllm-proxy.py") for arg in argv))
    begin = router_argv.index("--prefiller-ports") + 1
    assert router_argv[begin:begin + 3] == ["30100", "30110", "30120"]
    # Each native DP rank offsets the NIXL base port by its rank. The two
    # producer groups on node0 must have disjoint four-port ranges.
    side_channels = {}
    for argv in workers:
        args = argv[argv.index("bash") + 2:]
        host, api_port = argv[argv.index("-w") + 1], int(args[1])
        ranges = side_channels.setdefault(host, set())
        ports = set(range(api_port - 24500, api_port - 24500 + 4))
        assert not ranges & ports
        ranges.update(ports)
    client_argv = next(argv for argv in commands if str(tmp_path / "scripts/official-aiperf.sh") in argv)
    assert client_argv[-2:] == [str(tmp_path), str(config["concurrency"])]


def test_vllm_worker_arguments_preserve_eagle_topology_and_role_settings(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    source = (ROOT / "platforms/firmus/framework-comparison/vllm.sbatch").read_text()
    worker = source.split("<<'EOF'\n", 1)[1].split("\nEOF", 1)[0]
    # Exercise actual shell argument construction without requiring GPUs or HCAs.
    probe = worker.split("UCX_IB=", 1)[0] + "worker_command=(" + worker.split("worker_command=(", 1)[1].split("\nprintf", 1)[0]
    repair = tmp_path / "prefill-handoff-repair.json"
    gpu_kv = tmp_path / "gpu-kv.py"
    gpu_kv.write_bytes(runpy.run_path(str(ROOT / "containers/fix-vllm-prefill-handoff.py"))["patch_gpu_kv"]((ROOT / "qwen-experiment-metrics/tests/fixtures/vllm-gpu-kv.py.txt").read_bytes()))
    probe = probe.replace("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/kv_connector.py", str(gpu_kv))
    repair.write_text(json.dumps({"push_worker_source_sha256": "87de3ea383dd3aa95fb2c287ea26ded8a330859cc1295f9faa69047e80865c2c", "push_release_cpu_contract": {"passed": True}, "multiproc_source_sha256": "3ed0149a8a8e1550c4f062af40e645d1b734e70f99981bc5bf15f0360507648a", "message_queue_source_sha256": "f3e7a3f3ff713fccbd535e0d41650ee05bdf8677e83d4b90519910645022c30b", "executor_readiness_cpu_contract": {"passed": True}, "gpu_kv_source_sha256": "b89a4333f1e552d86e1815674b59e9f1335f0f49c0a7b46082439076f62c6a81", "gpu_kv_patched_sha256": "6472ceefd8f1cba8a90c99bd99f3ac40bc101fd382295ab83275a199ce98407b", "push_start_cpu_contract": {"passed": True, "producer_prior_stream_fence": True}}))
    probe = probe.replace("/opt/apac/prefill-handoff-repair.json", str(repair))
    probe += '\nprintf "%s\\0" "$VLLM_ALLREDUCE_USE_SYMM_MEM" "${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-1}" "${TRTLLM_DG_ENABLED:-1}" "$VLLM_USE_DEEP_GEMM_E8M0" "$VLLM_USE_NCCL_SYMM_MEM" "${worker_command[@]}"\n'
    for role, tp, limit in [("prefill", 1, 48), ("decode", 1, 32)]:
        options = ["--speculative-config", '{"method":"eagle3","model":"/draft","num_speculative_tokens":3}', "--max-num-seqs", str(limit)]
        if role == "prefill":
            options += ["--disable-custom-all-reduce", "--enable-expert-parallel", "--moe-backend", "apac_sm90_megamoe"]
        else:
            options += ["--enable-expert-parallel", "--moe-backend", "apac_sm90_megamoe", "--compilation-config", '{"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":128}', "--cudagraph-capture-sizes", *[str(n * 4) for n in config["decode_cuda_graph_bs"]]]
        result = subprocess.run(["bash", "-c", probe, str(tmp_path / "worker.sh"), f"{role}-0", "30100", "/model", str(tp), "192.0.2.1", *options], check=True, capture_output=True)
        symm, dense_flashinfer, trtllm_deepgemm, e8m0, nccl_symm, *flags = result.stdout.decode().rstrip("\0").split("\0")
        assert symm == ("1" if role == "prefill" else "0")
        assert dense_flashinfer == "0"
        assert trtllm_deepgemm == "1"
        assert e8m0 == "0" and nccl_symm == "0"
        if role == "prefill":
            assert "--enforce-eager" not in flags
            assert json.loads(flags[flags.index("--compilation-config") + 1]) == {"mode": 3, "cudagraph_mode": "NONE", "pass_config": {"fuse_norm_quant": True}}
        else:
            assert json.loads(flags[flags.index("--compilation-config") + 1]) == {"cudagraph_mode": "FULL_AND_PIECEWISE", "max_cudagraph_capture_size": 128}
            begin = flags.index("--cudagraph-capture-sizes") + 1
            buckets = [n * 4 for n in config["decode_cuda_graph_bs"]]
            assert [int(n) for n in flags[begin:begin + len(buckets)]] == buckets
        assert flags[flags.index("--tensor-parallel-size") + 1] == str(tp)
        assert flags[flags.index("--max-num-seqs") + 1] == str(limit)
        assert "--enable-expert-parallel" in flags
        assert flags[flags.index("--moe-backend") + 1] == "apac_sm90_megamoe"
        assert flags[flags.index("--dtype") + 1] == "bfloat16"
        assert flags[flags.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
        assert int(flags[flags.index("--block-size") + 1]) * int(flags[flags.index("--num-gpu-blocks-override") + 1]) == (131072 if role == "prefill" else 393216)
        assert flags[flags.index("--block-size") + 1] == "64"
        assert flags[flags.index("--max-num-batched-tokens") + 1] == ("4096" if role == "prefill" else "2048")
        assert json.loads(flags[flags.index("--attention-config") + 1]) == {"backend": "FLASH_ATTN", "flash_attn_version": 3}
        assert json.loads(flags[flags.index("--kv-transfer-config") + 1])["kv_role"] == ("kv_producer" if role == "prefill" else "kv_consumer")
        kv = json.loads(flags[flags.index("--kv-transfer-config") + 1])
        assert kv["kv_connector"] == "NixlPushConnector"
        assert kv["engine_id"] == f"{tmp_path.parent.name}-{role}-0"
        assert json.loads(flags[flags.index("--speculative-config") + 1]) == {"method": "eagle3", "model": "/draft", "num_speculative_tokens": 3}


@pytest.mark.parametrize("worker_status,expected", [(127, 127), (0, 1)])
def test_readiness_stops_on_any_exited_server_before_curl(tmp_path: Path, worker_status: int, expected: int) -> None:
    source = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    readiness = "wait_http() {" + source.split("wait_http() {", 1)[1].split('\necho "[$(date', 1)[0]
    # The first server remains alive, while another has already failed. Even a
    # healthy endpoint must not hide that failure or retry until the 600s cap.
    harness = readiness + f'\nREADY_SECS=600\n(sleep 60) &\nhealthy=$!\ntrap \'kill "$healthy"; wait "$healthy" 2>/dev/null || true\' EXIT\n(exit {worker_status}) &\nfailed=$!\nwait "$failed" || true\nworker_pids=("$healthy" "$failed")\ncurl() {{ touch "{tmp_path / "curl-called"}"; return 0; }}\nwait_http http://fixture/health\n'
    result = subprocess.run(["bash", "-c", harness], capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == expected
    assert f"status={worker_status}" in result.stderr
    assert not (tmp_path / "curl-called").exists()


@pytest.mark.parametrize("framework,profile", [("sglang", "false"), ("sglang", "true"), ("vllm", "false"), ("vllm", "true")])
def test_provenance_distinguishes_server_client_and_prepared_cache(tmp_path: Path, framework: str, profile: str) -> None:
    runtime = tmp_path / "runtime"
    (runtime / "source").mkdir(parents=True)
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    commands = "jq -n --arg model_path" + provision.split("jq -n --arg model_path", 1)[1]
    config = ROOT / "configs/firmus-framework.json"
    environment = {**os.environ, "runtime_dir": str(runtime), "model": "/scratch/model", "framework": framework, "config": str(config), "GITHUB_SHA": "a" * 40, "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "PROFILE": profile}
    subprocess.run(["bash", "-euc", commands], env=environment, check=True)
    identity = json.loads((runtime / "source/provenance.json").read_text())
    images = json.loads(config.read_text())["images"]
    assert identity["profile"] is (profile == "true")
    assert identity["container"]["sha256"] == images[framework]["sha256"]
    assert identity["client_container"]["sha256"] == images["sglang"]["sha256"]
    assert identity["cache_source"] == ("matched-settings-comparison/home" if framework == "vllm" else None)
    model = json.loads((runtime / "source/model.json").read_text())
    assert model["schema_version"] == "qwen3-pd-model/v1"
    assert model["model"]["path"] == "/scratch/model"
    assert model["model"]["shards"] == 24


def test_collection_retries_interrupted_artifacts_without_submitting(tmp_path: Path) -> None:
    workflow = (ROOT / ".github/workflows/firmus.yml").read_text()
    collect = workflow.split("id: collect\n", 1)[1].split("      - name:", 1)[0]
    script = textwrap.dedent(collect.split("        run: |\n", 1)[1])
    shell = tmp_path / "firmus-shell"
    shell.write_text(f"#!{sys.executable}\nimport os,sys\nfrom pathlib import Path\nroot=Path(os.environ['RUNNER_TEMP'])\nargs=sys.argv[1:]\nwith (root/'reads').open('a') as f: f.write(repr(args)+'\\n')\nkey='source' if args[0]=='--pull' and args[1].endswith('/source') else 'marker' if args[0].startswith('if test') else ''\nif key and not (root/key).exists():\n (root/key).touch()\n sys.exit(2 if key=='source' else 255)\nif key=='marker': print('present')\n")
    shell.chmod(0o755)
    for name, body in {"timeout": 'shift 2; exec "$@"', "sleep": ':'}.items():
        stub = tmp_path / name
        stub.write_text(f"#!/bin/bash\n{body}\n")
        stub.chmod(0o755)
    environment = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "RUNNER_TEMP": str(tmp_path), "RUNTIME_DIR": "/runs/full-one-job"}
    subprocess.run(["bash", "-euc", script], env=environment, check=True)
    reads = (tmp_path / "reads").read_text().splitlines()
    assert len(reads) == 8  # Four directories, marker and run-success, plus two failed reads.
    assert sum("/source'" in row for row in reads) == 2
    assert sum("if test" in row for row in reads) == 2
    assert reads[-1].startswith("['--pull', '/runs/full-one-job/run-success'")
    assert all("sbatch" not in row for row in reads)


@pytest.mark.parametrize("corrupt", [None, "commit", "sha256", "size", "missing_build_output"])
def test_require_build_selects_only_this_runs_verified_image(tmp_path: Path, corrupt: str | None) -> None:
    runtime = tmp_path / "runs"
    archive = tmp_path / "images"
    archive.mkdir()
    commit, digest, size = "a" * 40, "b" * 64, 1234
    sidecar = {"source_commits": {"qwen": commit}, "container_artifact": {"artifact_relative_path": "vllm.sqsh", "sha256": digest, "size_bytes": size}}
    if corrupt == "commit":
        sidecar["source_commits"]["qwen"] = "c" * 40
    elif corrupt == "sha256":
        sidecar["container_artifact"]["sha256"] = "d" * 64
    elif corrupt == "size":
        sidecar["container_artifact"]["size_bytes"] += 1
    (archive / "123.container-sidecar.json").write_text(json.dumps(sidecar))
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    selection = provision.split("for image_framework in", 1)[0].replace("/scratch/home/ACCOUNT/qwen/vllm-containers", str(archive))
    environment = {**os.environ, "GITHUB_WORKSPACE": str(ROOT), "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": commit, "GITHUB_OUTPUT": str(tmp_path / "outputs"), "RUNTIME_ROOT": str(runtime), "PROFILE": "false", "REQUIRE_BUILD": "true", "CONTAINER_SHA256": digest, "CONTAINER_SIZE": str(size)}
    if corrupt == "missing_build_output":
        environment.pop("CONTAINER_SHA256")
    result = subprocess.run(["bash", "-c", selection], env=environment, capture_output=True, text=True, check=False)
    if corrupt:
        assert result.returncode != 0
        return
    assert result.returncode == 0, result.stderr
    prepared = json.loads(next(runtime.glob("*/source/firmus-framework.json")).read_text())
    original = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    assert prepared["images"]["vllm"] == {"path": str(archive / "123.sqsh"), "provenance": str(archive / "123.container-sidecar.json"), "sha256": digest, "size_bytes": size}
    assert prepared["images"]["sglang"] == original["images"]["sglang"]
    assert original["images"]["vllm"]["path"] != prepared["images"]["vllm"]["path"]


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("corruption", [None, "dp", "cache", "backend", "admission", "draft_ep", "graph", "pull_connector", "executor", "pipeline", "async", "compile", "norm_quant"])
def test_megamoe_effective_config_guard_rejects_wrong_runtime(tmp_path: Path, corruption: str | None, role: str) -> None:
    source = (ROOT / "platforms/firmus/framework-comparison/vllm.sbatch").read_text()
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    decoder = {
        "parallel_config": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1, "distributed_executor_backend": "mp", "data_parallel_size": 4, "enable_expert_parallel": True, "all2all_backend": "allgather_reducescatter"},
        "kernel_config": {"moe_backend": "apac_sm90_megamoe"},
        "kv_transfer_config": {"kv_connector": "NixlPushConnector"},
        "cache_config": {"cache_dtype": "fp8_e4m3", "block_size": 64, "num_gpu_blocks_override": 2048 if role == "prefill" else 6144, "enable_prefix_caching": False},
        "model_config": {"max_model_len": 8192, "enforce_eager": False},
        "scheduler_config": {"async_scheduling": True, "max_num_batched_tokens": 4096 if role == "prefill" else 2048, "max_num_seqs": 48 if role == "prefill" else 32},
        "compilation_config": {"mode": 3, "pass_config": {"fuse_norm_quant": True}, "max_cudagraph_capture_size": 0 if role == "prefill" else 128, "cudagraph_mode": 0 if role == "prefill" else [2, 1], "cudagraph_capture_sizes": [] if role == "prefill" else [n * 4 for n in config["decode_cuda_graph_bs"]]},
        "speculative_config": {"draft_parallel_config": {"enable_expert_parallel": False, "tensor_parallel_size": 1}, "method": "eagle3", "model": "/draft", "num_speculative_tokens": 3},
    }
    if corruption == "executor":
        decoder["parallel_config"]["distributed_executor_backend"] = "uni"
    elif corruption == "pipeline":
        decoder["parallel_config"]["pipeline_parallel_size"] = 2
    elif corruption == "async":
        decoder["scheduler_config"]["async_scheduling"] = False
    elif corruption == "dp":
        decoder["parallel_config"]["data_parallel_size"] = 1
    elif corruption == "cache":
        decoder["cache_config"]["num_gpu_blocks_override"] = 8192
    elif corruption == "backend":
        decoder["parallel_config"]["all2all_backend"] = "naive"
    elif corruption == "admission":
        decoder["scheduler_config"]["max_num_seqs"] = 2
    elif corruption == "draft_ep":
        decoder["speculative_config"]["draft_parallel_config"]["enable_expert_parallel"] = True
    elif corruption == "compile":
        if role == "decode":
            pytest.skip("This candidate changes only the producer compilation contract")
        decoder["compilation_config"]["mode"] = 0
    elif corruption == "norm_quant":
        if role == "decode":
            pytest.skip("This candidate changes only the producer fusion contract")
        decoder["compilation_config"]["pass_config"]["fuse_norm_quant"] = False
    elif corruption == "graph":
        decoder["model_config"]["enforce_eager"] = False
        decoder["compilation_config"]["cudagraph_mode"] = 2
    elif corruption == "pull_connector":
        decoder["kv_transfer_config"]["kv_connector"] = "NixlConnector"
    (tmp_path / "source").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "source/firmus-framework.json").write_text(json.dumps(config))
    (tmp_path / f"results/server-info-{role}-0.json").write_text(json.dumps({"vllm_config": decoder}))
    guard = next(line for line in source.splitlines() if line.lstrip().startswith('jq -e --arg role'))
    result = subprocess.run(["bash", "-c", guard], env={**os.environ, "ROOT": str(tmp_path), "worker": f"{role}-0", "_tp": "1"}, check=False, capture_output=True)
    assert (result.returncode == 0) is (corruption is None), result.stderr.decode()


@pytest.mark.parametrize("name", ["prefill-0", "decode-0", "prefill-1"])
@pytest.mark.parametrize("profile", [False, True])
def test_profile_worker_records_nic_inventory_without_changing_device_selection(tmp_path: Path, name: str, profile: bool) -> None:
    source = (ROOT / "platforms/firmus/framework-comparison/vllm.sbatch").read_text()
    block = 'if [[ -x /opt/nsys/bin/nsys ]]; then' + source.split('if [[ -x /opt/nsys/bin/nsys ]]; then', 1)[1].split('\nmkdir -p "$TRITON_CACHE_DIR"', 1)[0]
    binary = tmp_path / "nsys"
    if profile:
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
    devices = tmp_path / "infiniband"
    for device, layer, state, gid_index, gid in [("mlx5_0", "InfiniBand", "4: ACTIVE", "0", "fe80:0000:0000:0000:0000:0000:0000:0001"), ("mlx5_1", "Ethernet", "4: ACTIVE", "3", "0000:0000:0000:0000:0000:ffff:c612:3933"), ("mlx5_2", "InfiniBand", "1: DOWN", "0", "fe80:0000:0000:0000:0000:0000:0000:0002")]:
        port = devices / device / "ports/1"
        for directory in ("gids", "gid_attrs/ndevs", "gid_attrs/types"):
            (port / directory).mkdir(parents=True)
        for filename, content in {"link_layer": layer, "state": state, "rate": "400 Gb/sec", f"gids/{gid_index}": gid, "gids/1": "0000:0000:0000:0000:0000:0000:0000:0000", f"gid_attrs/ndevs/{gid_index}": "eth0" if layer == "Ethernet" else "", f"gid_attrs/types/{gid_index}": "RoCE v2" if layer == "Ethernet" else "IB/RoCE v1"}.items():
            (port / filename).write_text(content + "\n")
    block = block.replace("/opt/nsys/bin/nsys", str(binary)).replace("/sys/class/infiniband", str(devices))
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", block + '\nprintf "SELECTED=%s\\n" "$UCX_NET_DEVICES"'], env={**os.environ, "name": name, "UCX_NET_DEVICES": "unchanged:1"}, check=True, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    assert lines[-1] == "SELECTED=unchanged:1"
    if profile and name in {"prefill-0", "decode-0"}:
        assert sum(line.startswith("NIC device=") for line in lines) == 3
        assert sum(line.startswith("NIC_GID ") for line in lines) == 3
        assert any("link_layer=Ethernet" in line and "state=4: ACTIVE" in line for line in lines)
        assert any("index=3" in line and "netdev=eth0 type=RoCE v2" in line for line in lines)
        assert not any("NIC_GID" in line and "index=1 " in line for line in lines)
    else:
        assert lines == ["SELECTED=unchanged:1"]
