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


def test_sglang_launch_matches_p5tp2_mixed_decode_conditions(tmp_path: Path) -> None:
    source = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    assert (config["framework"], config["concurrency"], config["workload"]) == ("sglang", 104, "official")
    assert config["decode_max_total_tokens"] == 524288
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
    environment = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "RECORD": str(record), "ROOT": str(tmp_path), "ART": str(tmp_path / "logs"), "HEAD": "node0", "TAIL": "node1", "HEAD_IP": "192.0.2.1", "TAIL_IP": "192.0.2.2", "SGLANG": "/image", "AIPERF": "/image", "MODEL": "/model", "N": str(config["concurrency"]), "R_PORT": "8000"}
    subprocess.run(["bash", "-c", harness], env=environment, check=True)
    commands = [json.loads(line) for line in record.read_text().splitlines()]
    workers = [argv for argv in commands if "--exact" in argv]
    assert len(workers) == 7
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
        expected_tp = 2 if prefill or name == "decode-1" else 4
        assert f"--gres=gpu:{expected_tp}" in argv
        assert f"--cpus-per-task={16 if expected_tp == 2 else 32}" in argv
        assert flags[flags.index("--tp-size") + 1] == str(expected_tp)
        fixture_port = "30100" if prefill else "30200"
        assert host_ip == ("192.0.2.1" if name in {"prefill-0", "prefill-1", "prefill-2", "prefill-3"} else "192.0.2.2")
        observed[name] = argv[argv.index("-w") + 1]
        for flag, key in {"--ep-size": "ep_size", "--kv-cache-dtype": "kv_cache_dtype", "--nccl-port": "nccl_port", "--max-total-tokens": "max_total_tokens"}.items():
            expected = str(int(port) + 1000) if key == "nccl_port" else str(archived[fixture_port][key])
            # Decode appends a larger --max-total-tokens after the shared default; SGLang keeps the last value.
            value = flags[len(flags) - 1 - flags[::-1].index(flag) + 1]
            pools = {"decode-0": config["decode_max_total_tokens"], "decode-1": config["decode_tp2"]["max_total_tokens"]}
            assert value == (str(pools[name]) if key == "max_total_tokens" and name in pools else expected)
        assert flags[flags.index("--chunked-prefill-size") + 1] == "8192"
        assert flags[flags.index("--attention-backend") + 1] == "fa3"
        assert flags[flags.index("--max-running-requests") + 1] == (str(config["concurrency"]) if prefill else str(config["decode_tp2"]["max_running_requests"]) if name == "decode-1" else "256")
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
            graph = [size for size in config["decode_cuda_graph_bs"] if name == "decode-0" or size <= config["decode_tp2"]["graph_max_bs"]]
            assert [int(value) for value in flags[start:start + len(graph)]] == graph and (start + len(graph) == len(flags) or not flags[start + len(graph)].isdigit())
            assert "--enable-symm-mem" not in flags and (shuffle, routing) == ("0", "0")
    assert observed == {"prefill-0": "node0", "prefill-1": "node0", "prefill-2": "node0", "prefill-3": "node0", "prefill-4": "node1", "decode-0": "node1", "decode-1": "node1"}
    router_argv = next(argv for argv in commands if "sglang_router.launch_router" in argv)
    assert router_argv.count("--prefill") == 5 and router_argv.count("--decode") == 2
    assert router_argv[router_argv.index("--prefill-policy") + 1] == "power_of_two"
    assert router_argv[router_argv.index("--decode-policy") + 1] == "power_of_two"
    assert router_argv[router_argv.index("--worker-startup-check-interval") + 1] == "1"
    client_argv = next(argv for argv in commands if str(tmp_path / "scripts/official-aiperf.sh") in argv)
    assert client_argv[-2:] == [str(tmp_path), str(config["concurrency"])]


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


@pytest.mark.parametrize("profile,require_build", [("false", "true")])
def test_unsupported_modes_fail_before_container_staging(tmp_path: Path, profile: str, require_build: str) -> None:
    runtime = tmp_path / "runs"
    environment = {**os.environ, "GITHUB_WORKSPACE": str(ROOT), "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "a" * 40, "GITHUB_OUTPUT": str(tmp_path / "outputs"), "RUNTIME_ROOT": str(runtime), "PROFILE": profile, "REQUIRE_BUILD": require_build}
    result = subprocess.run(["bash", str(ROOT / "scripts/provision-firmus-framework.sh")], env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 64
    assert "reuses pinned baseline images" in result.stderr
    assert not runtime.exists()
