from __future__ import annotations

import copy
import json
import os
import re
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


def archived_server_info() -> dict:
    # The archived run predates placement, optimistic Prefill, the configured Prefill chunk and concurrency, the Decode cap 136 and the polling knob; carry the current settings into its config and every scheduler state.
    raw = json.loads((ROOT / "qwen-experiment-metrics/tests/fixtures/firmus-server-info-38369214512.json").read_text())
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    current = config["prefill"]
    location = current["expert_location"]
    raw["config"]["prefill"].update(expert_location=location, ep_num_redundant_experts=current["ep_num_redundant_experts"], optimistic_prefill_attempts=current["optimistic_prefill_attempts"], chunked_prefill_size=current["chunked_prefill_size"])
    raw["config"].update({key: config[key] for key in ("concurrency", "decode_max_running_requests", "decode_extra_slots", "decode_cuda_graph_bs", "decode_polling_interval")})
    raw["config"]["speculation"]["draft_quantization"] = config["speculation"].get("draft_quantization")
    for role in ("prefill", "decode"):
        placed = role == "prefill" and location != "trivial"
        for state in [raw[role], *raw[role]["internal_states"]]:
            state["speculative_draft_model_quantization"] = config["speculation"].get("draft_quantization")
            state.update(init_expert_location="/runtime/source/prefill-expert-location.json" if placed else "trivial", ep_dispatch_algorithm="static" if placed else None, optimistic_prefill_attempts=current["optimistic_prefill_attempts"] if role == "prefill" else 0)
            state.update(ep_num_redundant_experts=current["ep_num_redundant_experts"] if role == "prefill" else 0, enable_eplb=False)
            if role == "prefill":
                state.update(chunked_prefill_size=current["chunked_prefill_size"] // current["dp"], max_running_requests=config["concurrency"])
            if role == "decode":
                state.update(max_running_requests=config["decode_max_running_requests"], disaggregation_decode_polling_interval=config["decode_polling_interval"], disaggregation_decode_extra_slots=config["decode_extra_slots"])
    raw["decode"]["cuda_graph_bs_decode"] = config["decode_cuda_graph_bs"]
    for state in raw["decode"]["internal_states"]:
        state["effective_max_running_requests_per_dp"] = config["decode_max_running_requests"] // config["decode_dp_size"]
    return raw


def test_optimistic_prefill_attempts_are_checked_on_every_prefill_state() -> None:
    # Run E (39040135834) archived optimistic_prefill_attempts=2 on the top-level and all four DP states.
    raw = archived_server_info()
    check = runpy.run_path(str(ROOT / "scripts/check-firmus-server-info.py"))["check"]
    assert raw["config"]["prefill"]["optimistic_prefill_attempts"] == 2
    check(raw["config"], raw["prefill"], "prefill")
    for index in range(5):
        info = copy.deepcopy(raw["prefill"])
        [info, *info["internal_states"]][index]["optimistic_prefill_attempts"] = 0
        with pytest.raises(ValueError, match="optimistic_prefill_attempts"):
            check(raw["config"], info, "prefill")
    # Configs written before the key existed ran with SGLang's default of 0.
    legacy = copy.deepcopy(raw["config"])
    del legacy["prefill"]["optimistic_prefill_attempts"]
    info = copy.deepcopy(raw["prefill"])
    for state in [info, *info["internal_states"]]:
        state["optimistic_prefill_attempts"] = 0
    check(legacy, info, "prefill")


def test_decode_polling_interval_is_checked_on_every_decode_state() -> None:
    raw = archived_server_info()
    check = runpy.run_path(str(ROOT / "scripts/check-firmus-server-info.py"))["check"]
    check(raw["config"], raw["decode"], "decode")
    for index in range(5):
        info = copy.deepcopy(raw["decode"])
        [info, *info["internal_states"]][index]["disaggregation_decode_polling_interval"] = 2
        with pytest.raises(ValueError, match="disaggregation_decode_polling_interval"):
            check(raw["config"], info, "decode")


def test_prefill_expert_replicas_and_static_dispatch_checked_on_every_state() -> None:
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    placement = json.loads((ROOT / config["prefill"]["expert_location"]).read_text())
    assert list(placement) == ["physical_to_logical_map"]
    assert len(placement["physical_to_logical_map"]) == 94
    assert config["prefill"]["ep_num_redundant_experts"] == 12
    for row in placement["physical_to_logical_map"]:
        hot = {e for e in range(128) if row.count(e) == 4}
        assert len(row) == 140 and set(row) == set(range(128)) and len(hot) == 4
        assert all(row.count(e) == (4 if e in hot else 1) for e in range(128))
        for rank in range(4):
            local = row[rank * 35:(rank + 1) * 35]
            assert len(set(local)) == 35 and hot <= set(local)
    raw = archived_server_info()
    check = runpy.run_path(str(ROOT / "scripts/check-firmus-server-info.py"))["check"]
    for role, key, invalid in (("prefill", "init_expert_location", "trivial"), ("prefill", "ep_dispatch_algorithm", None), ("prefill", "ep_num_redundant_experts", 0), ("prefill", "enable_eplb", True), ("decode", "init_expert_location", "/runtime/source/prefill-expert-location.json"), ("decode", "ep_dispatch_algorithm", "static"), ("decode", "ep_num_redundant_experts", 12), ("decode", "enable_eplb", True)):
        for index in range(5):
            info = copy.deepcopy(raw[role])
            [info, *info["internal_states"]][index][key] = invalid
            with pytest.raises(ValueError, match=key):
                check(raw["config"], info, role)


def test_server_info_checks_resolved_draft_backend_from_real_run() -> None:
    # The failed run reached readiness, then rejected SGLang's None -> deep_gemm normalization.
    raw = archived_server_info()
    check = runpy.run_path(str(ROOT / "scripts/check-firmus-server-info.py"))["check"]
    for role in ("prefill", "decode"):
        check(raw["config"], raw[role], role)
    for index in range(5):
        for backend in (None, "triton"):
            info = copy.deepcopy(raw["decode"])
            [info, *info["internal_states"]][index]["speculative_moe_runner_backend"] = backend
            with pytest.raises(ValueError, match="speculative_moe_runner_backend"):
                check(raw["config"], info, "decode")


@pytest.mark.parametrize("draft_quantization", ["fp8", None, "unset"])
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_draft_precision_matches_config_on_every_scheduler(draft_quantization, role) -> None:
    raw = archived_server_info()
    check = runpy.run_path(str(ROOT / "scripts/check-firmus-server-info.py"))["check"]
    if draft_quantization == "unset":
        raw["config"]["speculation"].pop("draft_quantization", None)
    else:
        raw["config"]["speculation"]["draft_quantization"] = draft_quantization
    expected = raw["config"]["speculation"].get("draft_quantization")
    for state in [raw[role], *raw[role]["internal_states"]]:
        state["speculative_draft_model_quantization"] = expected
    check(raw["config"], raw[role], role)
    for index in range(5):
        changed = copy.deepcopy(raw[role])
        [changed, *changed["internal_states"]][index]["speculative_draft_model_quantization"] = None if expected == "fp8" else "fp8"
        with pytest.raises(ValueError, match="speculative_draft_model_quantization"):
            check(raw["config"], changed, role)


def test_prefill_deepep_checks_normal_mode_without_changing_decode() -> None:
    raw = archived_server_info()
    check = runpy.run_path(str(ROOT / "scripts/check-firmus-server-info.py"))["check"]
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    config["prefill"]["moe_a2a_backend"] = "deepep"
    prefill = copy.deepcopy(raw["prefill"])
    for state in [prefill, *prefill["internal_states"]]:
        state.update(moe_a2a_backend="deepep", deepep_mode="normal", disable_cuda_graph=True)
    check(config, prefill, "prefill")
    check(config, raw["decode"], "decode")
    for key, invalid in (("moe_a2a_backend", "megamoe"), ("deepep_mode", "low_latency"), ("disable_cuda_graph", False)):
        changed = copy.deepcopy(prefill)
        changed["internal_states"][2][key] = invalid
        with pytest.raises(ValueError, match=key):
            check(config, changed, "prefill")


@pytest.mark.parametrize("invalid", [False, True])
def test_server_info_validation_runs_after_gpu_collection(tmp_path: Path, invalid: bool) -> None:
    workflow = (ROOT / ".github/workflows/firmus.yml").read_text()
    step = workflow.split("      - name: GPU終了後に回収した実設定を検証\n", 1)[1].split("\n      - name:", 1)[0]
    assert "steps.collect.outcome == 'success'" in step
    assert workflow.index("GPU終了後に回収した実設定を検証") > workflow.index("uses: actions/upload-artifact@v7")
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    raw = archived_server_info()
    artifact = tmp_path / "qwen-firmus-pd"
    (artifact / "source/source").mkdir(parents=True)
    (artifact / "results/results").mkdir(parents=True)
    (artifact / "source/source/firmus-framework.json").write_text(json.dumps(raw["config"]))
    if invalid:
        raw["decode"]["internal_states"][3]["speculative_moe_runner_backend"] = "triton"
    for worker in ("prefill-0", "prefill-1", "prefill-2", "decode-0"):
        (artifact / f"results/results/server-info-{worker}.json").write_text(json.dumps(raw[worker.split("-")[0]]))
    uv = tmp_path / "uv"
    uv.write_text(f'#!/bin/bash\nshift 5\nexec "{sys.executable}" "$@"\n')
    uv.chmod(0o755)
    result = subprocess.run(["bash", "-euc", script], cwd=ROOT, env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "RUNNER_TEMP": str(tmp_path)}, capture_output=True, text=True, check=False)
    assert result.returncode == (1 if invalid else 0)
    assert result.stdout.count("PASS:") == (3 if invalid else 4)
    if invalid:
        assert "decode state 4: speculative_moe_runner_backend" in result.stderr


@pytest.mark.parametrize("draft_quantization", ["fp8", None, "unset"])
def test_sglang_launch_uses_dp4_ep4_for_prefill_and_decode(tmp_path: Path, draft_quantization) -> None:
    source = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    assert "#SBATCH -t 00:17:00" in source
    # Tier 2 switches are all on for the bundled image; the startup line prints each value.
    assert "\nexport SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY=1\n" in source
    startup = next(line for line in source.splitlines() if line.startswith('echo "$name host='))
    for switch in ("SGLANG_PREFILL_FUSED_FP8_SHUFFLE", "SGLANG_PREFILL_TILED_ROUTING", "SGLANG_PREFILL_REQUEST_ALIGNED_CHUNKS", "SGLANG_DP_TOTAL_REQUESTS_ROTATE_TIES", "SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY", "SGLANG_DECODE_ASYNC_ADMISSION"):
        assert "${" + switch + ":-0}" in startup
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    assert (config["framework"], config["concurrency"], config["workload"]) == ("sglang", 240, "official")
    assert config["decode_max_total_tokens"] * config["decode_dp_size"] == 1572864
    assert config["decode_max_running_requests"] == 176  # 44 requests/rank, four verification rows/request
    assert config["decode_polling_interval"] == 1  # Async admission replaces the polling interval of 2
    assert config["prefill"]["optimistic_prefill_attempts"] == 2
    assert config["decode_max_running_requests"] + config["decode_extra_slots"] * config["decode_dp_size"] >= config["concurrency"]
    assert max(config["decode_cuda_graph_bs"]) == config["decode_max_running_requests"] // config["decode_dp_size"]
    assert config["decode_cuda_graph_bs"][-1] == 44
    # KV capacity is per DP rank and includes both running and transfer slots.
    slots_per_rank = config["decode_max_running_requests"] // config["decode_dp_size"] + config["decode_extra_slots"]
    assert config["decode_max_total_tokens"] >= slots_per_rank * (4096 + 128)
    (tmp_path / "logs").mkdir()
    (tmp_path / "source").mkdir()
    if draft_quantization == "unset":
        config["speculation"].pop("draft_quantization", None)
    else:
        config["speculation"]["draft_quantization"] = draft_quantization
    (tmp_path / "source/firmus-framework.json").write_text(json.dumps(config))
    record = tmp_path / "commands.jsonl"
    stub = tmp_path / "srun"
    stub.write_text(f"#!{sys.executable}\nimport json,os,sys\nfd=os.open(os.environ['RECORD'],os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)\nos.write(fd,(json.dumps(sys.argv[1:])+'\\n').encode())\nos.close(fd)\n")
    stub.chmod(0o755)
    launches = "srun --exact" + source.split("srun --exact", 1)[1].split("\nwait_http", 1)[0]
    router = next(line for line in source.splitlines() if line.startswith("srun ") and "/opt/apac/bin/sglang-router" in line)
    client = next(line for line in source.splitlines() if line.startswith("srun ") and "official-aiperf.sh" in line)
    configuration = 'speculation=(' + source.split('speculation=(', 1)[1].split('\nART=', 1)[0]
    # The Decode worker launch waits for the dense selection started before the numeric gate and merges both env parts.
    selection = 'mkdir -p "$ROOT/results/decode-selection"\n: >"$ROOT/results/decode-selection/fa3.env"\n: >"$ROOT/results/decode-selection/dense.env"\ntrue &\ndense_selection=$!\n'
    harness = 'set -euo pipefail\nmounts() { printf /fixture; }\nworker_pids=()\n' + configuration + '\n' + selection + launches + '\nfor pid in "${worker_pids[@]}"; do wait "$pid"; done\n' + router + '\nwait "$!"\n' + client
    environment = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "RECORD": str(record), "ROOT": str(tmp_path), "ART": str(tmp_path / "logs"), "HEAD": "node0", "TAIL": "node1", "HEAD_IP": "192.0.2.1", "TAIL_IP": "192.0.2.2", "SGLANG": "/image", "AIPERF": "/image", "MODEL": "/model", "N": str(config["concurrency"]), "R_PORT": "8000"}
    subprocess.run(["bash", "-c", harness], env=environment, check=True)
    commands = [json.loads(line) for line in record.read_text().splitlines()]
    workers = [argv for argv in commands if "--exact" in argv and str(tmp_path / "source/worker.sh") in argv]
    assert source.index("check-firmus-megamoe.py") < source.index("srun --exact")
    assert "check-firmus-server-info.py" not in source
    assert len(workers) == 4
    assert sum(2 if "--gres=gpu:2" in argv else 4 for argv in workers) == 16
    archived = json.loads((ROOT / "qwen-experiment-metrics/tests/fixtures/aspire-integration-argv.json").read_text())["archived_server_info"]
    worker = source.split("<<'EOF'\n", 1)[1].split("\nEOF", 1)[0]
    assert "jq" not in worker  # Not installed in the pinned image; parse on the host.
    command = "command=(" + worker.split("command=(", 1)[1].split('if [[ -x /opt/nsys', 1)[0]
    launch = worker.split("UCX_IB=", 1)[0] + command + '\nprintf "%s\\0" "${SGLANG_PREFILL_FUSED_FP8_SHUFFLE:-0}" "${SGLANG_PREFILL_TILED_ROUTING:-0}" "${SGLANG_PREFILL_REQUEST_ALIGNED_CHUNKS:-0}" "${SGLANG_DP_TOTAL_REQUESTS_ROTATE_TIES:-0}" "${SGLANG_DECODE_ASYNC_ADMISSION:-0}" "$SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK" "$SGLANG_HOST_IP" "${command[@]}"\n'
    observed = {}
    # The Decode worker sources the pre-serving kernel selection next to worker.sh ($0 is "fixture" here).
    (tmp_path / "decode-kernels.env").write_text("")
    for argv in workers:
        args = argv[argv.index("bash") + 2:]
        probe = subprocess.run(["bash", "-c", launch, "fixture", *args], cwd=tmp_path, env={**os.environ, "SLURM_SUBMIT_DIR": str(tmp_path)}, check=True, capture_output=True)
        shuffle, routing, aligned, rotate_ties, async_admission, megamoe_tokens, host_ip, *flags = probe.stdout.decode().rstrip("\0").split("\0")
        name, port = args[:2]
        prefill = name.startswith("prefill")
        dist_port = int(flags[flags.index("--dist-init-addr") + 1].rsplit(":", 1)[1])
        assert dist_port + 5 < 32768  # All DP TCP endpoints avoid Firmus ephemeral ports.
        expected_tp = 4
        assert f"--gres=gpu:{expected_tp}" in argv
        assert "--cpus-per-task=32" in argv
        assert flags[flags.index("--tp-size") + 1] == str(expected_tp)
        fixture_port = "30100" if prefill else "30200"
        assert host_ip == ("192.0.2.1" if name in {"prefill-0", "prefill-1"} else "192.0.2.2")
        observed[name] = argv[argv.index("-w") + 1]
        for flag, key in {"--ep-size": "ep_size", "--kv-cache-dtype": "kv_cache_dtype", "--nccl-port": "nccl_port", "--max-total-tokens": "max_total_tokens"}.items():
            expected = str(int(port) + 1000) if key == "nccl_port" else str(4 if key == "ep_size" else archived[fixture_port][key])
            # Decode appends a larger --max-total-tokens after the shared default; SGLang keeps the last value.
            value = flags[len(flags) - 1 - flags[::-1].index(flag) + 1]
            assert value == (str(config["decode_max_total_tokens"]) if key == "max_total_tokens" and not prefill else expected)
        assert flags[flags.index("--chunked-prefill-size") + 1] == (str(config["prefill"]["chunked_prefill_size"]) if prefill else "8192")
        assert megamoe_tokens == str(config["prefill"]["chunked_prefill_size"] // config["prefill"]["dp"] if prefill else 4096)
        assert flags[flags.index("--attention-backend") + 1] == "fa3"
        assert flags[flags.index("--page-size") + 1] == "64"
        assert flags[flags.index("--load-balance-method") + 1] == "total_requests"
        assert flags[flags.index("--load-snapshot-publish-interval") + 1] == "1"
        assert flags[flags.index("--max-running-requests") + 1] == str(config["concurrency"] if prefill else config["decode_max_running_requests"])
        location = [index for index, flag in enumerate(flags) if flag == "--init-expert-location"]
        assert len(location) == (1 if prefill else 0)
        if prefill:
            assert flags[location[0] + 1] == str(tmp_path / "source/prefill-expert-location.json")
            assert flags[flags.index("--ep-num-redundant-experts") + 1] == "12"
        else:
            assert "--ep-num-redundant-experts" not in flags
        assert flags[flags.index("--speculative-algorithm") + 1] == "EAGLE3"
        assert flags[flags.index("--speculative-draft-model-path") + 1] == "/draft"
        assert flags[flags.index("--speculative-num-steps") + 1] == "3"
        assert flags[flags.index("--speculative-eagle-topk") + 1] == "1"
        assert flags[flags.index("--speculative-num-draft-tokens") + 1] == "4"
        assert ("--speculative-draft-model-quantization" in flags) == (draft_quantization == "fp8")
        if draft_quantization == "fp8":
            assert flags.count("--speculative-draft-model-quantization") == 1
            assert flags[flags.index("--speculative-draft-model-quantization") + 1] == "fp8"
        assert "--container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER" in argv
        if name.startswith("prefill"):
            assert config["prefill"]["moe_a2a_backend"] == "megamoe"
            assert flags[flags.index("--deepep-mode") + 1] == "normal"
            assert flags[flags.index("--moe-runner-backend") + 1] == config["prefill"]["moe_runner_backend"]
            assert flags[flags.index("--moe-a2a-backend") + 1] == config["prefill"]["moe_a2a_backend"]
            assert flags[flags.index("--dp-size") + 1] == "4"
            assert flags[flags.index("--dist-init-addr") + 1] == f"127.0.0.1:{32000 + int(name[-1]) * 100}"
            assert "--enable-dp-attention" in flags and "--disable-cuda-graph" in flags
            assert "--enable-symm-mem" not in flags and (shuffle, routing) == ("0", "0")
            assert flags[flags.index("--optimistic-prefill-attempts") + 1] == str(config["prefill"]["optimistic_prefill_attempts"])
            assert aligned == "1" and rotate_ties == "1" and async_admission == "1"
        else:
            assert flags[flags.index("--dp-size") + 1] == "4"
            assert "--enable-dp-attention" in flags and "--enable-dp-lm-head" in flags
            assert flags[flags.index("--moe-dense-tp-size") + 1] == "1"
            assert flags[flags.index("--moe-a2a-backend") + 1] == "megamoe"
            assert flags[flags.index("--moe-runner-backend") + 1] == "deep_gemm"
            assert "--speculative-moe-a2a-backend" not in flags
            assert "--speculative-moe-runner-backend" not in flags
            assert flags[flags.index("--dist-init-addr") + 1] == "127.0.0.1:29000"
            assert "--disable-cuda-graph" not in flags
            assert flags[flags.index("--disaggregation-decode-extra-slots") + 1] == str(config["decode_extra_slots"])
            assert flags[flags.index("--disaggregation-decode-polling-interval") + 1] == str(config["decode_polling_interval"])
            start = flags.index("--cuda-graph-bs-decode") + 1
            assert [int(value) for value in flags[start:start + len(config["decode_cuda_graph_bs"])]] == config["decode_cuda_graph_bs"]
            assert "--enable-symm-mem" not in flags and (shuffle, routing) == ("0", "0")
            assert "--optimistic-prefill-attempts" not in flags and aligned == "0" and rotate_ties == "0" and async_admission == "1"
    assert observed == {"prefill-0": "node0", "prefill-1": "node0", "prefill-2": "node1", "decode-0": "node1"}
    router_argv = next(argv for argv in commands if "/opt/apac/bin/sglang-router" in argv)
    assert router_argv.count("--prefill") == 3 and router_argv.count("--decode") == 1
    assert router_argv[router_argv.index("--prefill-policy") + 1] == "manual"
    assert router_argv[router_argv.index("--assignment-mode") + 1] == "min_load"
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
    assert "requires the successful CPU build SHA-256 and size" in result.stderr
    assert not runtime.exists()


def test_firmus_provisioning_guards_accept_the_configured_speculation_and_count_shards(tmp_path: Path) -> None:
    # Run 39032066280 failed in provisioning because the guard pinned steps 3 / draft 4 while the config asked for 2 / 3.
    script = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    guard = re.search(r"jq -e '(\.speculation\.algorithm[^']*)' \"\$\{config\}\"", script).group(1)
    config = ROOT / "configs/firmus-framework.json"
    assert subprocess.run(["jq", "-e", guard, str(config)], capture_output=True, text=True, check=False).returncode == 0
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"speculation": {"algorithm": "EAGLE3", "steps": 3, "draft_tokens": 3}}))
    assert subprocess.run(["jq", "-e", guard, str(wrong)], capture_output=True, text=True, check=False).returncode == 1
    # The shard list must come from the weight_map values; `.weight_map | values | unique` errors on the object and silently skipped every shard.
    shards = re.search(r"jq -r '(\[\.weight_map\[\]\] \| unique\[\])' \"\$\{model\}/model\.safetensors\.index\.json\"", script).group(1)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"metadata": {}, "weight_map": {"a": "m2", "b": "m1", "c": "m2"}}))
    result = subprocess.run(["jq", "-r", shards, str(index)], capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["m1", "m2"]


@pytest.mark.parametrize("quantization", ["fp8", None, "unset", "bf16", "int8", False, 8, ["fp8"], {}])
def test_firmus_draft_quantization_guard(tmp_path: Path, quantization) -> None:
    script = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    guard = re.search(r"jq -e '(\.speculation\.algorithm[^']*)' \"\$\{config\}\"", script).group(1)
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    if quantization == "unset":
        config["speculation"].pop("draft_quantization", None)
    else:
        config["speculation"]["draft_quantization"] = quantization
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = subprocess.run(["jq", "-e", guard, str(path)], capture_output=True, text=True, check=False)
    assert (result.returncode == 0) == (quantization in ("fp8", None, "unset"))


def test_verified_kernel_reuse_skips_gpu_selection_and_target_gate(tmp_path: Path) -> None:
    script = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    begin = script.index('if [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]')
    fragment = script[begin:script.index('\nsrun --exact', begin)]
    source = tmp_path / "source"
    source.mkdir()
    (source / "reused-kernel-selection.json").write_text(json.dumps({"source_run": 40313004834}))
    for name in ("prefill", "decode"):
        (source / f"{name}-kernels.env").write_text("VERIFIED=1\n")
    # Any unexpected GPU launch or numeric/selector call must fail this fixture.
    harness = 'set -euo pipefail\nsrun() { echo unexpected-GPU >&2; return 97; }\n' + fragment
    result = subprocess.run(["bash", "-c", harness], env={**os.environ, "ROOT": str(tmp_path)}, capture_output=True, text=True, check=True)
    assert "Reusing verified target kernel selections from run 40313004834" in result.stdout
    assert "unexpected-GPU" not in result.stderr


@pytest.mark.parametrize("reported", ["fp8", None])
def test_draft_precision_is_verified_before_official_benchmark(tmp_path: Path, reported) -> None:
    script = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    check = next(line.strip() for line in script.splitlines() if '.speculative_draft_model_quantization == "fp8"' in line)
    assert script.index(check) < script.index('benchmark_status=0')
    (tmp_path / "source").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "source/firmus-framework.json").write_text(json.dumps({"speculation": {"draft_quantization": "fp8"}}))
    (tmp_path / "results/server-info-prefill-0.json").write_text(json.dumps({"speculative_draft_model_quantization": reported}))
    result = subprocess.run(["bash", "-c", 'set -euo pipefail\n'+check], env={**os.environ, "ROOT": str(tmp_path), "worker": "prefill-0"}, capture_output=True, text=True, check=False)
    assert (result.returncode == 0) == (reported == "fp8")
