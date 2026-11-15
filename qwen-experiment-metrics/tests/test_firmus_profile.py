from __future__ import annotations

import gzip
import json
import os
import runpy
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VALIDATE = runpy.run_path(str(ROOT / "scripts/validate-firmus-profile.py"))["validate_capture"]
SUMMARY = runpy.run_path(str(ROOT / "scripts/summarize-firmus-run.py"))["summarize"]


@pytest.mark.parametrize("framework,separate,lightweight", [("sglang", False, False), ("vllm", False, False), ("sglang", True, False), ("sglang", True, True)])
@pytest.mark.parametrize("failure", [None, "start", "stop"])
def test_capture_stops_every_worker_even_after_partial_start(tmp_path: Path, failure: str | None, framework: str, separate: bool, lightweight: bool) -> None:
    requests = []
    request_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with request_lock:
                requests.append((self.path, json.loads(body) if body else None, "Phase profiling (profiling) complete |" in (tmp_path / "logs/aiperf.log").read_text()))
                number = len(requests)
            self.send_response(500 if self.path == f"/{failure}_profile" and number % len(workers) == 1 else 200)
            self.end_headers()
            self.wfile.write(b"response\n")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for directory in ("logs", "source", "profile", "bin"):
        (tmp_path / directory).mkdir()
    (tmp_path / "source/firmus-framework.json").write_text(json.dumps({"framework": framework, "workload": "official" if separate else "reference-07", "profile_only": lightweight, "prefill": {"pp": 4 if lightweight else 1}}))
    workers = [(f"prefill-{n}", 2) for n in range(6 if separate else 4)] + [(f"decode-{n}", 4) for n in range(1 if separate else 2)]
    (tmp_path / "source/worker-endpoints.tsv").write_text("".join(f"{name}\t127.0.0.1\t{server.server_port}\t{tp}\n" for name, tp in workers))
    (tmp_path / "logs/aiperf.log").write_text("Phase profiling (profiling) started\n")
    client_log = tmp_path / "logs" / ("profile-aiperf.log" if separate else "aiperf.log")
    sleep = tmp_path / "bin/sleep"
    sleep.write_text("#!/bin/bash\nprintf '%s\\n' \"$*\" >>\"$SLEEP_RECORD\"\nif [[ $1 == 1 ]]; then printf 'Phase profiling (profiling) started\\nPhase profiling (profiling) complete | elapsed=75.00s\\n' >>\"$CLIENT_LOG\"; fi\n")
    sleep.chmod(0o755)
    try:
        result = subprocess.run(["bash", str(ROOT / "scripts/capture-firmus-profile.sh"), str(tmp_path)], env={**os.environ, "PATH": f"{tmp_path}/bin:{os.environ['PATH']}", "SLEEP_RECORD": str(tmp_path / "sleeps"), "CLIENT_LOG": str(client_log)}, capture_output=True, text=True, check=False)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert (result.returncode != 0) is (failure is not None)
    assert [path for path, _, _ in requests].count("/start_profile") == len(workers)
    assert [path for path, _, _ in requests].count("/stop_profile") == len(workers)
    expected_body = {"activities": ["CUDA_PROFILER"]} if framework == "sglang" else None
    if separate and not lightweight:
        bodies = [body for path, body, _ in requests if path == "/start_profile"]
        assert {body["output_dir"] for body in bodies} == {str(tmp_path / "profile/torch" / name) for name, _ in workers}
        assert all(body["activities"] == ["CPU", "CUDA_PROFILER"] and body["with_stack"] is False and body["record_shapes"] is False for body in bodies)
    else:
        assert all(body == expected_body for path, body, _ in requests if path == "/start_profile")
    expected_sleeps = (["1"] if separate else []) + (["5"] if failure == "start" else ["5", "1" if framework == "sglang" and not separate else "8" if lightweight else "5"])
    assert (tmp_path / "sleeps").read_text().splitlines() == expected_sleeps
    if framework == "sglang" and not separate and failure != "start":
        assert all(complete for path, _, complete in requests if path == "/stop_profile")
    assert ("capture_stopped" in result.stdout) is (failure is None)


def test_profile_validation_rejects_missing_tp_rank(tmp_path: Path) -> None:
    database = tmp_path / "report.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER)")
        connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?)", [(0,), (0,), (1,)])
    assert VALIDATE(database, 2) == {"gpu_count": 2, "kernel_count": 3}
    with pytest.raises(ValueError, match="expected kernels on 4 GPUs"):
        VALIDATE(database, 4)


def test_pp4_profile_requires_each_pipeline_stage(tmp_path: Path) -> None:
    validate = runpy.run_path(str(ROOT / "scripts/validate-firmus-profile.py"))["validate_cpu_traces"]
    for pp in range(4):
        with gzip.open(tmp_path / f"123-TP-0-PP-{pp}.trace.json.gz", "wt") as handle:
            json.dump({"traceEvents": [{"cat": "user_annotation", "name": "pp_send"}]}, handle)
    result = validate(tmp_path, 1, 4)
    assert result == {"cpu_trace_count": 4, "cpu_stage_names": ["pp_send"]}
    (tmp_path / "123-TP-0-PP-2.trace.json.gz").unlink()
    with pytest.raises(ValueError, match="TP0/PP2"):
        validate(tmp_path, 1, 4)


def test_profile_validation_explains_idle_worker_capture(tmp_path: Path) -> None:
    database = tmp_path / "nsys-prefill-1.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE TARGET_INFO_GPU (deviceId INTEGER)")
        connection.executemany("INSERT INTO TARGET_INFO_GPU VALUES (?)", [(0,), (1,)])
    with pytest.raises(ValueError, match=r"no CUDA kernels captured in nsys-prefill-1.sqlite; expected kernels on 2 GPUs"):
        VALIDATE(database, 2)


def test_full_profile_retains_seven_reports_and_sixteen_cpu_traces(tmp_path: Path, monkeypatch) -> None:
    module = runpy.run_path(str(ROOT / "scripts/validate-firmus-profile.py"))
    (tmp_path / "source").mkdir()
    (tmp_path / "profile").mkdir()
    (tmp_path / "source/firmus-framework.json").write_text(json.dumps({"framework": "sglang", "workload": "official"}))
    workers = [(f"prefill-{n}", 2) for n in range(6)] + [("decode-0", 4)]
    (tmp_path / "source/worker-endpoints.tsv").write_text("".join(f"{name}\thost\t30000\t{tp}\n" for name, tp in workers))
    for name, tp in workers:
        (tmp_path / f"profile/nsys-{name}.nsys-rep").write_text("native report")
        directory = tmp_path / "profile/torch" / name
        directory.mkdir(parents=True)
        for rank in range(tp):
            with gzip.open(directory / f"123-TP-{rank}.trace.json.gz", "wt") as handle:
                json.dump({"traceEvents": [{"cat": "user_annotation", "name": "draft_extend"}]}, handle)

    def export(command, check):
        assert check and command[:3] == ["/opt/nsys/bin/nsys", "export", "--type=sqlite"]
        output = Path(command[3].removeprefix("--output="))
        tp = 4 if "decode" in output.name else 2
        with sqlite3.connect(output) as connection:
            connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER, start INTEGER, end INTEGER, demangledName INTEGER)")
            connection.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
            connection.execute("INSERT INTO StringIds VALUES (1,'fused_moe_kernel')")
            connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,0,100,1)", [(rank,) for rank in range(tp)])

    monkeypatch.setattr(subprocess, "run", export)
    monkeypatch.setattr(sys, "argv", ["validate", str(tmp_path)])
    module["main"]()
    result = json.loads((tmp_path / "profile/validation.json").read_text())
    assert result["valid"] and len(result["workers"]) == 7
    assert sum(worker["gpu_count"] for worker in result["workers"].values()) == 16
    assert sum(worker["cpu_trace_count"] for worker in result["workers"].values()) == 16
    assert len(json.loads((tmp_path / "profile/kernel-summary.json").read_text())) == 7
    assert not list((tmp_path / "profile").glob("*.sqlite"))
    (tmp_path / "profile/torch/decode-0/123-TP-3.trace.json.gz").unlink()
    with pytest.raises(ValueError, match="expected one CPU trace.*TP3"):
        module["validate_cpu_traces"](tmp_path / "profile/torch/decode-0", 4)


def test_summary_marks_profile_score_and_handles_missing_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    assert "unavailable" in SUMMARY(tmp_path, "456", "failure")
    source = tmp_path / "source/source"
    source.mkdir(parents=True)
    (source / "provenance.json").write_text(json.dumps({"framework": "sglang", "profile": True}))
    (source / "firmus-framework.json").write_text(json.dumps({"concurrency": 8}))
    profile = tmp_path / "profile/profile"
    profile.mkdir(parents=True)
    (profile / "nsys-prefill-0.nsys-rep").write_text("report")
    assert "GPU coverage unverified" in SUMMARY(tmp_path, "456", "failure")
    (profile / "interpretation.txt").write_text("Capture continues through benchmark and grace completion.\n")
    (profile / "validation.json").write_text(json.dumps({"valid": True, "workers": {"prefill-0": {"gpu_count": 2}}}))
    summary = SUMMARY(tmp_path, "456", "failure")
    assert "1 / 6 reports" in summary and "2 / 16 GPUs" in summary
    assert "Instrumented score" in summary and "not an official submission" in summary
    assert "Capture continues through benchmark and grace completion." in summary
    assert "measurement `123/1`" in summary
    assert "/qwen-artifacts/" not in summary


def test_summary_uses_official_score_and_measured_latency(tmp_path: Path) -> None:
    (tmp_path / "provenance.json").write_text(json.dumps({"framework": "vllm", "workload": "official"}))
    (tmp_path / "score.json").write_text(json.dumps({"request_count": 360, "good_request_fraction": 1.0, "goodput_requests_per_second": 3.0, "score_output_tokens_per_second": 600.0, "eligible": True, "official_validation": {"valid": True}}))
    (tmp_path / "profile_export_aiperf.json").write_text(json.dumps({"good_request_count": {"avg": 360}, "time_to_first_token": {"avg": 330, "p99": 580}, "inter_token_latency": {"avg": 12, "p99": 13}}))
    summary = SUMMARY(tmp_path, "456", "success")
    assert "60 / 120 / 30" in summary and "reference-07" not in summary
    assert "360 / 360" in summary and "600.000 output tok/s" in summary
    assert "330.000 / 580.000 ms" in summary and "12.000 / 13.000 ms" in summary
    assert "Official workload validation | True" in summary
    (tmp_path / "provenance.json").write_text(json.dumps({"framework": "sglang", "workload": "official"}))
    assert "P2 × TP4 / D2 × TP4 EP1" in SUMMARY(tmp_path, "456", "success")


@pytest.mark.parametrize("has_score", [True, False])
def test_aspire_summary_reads_collected_results_and_bootstrap_endpoints(tmp_path: Path, has_score: bool) -> None:
    source = tmp_path / "source/source"
    source.mkdir(parents=True)
    (source / "provenance.json").write_text(json.dumps({"cluster": "aspire2ap", "profile": False}))
    config = {"concurrency": 104, "dp_size": 4, "workers": [{"role": role, "tp": 4, "ep": 4} for role in ("prefill", "prefill", "decode", "decode")]}
    (source / "aspire-serving.json").write_text(json.dumps(config))
    (source / "worker-endpoints.tsv").write_text("prefill\thost\t30100\t9000\ndecode\thost\t30200\tunused\n")
    if has_score:
        results = tmp_path / "results/results"
        results.mkdir(parents=True)
        (results / "score.json").write_text(json.dumps({"request_count": 3178, "error_request_count": 0, "good_request_fraction": 0.985840151, "goodput_requests_per_second": 25.382131849, "score_output_tokens_per_second": 5076.42636986, "eligible": True, "official_validation": {"valid": True}}))
    status = "success" if has_score else "failure"
    summary = subprocess.check_output([sys.executable, str(ROOT / "scripts/summarize-firmus-run.py"), str(tmp_path), "--cluster", "aspire2ap", "--job-id", "221216.pbs111", "--status", status], text=True)
    assert "ASPIRE 2A+ Qwen" in summary and "PBS `221216.pbs111`" in summary
    assert "H100 × 16" in summary and "P2 × TP4 DP4 EP4 / D2 × TP4 DP4 EP4" in summary
    assert "60 / 120 / 30" in summary and "concurrency **104**" in summary
    if has_score:
        assert "5076.426 output tok/s" in summary and "98.58%" in summary
        assert "Official workload validation | True" in summary and "SLA eligible | True" in summary
        assert "Request errors | 0" in summary
    else:
        assert "**failure**" in summary and "Validated AIPerf result unavailable" in summary
        assert "ASPIRE 2A+ Qwen" in SUMMARY(tmp_path / "missing", "221216.pbs111", "failure", "aspire2ap")


def test_full_profile_summary_keeps_normal_latency_and_actual_topology(tmp_path: Path) -> None:
    (tmp_path / "provenance.json").write_text(json.dumps({"framework": "sglang", "workload": "official", "profile": True, "normal_benchmark_before_profile": True}))
    (tmp_path / "score.json").write_text(json.dumps({"request_count": 360, "good_request_fraction": 1.0, "goodput_requests_per_second": 3.0, "score_output_tokens_per_second": 600.0, "eligible": True}))
    normal = tmp_path / "results/results"
    normal.mkdir(parents=True)
    (normal / "profile_export_aiperf.json").write_text(json.dumps({"inter_token_latency": {"avg": 12, "p99": 13}}))
    instrumented = tmp_path / "profile/client"
    instrumented.mkdir(parents=True)
    (instrumented / "profile_export_aiperf.json").write_text(json.dumps({"inter_token_latency": {"avg": 80, "p99": 90}}))
    workers = {**{f"prefill-{i}": {"gpu_count": 2} for i in range(6)}, "decode-0": {"gpu_count": 4}}
    (tmp_path / "worker-endpoints.tsv").write_text("".join(f"{name}\thost\t30000\t{info['gpu_count']}\n" for name, info in workers.items()))
    (tmp_path / "validation.json").write_text(json.dumps({"valid": True, "workers": workers}))
    for name in workers:
        (tmp_path / f"nsys-{name}.nsys-rep").touch()
    summary = SUMMARY(tmp_path, "456", "success")
    assert "P6 × TP2 / D1 × TP4" in summary and "7 / 7 reports" in summary and "16 / 16 GPUs" in summary
    assert "Normal score is from the earlier tracing-disabled pass" in summary
    assert "12.000 / 13.000 ms" in summary and "80.000 / 90.000 ms" not in summary
    (normal / "profile_export_aiperf.json").unlink()
    assert "80.000 / 90.000 ms" not in SUMMARY(tmp_path, "456", "failure")


def test_sglang_nsys_launch_preserves_uninstrumented_worker_arguments(tmp_path: Path) -> None:
    (tmp_path / "profile").mkdir()
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/bash\nprintf 'GPU-fixture, 00000000:01:00.0\\n'\n")
    smi.chmod(0o755)
    (tmp_path / "source").mkdir()
    (tmp_path / "source/decode-kernels.env").write_text("")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/exec-profiler.sh").write_bytes((ROOT / "scripts/exec-profiler.sh").read_bytes())
    stub = '#!/bin/bash\nprintf "%s\\0" "$@" >"$ARGV_RECORD"\n'
    python = tmp_path / "python"
    python.write_text(stub)
    python.chmod(0o755)
    nsys = tmp_path / "nsys"
    source = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    worker = source.split("<<'EOF'\n", 1)[1].split("\nEOF", 1)[0]
    launch = worker.split("UCX_IB=", 1)[0] + "command=(" + worker.split("command=(", 1)[1]
    launch = launch.replace("/opt/nsys/bin/nsys", str(nsys)).replace("/opt/sglang/venv/bin/python", str(python))
    path = tmp_path / "source/worker.sh"
    path.write_text(launch)
    args = ["bash", str(path), "decode-0", "30200", "/model", "4", "192.0.2.1", "--speculative-algorithm", "EAGLE3"]
    environment = {**os.environ, "ARGV_RECORD": str(tmp_path / "argv"), "PATH": f"{tmp_path}:{os.environ['PATH']}", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    subprocess.run(args, env=environment, check=True)
    baseline = (tmp_path / "argv").read_text().rstrip("\0").split("\0")
    nsys.write_text(stub)
    nsys.chmod(0o755)
    subprocess.run(args, env=environment, check=True)
    instrumented = (tmp_path / "argv").read_text().rstrip("\0").split("\0")
    assert len((tmp_path / "profile/gpu-map-decode-0.csv").read_text().splitlines()) == 4
    assert "--trace-fork-before-exec=true" in instrumented and "--cuda-graph-trace=node" in instrumented
    assert instrumented[instrumented.index(str(python)) + 1:] == baseline


@pytest.mark.parametrize("counter_access", [False, True])
def test_nsys_wrapper_preserves_launch_arguments(tmp_path: Path, counter_access: bool) -> None:
    stub = tmp_path / "nsys"
    stub.write_text("#!/bin/bash\nprintf '%s\\0' \"$@\" >\"$NSYS_RECORD\"\n")
    stub.chmod(0o755)
    grep = tmp_path / "grep"
    grep.write_text(f"#!/bin/bash\nexit {0 if counter_access else 1}\n")
    grep.chmod(0o755)
    invocation = tmp_path / "decode.argv"
    subprocess.run(["bash", str(ROOT / "scripts/exec-profiler.sh"), "/opt/python", "-m", "sglang.launch_server", "--port", "30200"], env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "QWEN_PROFILER_INVOCATION_PATH": str(invocation), "QWEN_PROFILER_NSYS_BIN": str(stub), "NSYS_RECORD": str(tmp_path / "argv")}, check=True)
    argv = (tmp_path / "argv").read_text().rstrip("\0").split("\0")
    assert "--capture-range=cudaProfilerApi" in argv and "--capture-range-end=stop" in argv
    assert "--cuda-graph-trace=node" in argv and "--sample=none" in argv
    assert "--trace=cuda,nvtx" in argv and "--cuda-trace-all-apis=false" in argv
    assert "--trace=cuda,nvtx,osrt" not in argv
    assert (tmp_path / "decode.nsys-command.argv").is_file()
    assert argv[-5:] == ["/opt/python", "-m", "sglang.launch_server", "--port", "30200"]
    assert ("--gpu-metrics-devices=cuda-visible" in argv) is counter_access
    assert ("--gpu-metrics-frequency=1000" in argv) is counter_access
    assert (tmp_path / "decode.gpu-metrics.txt").read_text().startswith("requested:" if counter_access else "skipped:")


@pytest.mark.parametrize("profile", [False, True])
def test_vllm_launch_keeps_baseline_arguments_and_enables_cuda_only_for_profile(tmp_path: Path, profile: bool) -> None:
    (tmp_path / "source").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/exec-profiler.sh").write_bytes((ROOT / "scripts/exec-profiler.sh").read_bytes())
    stub = '#!/bin/bash\nprintf "%s\\0" "$@" >"$ARGV_RECORD"\nprintf "%s" "${VLLM_WORKER_MULTIPROC_METHOD:-}" >"$METHOD_RECORD"\n'
    vllm = tmp_path / "vllm"
    vllm.write_text(stub)
    vllm.chmod(0o755)
    nsys = tmp_path / "nsys"
    if profile:
        nsys.write_text(stub)
        nsys.chmod(0o755)
    source = (ROOT / "platforms/firmus/framework-comparison/vllm.sbatch").read_text()
    launch = 'runtime_dir=' + source.split('runtime_dir=', 1)[1].split('\nEOF', 1)[0]
    launch = launch.replace('/opt/nsys/bin/nsys', str(nsys)).replace('/usr/local/bin/vllm', str(vllm))
    kv_config = '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
    worker = tmp_path / "source/worker.sh"
    worker.write_text('set -euo pipefail\nname=$1\nport=$2\nmodel=$3\ntp=$4\nshift 4\nKV_CONFIG=\'' + kv_config + "'\n" + launch)
    subprocess.run(["bash", str(worker), "decode-0", "30200", "/model", "4", "--enable-expert-parallel"], env={**os.environ, "ARGV_RECORD": str(tmp_path / "argv"), "METHOD_RECORD": str(tmp_path / "method"), "VLLM_WORKER_MULTIPROC_METHOD": ""}, check=True)
    argv = (tmp_path / "argv").read_text().rstrip("\0").split("\0")
    if profile:
        assert "--capture-range=cudaProfilerApi" in argv
        argv = argv[argv.index(str(vllm)) + 1:]
    expected = ["serve", "/model", "--served-model-name", "qwen", "--trust-remote-code", "--host", "0.0.0.0", "--port", "30200", "--tensor-parallel-size", "4", "--max-model-len", "8192", "--gpu-memory-utilization", "0.90", "--kv-transfer-config", kv_config, "--enable-expert-parallel"]
    assert argv == expected + (["--profiler-config", '{"profiler":"cuda"}'] if profile else [])
    assert (tmp_path / "method").read_text() == ("spawn" if profile else "")


def test_dp_ep_cpu_trace_names_require_all_global_tp_ranks(tmp_path):
    module = runpy.run_path(str(ROOT / "scripts/validate-firmus-profile.py"))
    for rank in range(4):
        with gzip.open(tmp_path / f"123-TP-{rank}-DP-{rank}-EP-{rank}.trace.json.gz", "wt") as handle:
            json.dump({"traceEvents": [{"cat": "user_annotation", "name": "target_verify"}]}, handle)
    assert module["validate_cpu_traces"](tmp_path, 4)["cpu_trace_count"] == 4
    (tmp_path / "123-TP-3-DP-3-EP-3.trace.json.gz").unlink()
    with pytest.raises(ValueError, match="TP3"):
        module["validate_cpu_traces"](tmp_path, 4)
