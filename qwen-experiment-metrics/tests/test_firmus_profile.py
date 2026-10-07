from __future__ import annotations

import json
import os
import runpy
import sqlite3
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VALIDATE = runpy.run_path(str(ROOT / "scripts/validate-firmus-profile.py"))["validate_capture"]
SUMMARY = runpy.run_path(str(ROOT / "scripts/summarize-firmus-run.py"))["summarize"]


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
@pytest.mark.parametrize("failure", [None, "start", "stop"])
def test_capture_stops_every_worker_even_after_partial_start(tmp_path: Path, failure: str | None, framework: str) -> None:
    requests = []
    request_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with request_lock:
                requests.append((self.path, json.loads(body) if body else None))
                number = len(requests)
            self.send_response(500 if self.path == f"/{failure}_profile" and number % 6 == 1 else 200)
            self.end_headers()
            self.wfile.write(b"response\n")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for directory in ("logs", "source", "profile", "bin"):
        (tmp_path / directory).mkdir()
    (tmp_path / "source/firmus-framework.json").write_text(json.dumps({"framework": framework}))
    workers = [(f"prefill-{n}", 2) for n in range(4)] + [(f"decode-{n}", 4) for n in range(2)]
    (tmp_path / "source/worker-endpoints.tsv").write_text("".join(f"{name}\t127.0.0.1\t{server.server_port}\t{tp}\n" for name, tp in workers))
    (tmp_path / "logs/aiperf.log").write_text("Phase profiling (profiling) started\n")
    sleep = tmp_path / "bin/sleep"
    sleep.write_text("#!/bin/bash\nprintf '%s\\n' \"$*\" >>\"$SLEEP_RECORD\"\n")
    sleep.chmod(0o755)
    try:
        result = subprocess.run(["bash", str(ROOT / "scripts/capture-firmus-profile.sh"), str(tmp_path)], env={**os.environ, "PATH": f"{tmp_path}/bin:{os.environ['PATH']}", "SLEEP_RECORD": str(tmp_path / "sleeps")}, capture_output=True, text=True, check=False)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert (result.returncode != 0) is (failure is not None)
    assert [path for path, _ in requests].count("/start_profile") == 6
    assert [path for path, _ in requests].count("/stop_profile") == 6
    expected_body = {"activities": ["CUDA_PROFILER"]} if framework == "sglang" else None
    assert all(body == expected_body for path, body in requests if path == "/start_profile")
    assert (tmp_path / "sleeps").read_text().splitlines() == (["5"] if failure == "start" else ["5", "5"])
    assert ("capture_stopped" in result.stdout) is (failure is None)


def test_profile_validation_rejects_missing_tp_rank(tmp_path: Path) -> None:
    database = tmp_path / "report.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER)")
        connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?)", [(0,), (0,), (1,)])
    assert VALIDATE(database, 2) == {"gpu_count": 2, "kernel_count": 3}
    with pytest.raises(ValueError, match="expected kernels on 4 GPUs"):
        VALIDATE(database, 4)


def test_profile_validation_explains_idle_worker_capture(tmp_path: Path) -> None:
    database = tmp_path / "nsys-prefill-1.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE TARGET_INFO_GPU (deviceId INTEGER)")
        connection.executemany("INSERT INTO TARGET_INFO_GPU VALUES (?)", [(0,), (1,)])
    with pytest.raises(ValueError, match=r"no CUDA kernels captured in nsys-prefill-1.sqlite; expected kernels on 2 GPUs"):
        VALIDATE(database, 2)


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
    (profile / "validation.json").write_text(json.dumps({"valid": True, "workers": {"prefill-0": {"gpu_count": 2}}}))
    summary = SUMMARY(tmp_path, "456", "failure")
    assert "1 / 6 reports" in summary and "2 / 16 GPUs" in summary
    assert "Instrumented score" in summary and "not an official submission" in summary
    assert "/qwen-artifacts/123/1/" in summary


def test_nsys_wrapper_preserves_launch_arguments(tmp_path: Path) -> None:
    stub = tmp_path / "nsys"
    stub.write_text("#!/bin/bash\nprintf '%s\\0' \"$@\" >\"$NSYS_RECORD\"\n")
    stub.chmod(0o755)
    invocation = tmp_path / "decode.argv"
    subprocess.run(["bash", str(ROOT / "scripts/exec-profiler.sh"), "/opt/python", "-m", "sglang.launch_server", "--port", "30200"], env={**os.environ, "QWEN_PROFILER_INVOCATION_PATH": str(invocation), "QWEN_PROFILER_NSYS_BIN": str(stub), "NSYS_RECORD": str(tmp_path / "argv")}, check=True)
    argv = (tmp_path / "argv").read_text().rstrip("\0").split("\0")
    assert "--capture-range=cudaProfilerApi" in argv and "--capture-range-end=stop" in argv
    assert "--cuda-graph-trace=node" in argv and "--sample=none" in argv
    assert argv[-5:] == ["/opt/python", "-m", "sglang.launch_server", "--port", "30200"]


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
