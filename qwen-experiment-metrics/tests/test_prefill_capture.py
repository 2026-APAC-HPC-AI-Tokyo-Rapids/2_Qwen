from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).parents[2]


@pytest.mark.parametrize("failed_stop,failed_start", [(False, False), (True, False), (False, True)])
def test_finite_capture_parallel_prefill_only_and_stop_failure(tmp_path: Path, failed_stop: bool, failed_start: bool) -> None:
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            requests.append((self.path, json.loads(body) if body else None))
            self.send_response(500 if (failed_stop and self.path == "/stop_profile") or (failed_start and self.path == "/start_profile" and len(requests) == 1) else 200)
            self.end_headers()
            self.wfile.write(b"profiling response\n")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for directory in ("logs", "source", "profile", "bin"):
        (tmp_path / directory).mkdir()
    port = server.server_port
    (tmp_path / "source/worker-endpoints.tsv").write_text(f"prefill\t127.0.0.1\t{port}\t9000\nprefill\t127.0.0.1\t{port}\t9001\ndecode\t127.0.0.1\t1\tunused\n")
    (tmp_path / "logs/aiperf.log").write_text("Phase warmup (warmup) completed\nPhase profiling (profiling) started\n")
    sleep = tmp_path / "bin/sleep"
    sleep.write_text("#!/bin/bash\nprintf '%s\\n' \"$*\" >>\"$SLEEP_RECORD\"\n")
    sleep.chmod(0o755)
    try:
        result = subprocess.run(["bash", str(REPOSITORY / "scripts/capture-prefill-profile.sh"), str(tmp_path)], env={**os.environ, "PATH": f"{tmp_path}/bin:{os.environ['PATH']}", "SLEEP_RECORD": str(tmp_path / "sleeps")}, capture_output=True, text=True, check=False)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert (result.returncode != 0) is (failed_stop or failed_start)
    assert [path for path, _ in requests] == ["/start_profile", "/start_profile", "/stop_profile", "/stop_profile"]
    for path, body in requests[:2]:
        assert path == "/start_profile"
        assert body["activities"] == ["CPU", "CUDA_PROFILER"]
        assert body["record_shapes"] is False
        assert body["with_stack"] is False
    assert (tmp_path / "sleeps").read_text().splitlines() == (["5"] if failed_start else ["5", "5"])
    assert (tmp_path / "profile/start-prefill-0.txt").exists()
    assert (tmp_path / "profile/start-prefill-1.txt").exists()
    assert ("capture_stopped" in result.stdout) is not (failed_stop or failed_start)


def test_profiler_exec_actual_nsys_options_and_target_argv(tmp_path: Path) -> None:
    stub = tmp_path / "nsys"
    stub.write_text("#!/bin/bash\nprintf '%s\\0' \"$@\" >\"$NSYS_RECORD\"\n")
    stub.chmod(0o755)
    invocation = tmp_path / "prefill.argv"
    subprocess.run(["bash", str(REPOSITORY / "scripts/exec-profiler.sh"), "/opt/python", "-m", "sglang.launch_server", "--port", "30100"], env={**os.environ, "QWEN_PROFILER_INVOCATION_PATH": str(invocation), "QWEN_PROFILER_NSYS_BIN": str(stub), "NSYS_RECORD": str(tmp_path / "argv")}, check=True)
    argv = (tmp_path / "argv").read_text().rstrip("\0").split("\0")
    assert "--capture-range=cudaProfilerApi" in argv
    assert "--capture-range-end=stop" in argv
    assert "--cuda-graph-trace=node" in argv
    assert argv[-5:] == ["/opt/python", "-m", "sglang.launch_server", "--port", "30100"]
    assert "--sample=none" in argv and "--cpuctxsw=none" in argv
    assert "sglang.launch_server" in invocation.read_text()

