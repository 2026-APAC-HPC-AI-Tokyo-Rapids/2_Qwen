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
@pytest.mark.parametrize("worker_count", [1, 6, 7])
def test_capture_stops_every_worker_even_after_partial_start(tmp_path: Path, failure: str | None, framework: str, worker_count: int) -> None:
    requests = []
    request_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with request_lock:
                requests.append((self.path, json.loads(body) if body else None, "Phase profiling (profiling) complete |" in (tmp_path / "logs/aiperf.log").read_text()))
                number = len(requests)
            self.send_response(500 if self.path == f"/{failure}_profile" and ((number - 1) % worker_count == 0) else 200)
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
    workers = [(f"prefill-{n}", 2) for n in range(6)] + [("decode-0", 4)] if worker_count == 7 else [(f"prefill-{n}", 2) for n in range(4)] + [(f"decode-{n}", 4) for n in range(2)]
    (tmp_path / "source/worker-endpoints.tsv").write_text("".join(f"{name}\t127.0.0.1\t{server.server_port}\t{tp}\n" for name, tp in workers))
    if worker_count == 1:
        (tmp_path / "source/profile-worker-endpoints.tsv").write_text(f"decode-0\t127.0.0.1\t{server.server_port}\t4\n")
    (tmp_path / "logs/aiperf.log").write_text("Phase profiling (profiling) started\n")
    sleep = tmp_path / "bin/sleep"
    sleep.write_text("#!/bin/bash\nprintf '%s\\n' \"$*\" >>\"$SLEEP_RECORD\"\nif [[ $1 == 1 ]]; then printf 'Phase profiling (profiling) complete | elapsed=75.00s\\n' >>\"${SLEEP_RECORD%/*}/logs/aiperf.log\"; fi\n")
    sleep.chmod(0o755)
    try:
        result = subprocess.run(["bash", str(ROOT / "scripts/capture-firmus-profile.sh"), str(tmp_path)], env={**os.environ, "PATH": f"{tmp_path}/bin:{os.environ['PATH']}", "SLEEP_RECORD": str(tmp_path / "sleeps")}, capture_output=True, text=True, check=False)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert (result.returncode != 0) is (failure is not None)
    assert [path for path, _, _ in requests].count("/start_profile") == worker_count
    assert [path for path, _, _ in requests].count("/stop_profile") == worker_count
    expected_body = {"activities": ["CUDA_PROFILER"]} if framework == "sglang" else None
    assert all(body == expected_body for path, body, _ in requests if path == "/start_profile")
    assert (tmp_path / "sleeps").read_text().splitlines() == (["5"] if failure == "start" else ["5", "1" if framework == "sglang" else "5"])
    if framework == "sglang" and failure != "start":
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
    assert "GPU coverage unverified" in SUMMARY(tmp_path, "456", "failure")
    (profile / "interpretation.txt").write_text("Capture continues through benchmark and grace completion.\n")
    (profile / "validation.json").write_text(json.dumps({"valid": True, "workers": {"prefill-0": {"gpu_count": 2}}}))
    summary = SUMMARY(tmp_path, "456", "failure")
    assert "1 / 6 reports" in summary and "2 / 16 GPUs" in summary
    assert "Instrumented score" in summary and "not an official submission" in summary
    assert "Capture continues through benchmark and grace completion." in summary
    assert "/qwen-artifacts/123/1/" in summary


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


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_optimized_c64_summary_reports_ep1_topology(tmp_path: Path, framework: str) -> None:
    (tmp_path / "provenance.json").write_text(json.dumps({"framework": framework, "workload": "official"}))
    (tmp_path / "firmus-framework.json").write_text(json.dumps({"concurrency": 64, "speculation": {"algorithm": "EAGLE3"}}))
    assert "P4 × TP2 / D2 × TP4 EP1" in SUMMARY(tmp_path, "22463", "success")


@pytest.mark.parametrize("profile,expected", [("false", 0), ("true", 0), ("invalid", 1)])
def test_optimized_vllm_accepts_explicit_profile_before_worker_launch(profile: str, expected: int) -> None:
    source = (ROOT / "platforms/firmus/framework-comparison/vllm.sbatch").read_text()
    guard = next(line for line in source.splitlines() if line.startswith('[[ "${PROFILE:'))
    result = subprocess.run(["bash", "-euc", guard], env={**os.environ, "PROFILE": profile}, check=False)
    assert result.returncode == expected


def test_vllm_c104_summary_reports_seven_worker_topology(tmp_path: Path) -> None:
    (tmp_path / "provenance.json").write_text(json.dumps({"framework": "vllm", "workload": "official"}))
    (tmp_path / "firmus-framework.json").write_text(json.dumps({"concurrency": 104, "speculation": {"algorithm": "EAGLE3"}}))
    assert "P6 × TP2 / D1 × TP4 EP1" in SUMMARY(tmp_path, "new", "success")



def test_vllm_seven_worker_profile_validation(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    profile = tmp_path / "profile"
    source.mkdir()
    profile.mkdir()
    workers = [(f"prefill-{n}", 2) for n in range(6)] + [("decode-0", 4)]
    (source / "worker-endpoints.tsv").write_text("".join(f"{name}\t127.0.0.1\t30000\t{tp}\n" for name, tp in workers))
    for name, _tp in workers:
        (profile / f"nsys-{name}.nsys-rep").write_text("report")

    def export(command, **_kwargs):
        database = Path(next(value for value in command if value.startswith("--output=")).split("=", 1)[1])
        tp = 4 if "decode" in database.name else 2
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER)")
            connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?)", [(rank,) for rank in range(tp)])

    monkeypatch.setattr(subprocess, "run", export)
    monkeypatch.setattr("sys.argv", ["validate", str(tmp_path)])
    runpy.run_path(str(ROOT / "scripts/validate-firmus-profile.py"), run_name="__main__")
    validation = json.loads((profile / "validation.json").read_text())
    assert validation["valid"] is True
    assert len(validation["workers"]) == 7
    assert sum(worker["gpu_count"] for worker in validation["workers"].values()) == 16
    assert not list(profile.glob("*.sqlite"))


@pytest.mark.parametrize("selection", ["decode-0\t127.0.0.1\t30200\t4\n", "prefill-0\t127.0.0.1\t30100\t2\n", "decode-0\t127.0.0.1\t30200\t2\n"])
def test_decoder_only_profile_requires_selected_four_ranks(tmp_path: Path, monkeypatch, selection: str) -> None:
    source = tmp_path / "source"
    profile = tmp_path / "profile"
    source.mkdir()
    profile.mkdir()
    (source / "worker-endpoints.tsv").write_text("prefill-0\t127.0.0.1\t30100\t2\ndecode-0\t127.0.0.1\t30200\t4\n")
    (source / "profile-worker-endpoints.tsv").write_text(selection)
    (profile / "nsys-decode-0.nsys-rep").write_text("report")

    def export(command, **_kwargs):
        database = Path(next(value for value in command if value.startswith("--output=")).split("=", 1)[1])
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER)")
            connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?)", [(rank,) for rank in range(4)])

    monkeypatch.setattr(subprocess, "run", export)
    monkeypatch.setattr("sys.argv", ["validate", str(tmp_path)])
    if selection.startswith("decode-0") and selection.endswith("4\n"):
        runpy.run_path(str(ROOT / "scripts/validate-firmus-profile.py"), run_name="__main__")
        result = json.loads((profile / "validation.json").read_text())
        assert result["scope"] == "decode-0" and result["workers"]["decode-0"]["gpu_count"] == 4
        assert not list(profile.glob("*.sqlite"))
    else:
        with pytest.raises(ValueError, match="exactly decode-0 TP4"):
            runpy.run_path(str(ROOT / "scripts/validate-firmus-profile.py"), run_name="__main__")


def test_profile_rejects_missing_event_diagnostic(tmp_path: Path) -> None:
    database = tmp_path / "report.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER)")
        connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?)", [(rank,) for rank in range(4)])
        connection.execute("CREATE TABLE DIAGNOSTIC_EVENT (severity INTEGER, text TEXT)")
        connection.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (2, 'Not all CUDA events might have been collected')")
    with pytest.raises(ValueError, match="incomplete CUDA capture"):
        VALIDATE(database, 4)


@pytest.mark.parametrize("raw_error,coverage,expected", [(False, 4, 0), (True, 4, 1), (False, 3, 1)])
def test_profile_only_completion_requires_raw_and_selected_coverage(tmp_path: Path, raw_error: bool, coverage: int, expected: int) -> None:
    for name in ["bin", "logs", "source", "profile/client"]:
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "run-success").touch()
    (tmp_path / "source/provenance.json").write_text(json.dumps({"profile": True, "framework": "vllm", "measurement_mode": "profile-only"}))
    (tmp_path / "profile/client/profile_export.jsonl").write_text("{}\n")
    (tmp_path / "profile/client/profile_export_aiperf.json").write_text(json.dumps({"was_cancelled": False, "error_summary": ["failure"] if raw_error else [], "request_count": {"avg": 10}}))
    (tmp_path / "profile/validation.json").write_text(json.dumps({"valid": True, "scope": "decode-0", "workers": {"decode-0": {"gpu_count": coverage}}}))
    (tmp_path / "logs/slurm-1.log").write_text("profile diagnostic\n")
    for name, body in {"squeue": "#!/bin/sh\nexit 0\n", "sacct": "#!/bin/sh\necho '1|COMPLETED|0:0|00:12:00|2'\n"}.items():
        (tmp_path / "bin" / name).write_text(body)
        (tmp_path / "bin" / name).chmod(0o755)
    r = subprocess.run(["bash", str(ROOT / "platforms/firmus/check-qwen-job.sh")], check=False, env={**os.environ, "PATH": f"{tmp_path}/bin:{os.environ['PATH']}", "SLURM_JOB_ID": "1", "RUNTIME_DIR": str(tmp_path)}, capture_output=True, text=True)
    assert r.returncode == expected, r.stdout + r.stderr


def test_profile_only_path_does_not_run_official_eligibility_gate(tmp_path: Path) -> None:
    source = (ROOT / "platforms/firmus/framework-comparison/vllm.sbatch").read_text()
    start = source.index('# Profile-only execution is independent of official eligibility.')
    end = source.index('# Prometheus snapshots bracket the official pass;', start)
    for name in ["source", "scripts", "logs"]:
        (tmp_path / name).mkdir()
    (tmp_path / "source/worker-endpoints.tsv").write_text("prefill-0\t127.0.0.1\t30100\t2\ndecode-0\t127.0.0.1\t30200\t4\n")
    (tmp_path / "scripts/capture-firmus-profile.sh").write_text("#!/bin/sh\necho captured\n")
    script = 'set -euo pipefail\nROOT=$1; ART=$ROOT/logs; N=108; HEAD=head; AIPERF=client; VLLM=server; PROFILE=true\nmounts() { echo fixture; }\nsrun() { printf \'%s\\n\' "$*" >>"$ROOT/calls"; if [[ "$*" == *official-aiperf.sh* && "$*" != *\'108 profile\' ]]; then return 1; fi; }\n'
    result = subprocess.run(["bash", "-c", script + source[start:end] + '\nexit 99', "fixture", str(tmp_path)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "run-success").exists()
    calls = (tmp_path / "calls").read_text().splitlines()
    assert len(calls) == 2 and calls[0].endswith('108 profile') and 'validate-firmus-profile.py' in calls[1]
