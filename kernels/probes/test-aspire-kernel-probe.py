"""CPU fixtures for PBS trial bounds, fatal errors and terminal result gates."""
import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLATFORM = ROOT / "platforms/aspire2ap/kernel-probe"


def runtime(tmp_path):
    for name in ("source", "results", "logs", "scripts", "home/.local", "bin"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "home/.local/.setenv").write_text("")
    (tmp_path / "scripts/build-env.sh").write_text("")
    return {**os.environ, "RUNTIME_DIR": str(tmp_path), "HOME": str(tmp_path / "home"), "PATH": f"{tmp_path}/bin:{os.environ['PATH']}"}


@pytest.mark.parametrize("calibration,probe_exit", [(True, 0), (False, 0), (True, 7)])
def test_pbs_trial_protocol(tmp_path, calibration, probe_exit):
    env = runtime(tmp_path)
    (tmp_path / "source/provenance.json").write_text(json.dumps({"calibration": calibration}))
    probe = tmp_path / "source/probe"
    probe.write_text(f"#!/bin/bash\nprintf 'PASS fixture\\n'\nexit {probe_exit}\n")
    probe.chmod(0o755)
    # Observe timeout budgets without waiting; execute the real probe command.
    timeout = tmp_path / "bin/timeout"
    timeout.write_text('#!/bin/bash\necho "$1" >> "$RUNTIME_DIR/timeouts"\nshift\nexec "$@"\n')
    timeout.chmod(0o755)
    result = subprocess.run(["bash", str(PLATFORM / "run.pbs")], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == probe_exit, result.stderr
    assert (tmp_path / "results/exit-code").read_text().strip() == str(probe_exit)
    assert (tmp_path / "run-success").exists() == (probe_exit == 0)
    assert (tmp_path / "timeouts").read_text().splitlines() == (["165"] if calibration else ["90"] * 3)


@pytest.mark.parametrize("state,scheduler_exit,marker,passed,expected", [("Q", 0, False, False, 0), ("R", 0, False, False, 0), ("F", 0, True, True, 0), ("F", 7, True, True, 1), ("F", 0, False, True, 1), ("F", 0, True, False, 1)])
def test_pbs_terminal_and_numeric_gates(tmp_path, state, scheduler_exit, marker, passed, expected):
    env = runtime(tmp_path)
    env["PBS_JOB_ID"] = "123.pbs101"
    (tmp_path / "source/provenance.json").write_text('{"calibration":true}')
    (tmp_path / "results/exit-code").write_text("0\n")
    (tmp_path / "results/trial-1.log").write_text("PASS fixture\n" if passed else "numeric failure\n")
    if marker:
        (tmp_path / "run-success").touch()
    command = f'/opt/pbs/bin/qstat() {{ printf "job_state = {state}\\nExit_status = {scheduler_exit}\\n"; }}\nsource "$1"'
    result = subprocess.run(["bash", "-c", command, "fixture", str(PLATFORM / "check-job.sh")], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == expected, result.stdout + result.stderr
    if state != "F":
        assert result.stdout == f"scheduler_job_pending=123.pbs101 state={state}\n"


@pytest.mark.parametrize("benchmark,patch", [("kernels/probes/prepare-megamoe-full.sh", ""), ("kernels/probes/../../unsafe.sh", ""), ("kernels/probes/prepare-fa3-prefill.sh", "candidate.patch")])
def test_provision_rejects_unbounded_inputs_before_allocation(tmp_path, benchmark, patch):
    env = {**os.environ, "GITHUB_WORKSPACE": str(ROOT), "GITHUB_RUN_ID": "fixture", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_OUTPUT": str(tmp_path / "output"), "RUNTIME_ROOT": str(tmp_path / "runs"), "BENCHMARK": benchmark, "CANDIDATE_PATCH": patch}
    result = subprocess.run(["bash", str(ROOT / "scripts/provision-aspire-kernel-probe.sh")], env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "output").exists()


def test_summary_records_both_su_observations(tmp_path):
    env = runtime(tmp_path)
    (tmp_path / "logs/su-before.txt").write_text("Balance 8245.937\n")
    command = 'bash() { test "$1" = -lc; test "$2" = "myprojects -p 59010014"; echo "Balance 8239.937"; }\nsource "$1"'
    result = subprocess.run(["bash", "-c", command, "fixture", str(PLATFORM / "summary.sh")], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "8245.937" in result.stdout and "8239.937" in result.stdout
    assert "8239.937" in (tmp_path / "logs/su-after.txt").read_text()


@pytest.mark.parametrize("state,archive,removed", [("Q", "success", False), ("F", "failure", False), ("F", "success", True), ("", "failure", False), ("", "success", True)])
def test_cleanup_preserves_active_jobs_and_unrecovered_results(tmp_path, state, archive, removed):
    env = runtime(tmp_path)
    env["PBS_JOB_ID"] = "123.pbs101" if state else ""
    workflow = (ROOT / ".github/workflows/aspire-kernel-probe.yml").read_text()
    cleanup = textwrap.dedent(workflow.rsplit("        run: |\n", 1)[1]).replace("${{ steps.archive.outcome }}", archive)
    command = f'/opt/pbs/bin/qstat() {{ printf "job_state = {state}\\n"; }}\nset -euo pipefail\n' + cleanup
    result = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == (0 if removed else 1), result.stdout + result.stderr
    assert tmp_path.exists() != removed
