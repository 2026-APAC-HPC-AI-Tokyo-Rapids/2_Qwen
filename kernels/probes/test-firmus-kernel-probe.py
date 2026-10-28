"""CPU fixtures for Firmus allocation, counter preflight, trial and cleanup gates."""
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLATFORM = ROOT / "platforms/firmus/kernel-probe"


def executable(path, source):
    path.write_text(source)
    path.chmod(0o755)


def runtime(tmp_path):
    for name in ("source", "results", "logs", "scripts", "bin"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "source/provenance.json").write_text('{"calibration":true,"probe_gpus":1,"benchmark":"kernels/probes/prepare-fa3-prefill.sh"}')
    shutil.copyfile(PLATFORM / "run-trials.sh", tmp_path / "scripts/run-trials.sh")
    return {**os.environ, "RUNTIME_DIR": str(tmp_path), "SLURM_JOB_ID": "123", "PATH": f"{tmp_path}/bin:{os.environ['PATH']}"}


@pytest.mark.parametrize("permission,probe_exit", [("allowed", 0), ("allowed", 7), ("denied", 0), ("timeout", 0), ("no-metric", 0)])
def test_slurm_step_and_preflight_stop_before_trial(tmp_path, permission, probe_exit):
    env = runtime(tmp_path)
    env["PERMISSION"] = permission
    executable(tmp_path / "bin/srun", '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
(Path(os.environ['RUNTIME_DIR'])/'srun-args.json').write_text(json.dumps(args))
command = args[args.index('env'):]
os.execvp(command[0], command)
''')
    executable(tmp_path / "bin/timeout", '''#!/bin/bash
printf '%s\n' "$1" >>"$RUNTIME_DIR/timeouts"
shift
exec "$@"
''')
    executable(tmp_path / "bin/ncu", '''#!/bin/bash
set -eu
test "$CUDA_VISIBLE_DEVICES" = 0
printf '%s\n' "$@" >"$RUNTIME_DIR/ncu-args"
case "$PERMISSION" in
  denied) echo '==ERROR== ERR_NVGPUCTRPERM' >&2; exit 9 ;;
  timeout) exit 124 ;;
  no-metric) echo 'No kernels profiled' ;;
  allowed) echo 'sm__cycles_elapsed.avg,1' ;;
esac
''')
    executable(tmp_path / "source/probe", f'''#!/bin/bash
test "$CUDA_VISIBLE_DEVICES" = 0
test "${{TRITON_CACHE_DIR#${{TMPDIR:-/tmp}}/firmus-probe.}}" != "$TRITON_CACHE_DIR"
printf '%s\n' "$TRITON_CACHE_DIR" >"$RUNTIME_DIR/cache-location"
printf 'probe stderr diagnostic\n' >&2
printf 'PASS fixture\n'
exit {probe_exit}
''')
    result = subprocess.run(["bash", str(PLATFORM / "run.sbatch")], env=env, capture_output=True, text=True, check=False)
    expected = probe_exit if permission == "allowed" else 124 if permission == "timeout" else 1
    assert result.returncode == expected, result.stdout + result.stderr
    assert (tmp_path / "results/exit-code").read_text().strip() == str(expected)
    assert (tmp_path / "run-success").exists() == (expected == 0)
    assert (tmp_path / "timeouts").read_text().splitlines() == (["30", "165"] if permission == "allowed" else ["30"])
    if permission == "allowed":
        assert "probe stderr diagnostic" in (tmp_path / "results/trial-1.log").read_text()
        assert not Path((tmp_path / "cache-location").read_text().strip()).parent.exists()
    else:
        assert not (tmp_path / "results/trial-1.log").exists()
    if permission == "denied":
        assert "FATAL: ERR_NVGPUCTRPERM" in (tmp_path / "logs/driver.log").read_text()
    args = json.loads((tmp_path / "srun-args.json").read_text())
    assert "--gres=gpu:1" in args and "-n1" in args and "--exact" in args
    mounts = next(arg for arg in args if arg.startswith("--container-mounts="))
    assert f"{tmp_path}/metadata:/opt/apac/fa3-verify:ro" in mounts
    assert ":/opt/fa3-ncu:ro" in mounts
    assert "--metrics\nsm__cycles_elapsed.avg\n" in (tmp_path / "ncu-args").read_text()


@pytest.mark.parametrize("active,state,code,marker,passed,expected", [("RUNNING", "", "", False, False, 75), ("", "COMPLETED", "0:0", True, True, 0), ("", "FAILED", "1:0", True, True, 1), ("", "COMPLETED", "0:0", False, True, 1), ("", "COMPLETED", "0:0", True, False, 1), ("", "", "", True, True, 1)])
def test_checker_requires_terminal_and_protocol_success(tmp_path, active, state, code, marker, passed, expected):
    env = runtime(tmp_path)
    executable(tmp_path / "bin/squeue", f"#!/bin/bash\necho '123|{active}'\n")
    executable(tmp_path / "bin/sacct", f"#!/bin/bash\necho '123|{state}|{code}|'\n")
    (tmp_path / "results/exit-code").write_text("0\n")
    (tmp_path / "results/trial-1.log").write_text("PASS fixture\n" if passed else "fatal error\n")
    if marker:
        (tmp_path / "run-success").touch()
    result = subprocess.run(["bash", str(PLATFORM / "check-job.sh")], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == expected, result.stdout + result.stderr
    if active:
        assert result.stdout == "scheduler_job_pending=123 state=RUNNING\n"


@pytest.mark.parametrize("active,state,archive,removed", [("RUNNING", "COMPLETED", "success", False), ("COMPLETING", "FAILED", "success", False), ("", "COMPLETED", "success", True), ("", "FAILED", "success", True), ("", "COMPLETED", "failure", False), ("", "RUNNING", "success", False), ("", "", "success", False)])
def test_cleanup_needs_terminal_no_active_job_and_archive(tmp_path, active, state, archive, removed):
    env = runtime(tmp_path)
    executable(tmp_path / "bin/squeue", f"#!/bin/bash\necho '123|{active}'\n")
    executable(tmp_path / "bin/sacct", f"#!/bin/bash\necho '123|{state}|'\n")
    workflow = (ROOT / ".github/workflows/firmus-kernel-probe.yml").read_text()
    cleanup = textwrap.dedent(workflow.rsplit("        run: |\n", 1)[1]).replace("${{ steps.archive.outcome }}", archive)
    result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + cleanup], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == (0 if removed else 1), result.stdout + result.stderr
    assert tmp_path.exists() != removed


@pytest.mark.parametrize("benchmark,patch", [("kernels/probes/prepare-megamoe-full.sh", ""), ("kernels/probes/../../unsafe.sh", ""), ("kernels/probes/prepare-fa3-prefill.sh", "candidate.patch")])
def test_provision_rejects_unbounded_inputs(tmp_path, benchmark, patch):
    env = {**os.environ, "GITHUB_WORKSPACE": str(ROOT), "GITHUB_RUN_ID": "fixture", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "fixture", "GITHUB_OUTPUT": str(tmp_path / "output"), "RUNTIME_ROOT": str(tmp_path / "runs"), "BENCHMARK": benchmark, "CANDIDATE_PATCH": patch}
    result = subprocess.run(["bash", str(ROOT / "scripts/provision-firmus-kernel-probe.sh")], env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "output").exists()


def test_m2_profiler_failure_reaches_firmus_trial_log(tmp_path):
    env = runtime(tmp_path)
    shutil.copyfile(ROOT / "kernels/probes/run-fa3-prefill-profile.sh", tmp_path / "source/probe")
    (tmp_path / "source/probe").chmod(0o755)
    executable(tmp_path / "bin/ncu", '''#!/bin/bash
if [[ "$1" == --import ]]; then touch "$RUNTIME_DIR/import-called"; exit 1; fi
echo 'sm__cycles_elapsed.avg,1'
''')
    executable(tmp_path / "scripts/run-container.sh", '''#!/bin/bash
echo '==ERROR== ERR_NVGPUCTRPERM - fixture profiler failure' >&2
echo 'PASS kernel=fa3_prefill_m2 child-only'
exit 9
''')
    result = subprocess.run(["bash", str(PLATFORM / "run-trials.sh")], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 9, result.stdout + result.stderr
    trial = (tmp_path / "results/trial-1.log").read_text()
    assert "ERR_NVGPUCTRPERM" in trial
    assert "PASS " not in trial
    assert not (tmp_path / "import-called").exists()
    assert not (tmp_path / "run-success").exists()


def test_eagle_uses_all_eight_gpus_without_counter_preflight(tmp_path):
    env = runtime(tmp_path)
    (tmp_path / "source/provenance.json").write_text('{"calibration":true,"probe_gpus":8,"benchmark":"kernels/probes/prepare-firmus-eagle-accept.sh"}')
    executable(tmp_path / "bin/srun", '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
(Path(os.environ['RUNTIME_DIR'])/'srun-args.json').write_text(json.dumps(args))
command = args[args.index('env'):]
os.execvp(command[0], command)
''')
    executable(tmp_path / "bin/timeout", '''#!/bin/bash
printf '%s\\n' "$1" >>"$RUNTIME_DIR/timeouts"
shift
exec "$@"
''')
    executable(tmp_path / "source/probe", '''#!/bin/bash
test "$CUDA_VISIBLE_DEVICES" = 0,1,2,3,4,5,6,7
echo 'PASS two TP4 workers'
''')
    result = subprocess.run(["bash", str(PLATFORM / "run.sbatch")], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads((tmp_path / "srun-args.json").read_text())
    assert "--gres=gpu:8" in args
    assert "--cpus-per-task=32" in args
    mounts = next(arg for arg in args if arg.startswith("--container-mounts="))
    assert "batch_result_processor.py:ro" in mounts
    assert ":/model:ro" in mounts
    assert ":/opt/fa3-ncu:ro" not in mounts
    assert (tmp_path / "timeouts").read_text().splitlines() == ["960"]
    assert not (tmp_path / "results/ncu-permission.csv").exists()
    assert (tmp_path / "run-success").exists()
