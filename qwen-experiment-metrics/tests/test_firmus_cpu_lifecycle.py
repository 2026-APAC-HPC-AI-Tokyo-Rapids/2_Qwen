"""Execute provision traps and Slurm cancellation with CPU command fixtures."""

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def executable(path, text):
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + text)
    path.chmod(0o755)


@pytest.mark.parametrize("ending,status,preserved", [("exit 7", 7, False), ("kill -TERM $$", 143, False), ("kill -INT $$", 130, False), ("exit 0", 0, False), ("provision_complete=true", 0, True)])
def test_provision_owns_cleanup_and_progress(tmp_path, ending, status, preserved):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "config.json").write_text(json.dumps({"framework": "sglang", "workload": "official", "concurrency": 1}))
    root = tmp_path / "runs"
    sibling = root / "unrelated"
    sibling.mkdir(parents=True)
    output = tmp_path / "output"
    env = dict(os.environ, GITHUB_WORKSPACE=str(workspace), GITHUB_RUN_ID="123", GITHUB_RUN_ATTEMPT="1", GITHUB_SHA="fixture", GITHUB_OUTPUT=str(output), RUNTIME_ROOT=str(root), PROFILE="false", REQUIRE_BUILD="false")
    script = (REPO / "scripts/provision-firmus-framework.sh").read_text().split('install -m 0644 "${config}"')[0]
    script += '\nsleep 60 &\nprogress_pid=$!\nprintf "%s\\n" "${progress_pid}" >"${RUNTIME_ROOT}/progress-pid"\n' + ending + "\n"
    result = subprocess.run(["bash", "-c", script, "provision", "config.json"], env=env, capture_output=True, check=False, text=True, timeout=10)
    assert result.returncode == status, result.stderr
    runtime = Path(output.read_text().strip().split("=", 1)[1])
    assert runtime.exists() == preserved
    assert sibling.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int((root / "progress-pid").read_text()), 0)


def workflow_step(step_id):
    workflow = (REPO / ".github/workflows/firmus.yml").read_text()
    step = workflow.split(f"        id: {step_id}\n", 1)[1].split("\n      - name:", 1)[0]
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


@pytest.mark.parametrize("cancel,state,expected", [("true", "CANCELLED", 0), ("false", "COMPLETED", 0), ("true", "RUNNING", 1), ("true", "", 1)])
def test_exact_job_cancellation_and_bounded_terminal_wait(tmp_path, cancel, state, expected):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(bin_dir / "scancel", 'printf "%s\\n" "$*" >>"${COMMAND_LOG}"\n')
    executable(bin_dir / "sacct", 'printf "%s|%s\\n" "${OWN_JOB_ID}" "${FIXTURE_STATE}"\n')
    executable(bin_dir / "sleep", 'printf "sleep\\n" >>"${WAIT_LOG}"\n')
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", OWN_JOB_ID="27159", CANCEL_JOB=cancel, FIXTURE_STATE=state, COMMAND_LOG=str(tmp_path / "commands"), WAIT_LOG=str(tmp_path / "waits"))
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", workflow_step("terminal")], env=env, capture_output=True, check=False, text=True, timeout=10)
    assert result.returncode == expected, result.stderr
    commands = tmp_path / "commands"
    assert (commands.read_text() if commands.exists() else "") == ("27159\n" if cancel == "true" else "")
    waits = tmp_path / "waits"
    assert (len(waits.read_text().splitlines()) if waits.exists() else 0) == (24 if expected else 0)


@pytest.mark.parametrize("report", [None, {"result": "fail"}, {"schema": "pp4-cpu-gate/v1", "result": "pass", "image_sha256": "stale"}, {"schema": "pp4-cpu-gate/v1", "result": "pass", "image_sha256": "image"}])
def test_submission_requires_pp4_pass_before_sbatch(tmp_path, report):
    (tmp_path / "source").mkdir()
    (tmp_path / "source/firmus-framework.json").write_text(json.dumps({"prefill": {"pp": 4}, "images": {"sglang": {"sha256": "image"}}}))
    if report is not None:
        (tmp_path / "source/pp4-cpu-gate.json").write_text(json.dumps(report))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(bin_dir / "sbatch", 'touch "${RUNTIME_DIR}/submitted"\necho 27159\n')
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", RUNTIME_DIR=str(tmp_path), DYNAMO="false", GITHUB_ENV=str(tmp_path / "env"), GITHUB_OUTPUT=str(tmp_path / "output"))
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", workflow_step("submit")], env=env, capture_output=True, check=False, text=True)
    success = report is not None and report.get("image_sha256") == "image"
    assert (result.returncode == 0) == success, result.stderr
    assert (tmp_path / "submitted").exists() == success
