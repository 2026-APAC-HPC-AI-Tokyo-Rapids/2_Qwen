"""Submission entrypoint and terminal scoring checks, without GPU allocation."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def test_actions_stages_the_submission_launcher():
    launcher = ROOT / "submission/run/vllm.sbatch"
    assert (ROOT / "platforms/firmus/framework-comparison/vllm.sbatch").resolve() == launcher
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    assert 'launcher="${GITHUB_WORKSPACE}/submission/run/vllm.sbatch"' in provision
    assert 'install -m 0755 "${launcher}" "${runtime_dir}/scripts/submit.sbatch"' in provision


@pytest.mark.parametrize("fault", [None, "score", "ineligible", "cancelled", "raw_missing"])
def test_terminal_replays_the_pinned_official_client_export(tmp_path, fault):
    for directory in ("bin", "source", "scripts", "logs", "results"):
        (tmp_path / directory).mkdir()
    source = ROOT / "submission/results/logs/17021"
    for name in ("profile_export_aiperf.json", "score.json", "aiperf-command.txt", "aiperf-provenance.txt"):
        shutil.copyfile(source / name, tmp_path / "results" / name)
    shutil.copyfile(ROOT / "scripts/parse-aiperf-result.py", tmp_path / "scripts/parse-aiperf-result.py")
    (tmp_path / "run-success").touch()
    (tmp_path / "logs/slurm-7.log").write_text("fixture terminal allocation\n")
    (tmp_path / "source/provenance.json").write_text(json.dumps({"workload": "official", "profile": False, "framework": "vllm"}))
    (tmp_path / "source/firmus-framework.json").write_text('{"concurrency":16}')
    if fault != "raw_missing":
        # Only presence is checked here; this is not request-level SLA evidence.
        (tmp_path / "results/profile_export.jsonl").write_text('{"fixture":true}\n')
    if fault == "score":
        path = tmp_path / "results/score.json"
        score = json.loads(path.read_text())
        score["score_output_tokens_per_second"] *= 2
        path.write_text(json.dumps(score))
    elif fault in ("ineligible", "cancelled"):
        path = tmp_path / "results/profile_export_aiperf.json"
        export = json.loads(path.read_text())
        if fault == "cancelled":
            export["was_cancelled"] = True
        else:
            export["good_request_count"]["avg"] = 400
            export["good_request_fraction"]["avg"] = 400 / 558
        path.write_text(json.dumps(export))
    for name, body in {"squeue": "exit 0", "sacct": "echo '7|COMPLETED|0:0|00:12:00|2'"}.items():
        executable = tmp_path / "bin" / name
        executable.write_text(f"#!/bin/sh\n{body}\n")
        executable.chmod(0o755)
    result = subprocess.run(["bash", str(ROOT / "platforms/firmus/check-qwen-job.sh")], env={**os.environ, "PATH": f"{tmp_path}/bin:{os.environ['PATH']}", "SLURM_JOB_ID": "7", "RUNTIME_DIR": str(tmp_path)}, capture_output=True, text=True, check=False)
    assert (result.returncode == 0) is (fault is None), result.stdout + result.stderr


@pytest.mark.parametrize("client_status", [0, 23])
def test_post_measurement_snapshot_preserves_client_status(tmp_path, client_status):
    launcher = (ROOT / "submission/run/vllm.sbatch").read_text()
    start = launcher.index("while IFS=", launcher.index("aiperf_status=0"))
    end = launcher.index("# The scheduler log", start)
    (tmp_path / "source").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "source/worker-endpoints.tsv").write_text("decode-0\t127.0.0.1\t30200\t1\n")
    script = 'set -euo pipefail\ncurl() { return 7; }\n' + launcher[start:end] + '\nexit "${aiperf_status}"\n'
    result = subprocess.run(["bash", "-c", script], env={**os.environ, "ROOT": str(tmp_path), "aiperf_status": str(client_status)}, capture_output=True, text=True, check=False)
    assert result.returncode == client_status
    assert "metrics_snapshot_failed worker=decode-0" in result.stderr
