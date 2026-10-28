"""Exercise the launcher after readiness; no scheduler or GPU execution."""

import json
import os
import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("only,normal_status,profile_status", [(False, 0, 0), (False, 1, 0), (True, 0, 0), (True, 0, 2)])
def test_profile_survives_sla_failure_without_repeating_official_in_recovery(tmp_path, only, normal_status, profile_status):
    for directory in ("scripts", "source", "logs", "profile", "results"):
        (tmp_path / directory).mkdir()
    (tmp_path / "source/worker-endpoints.tsv").write_text("")
    (tmp_path / "scripts/capture-firmus-profile.sh").write_text("#!/bin/bash\ntouch \"$1/captured\"\n")
    (tmp_path / "scripts/capture-firmus-profile.sh").chmod(0o755)
    # Acceptance aggregation has separate tests and is not invoked by recovery.
    (tmp_path / "scripts/collect-eagle-acceptance.py").write_text("pass\n")
    source = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    tail = "benchmark_status=0\n" + source.split("benchmark_status=0\n", 1)[1]
    prelude = '''
set -euo pipefail
mounts() { :; }
srun() {
  printf '%s\\n' "$*" >>"${ROOT}/calls"
  if [[ "$*" == *official-aiperf.sh* ]]; then
    if [[ "${*: -1}" == profile ]]; then return "$PROFILE_STATUS"; fi
    printf '{}\\n' >"${ROOT}/results/score.json"
    return "$NORMAL_STATUS"
  fi
  return 0
}
'''
    result = subprocess.run(["bash", "-c", prelude + tail], env=os.environ | {
        "ROOT": str(tmp_path), "ART": str(tmp_path / "logs"), "HEAD": "stub",
        "AIPERF": "/image", "N": "215", "PROFILE": "true",
        "profile_only": str(only).lower(), "routing_capture_samples": "0",
        "NORMAL_STATUS": str(normal_status), "PROFILE_STATUS": str(profile_status),
    }, capture_output=True, text=True, check=False)
    assert result.returncode == (profile_status or (0 if only else normal_status)), result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    clients = [row for row in calls if "official-aiperf.sh" in row]
    assert len(clients) == (1 if only else 2)
    assert clients[-1].endswith(" profile")
    assert (tmp_path / "captured").exists()
    assert (tmp_path / "results/score.json").exists() is (not only)
    assert (tmp_path / "run-success").exists() is (result.returncode == 0)


@pytest.mark.parametrize("gpus,score,expected", [(16, False, 0), (12, False, 1), (16, True, 1)])
def test_profile_only_completion_requires_gpu_coverage_and_absent_score(tmp_path, gpus, score, expected):
    for directory in ("source", "profile/client", "results", "logs", "bin"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "source/firmus-framework.json").write_text(json.dumps({"profile_only": True}))
    (tmp_path / "profile/client/profile_export_aiperf.json").write_text("{}")
    (tmp_path / "profile/validation.json").write_text(json.dumps({"valid": True, "workers": {"all": {"gpu_count": gpus}}}))
    (tmp_path / "run-success").touch()
    (tmp_path / "logs/slurm-42.log").touch()
    if score:
        (tmp_path / "results/score.json").write_text("{}")
    for name, body in {"squeue": ":", "sacct": "echo '42|COMPLETED|0:0|00:08:00|2'"}.items():
        stub = tmp_path / "bin" / name
        stub.write_text("#!/bin/bash\n" + body + "\n")
        stub.chmod(0o755)
    result = subprocess.run(["bash", str(ROOT / "platforms/firmus/check-qwen-job.sh")], env=os.environ | {
        "RUNTIME_DIR": str(tmp_path), "SLURM_JOB_ID": "42", "PATH": f"{tmp_path}/bin:{os.environ['PATH']}",
    }, capture_output=True, text=True, check=False)
    assert result.returncode == expected, result.stderr


def test_summary_identifies_profile_only_without_an_official_score(tmp_path):
    (tmp_path / "provenance.json").write_text(json.dumps({"profile": True, "profile_only": True, "workload": "official"}))
    summary = runpy.run_path(str(ROOT / "scripts/summarize-firmus-run.py"))["summarize"](tmp_path, "42", "success")
    assert "ordinary benchmark omitted; no official score was produced" in summary
    assert "20 / 30 / 90 (profile only)" in summary
    assert "Validated AIPerf result unavailable" not in summary
