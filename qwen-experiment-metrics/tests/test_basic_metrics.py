from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PUBLISHER = runpy.run_path(str(ROOT / "scripts/publish-basic-metrics.py"))


def fixture_archive(root, state="COMPLETED", tier="basic", run="41"):
    archive = root / "41/2"
    files = {
        "results/results/score.json": {"concurrency": 128, "eligible": True, "good_output_tokens_per_second": 5649.5},
        "source/source/firmus-framework.json": {"tier": tier, "workload": "official", "concurrency": 128, "images": {"sglang": {"path": "/private/runtime/image"}}},
        # A failed submit can reuse attempt-1 preparation for attempt-2 execution.
        "source/source/provenance.json": {"run_id": run, "run_attempt": "1", "profile": False, "commit": "a" * 40},
        "results/results/profile_export_aiperf.json": {"aiperf_version": "0.12.0", **{key: {"avg": 12.5, "unit": "fixture"} for key in ("time_to_first_token", "inter_token_latency", "output_token_throughput", "input_token_throughput", "request_latency", "benchmark_duration")}},
    }
    for name, value in files.items():
        path = archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    accounting = archive / "logs/logs/slurm-accounting.txt"
    accounting.parent.mkdir(parents=True)
    accounting.write_text(f"27652|{state}|0:0|00:09:58|2\n")
    (archive / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{path.relative_to(archive)}\n" for path in sorted(archive.rglob("*")) if path.is_file()))
    return archive


def test_terminal_archive_uses_execution_identity_and_keeps_raw_immutable(tmp_path):
    archive = fixture_archive(tmp_path)
    checksum = (archive / "SHA256SUMS").read_bytes()
    sent = []

    def sender(run, rows):
        sent.append((run, rows))
        return True

    assert PUBLISHER["publish"](tmp_path, "41/2", "owner/repository", sender)
    assert len(sent[0][1]) == 9
    assert sent[0][0] == "github:owner/repository:41/2"
    values = {row.key: json.loads(row.value) for row in sent[0][1]}
    assert values["aiperf.provenance"]["measurement"] == "41/2"
    assert values["aiperf.provenance"]["prepared"]["run_attempt"] == "1"
    assert "images" not in values["aiperf.configuration"]
    assert values["aiperf.score"]["good_output_tokens_per_second"] == 5649.5
    fragment = archive.parent / "metrics-2.jsonl"
    assert len(fragment.read_text().splitlines()) == 9
    assert (archive / "SHA256SUMS").read_bytes() == checksum
    # Explicit re-publication permits duplicate inserts, as the sender specifies.
    assert PUBLISHER["publish"](tmp_path, "41/2", "owner/repository", sender)
    assert len(sent) == 2


@pytest.mark.parametrize("kwargs", [{"state": "RUNNING"}, {"tier": "bonus"}, {"run": "42"}])
def test_no_publication_before_terminal_or_from_wrong_measurement(tmp_path, kwargs):
    fixture_archive(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        PUBLISHER["collect"](tmp_path, "41/2", "owner/repository")
    assert not (tmp_path / "41/metrics-2.jsonl").exists()


def test_corrupt_archive_is_rejected_before_publication(tmp_path):
    archive = fixture_archive(tmp_path)
    (archive / "results/results/score.json").write_text("{}")
    with pytest.raises(subprocess.CalledProcessError):
        PUBLISHER["collect"](tmp_path, "41/2", "owner/repository")


def test_transport_failure_retains_diagnostic_without_changing_raw(tmp_path):
    archive = fixture_archive(tmp_path)
    checksum = (archive / "SHA256SUMS").read_bytes()
    assert not PUBLISHER["publish"](tmp_path, "41/2", "owner/repository", lambda *_: False)
    assert (archive.parent / "metrics-2.jsonl").exists()
    assert (archive / "SHA256SUMS").read_bytes() == checksum


def test_metrics_workflow_has_no_scheduler_or_compute_credential_route():
    workflow = (ROOT / ".github/workflows/qwen-basic-metrics.yml").read_text()
    assert "home-lab-shell" in workflow and "continue-on-error: true" in workflow
    assert not any(command in workflow for command in ("firmus-shell", "sbatch", "srun"))
    assert "GOOGLE_CREDENTIAL" not in (ROOT / "submission/basic/submit.sbatch").read_text()
