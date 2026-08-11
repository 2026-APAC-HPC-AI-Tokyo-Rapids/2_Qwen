from __future__ import annotations

import json
from pathlib import Path

from qwen_experiment_metrics.domain import Fact
from qwen_experiment_metrics.recording import Recorder
from qwen_experiment_metrics.system_sources import capture_environment_snapshot


def test_atomic_fragment_rows_are_sent_unchanged(
    manifest_path: Path, runtime: Path, monkeypatch: object
) -> None:
    sent = []
    monkeypatch.setenv("EXPERIMENT_METRICS_DIRECT_SEND", "1")
    monkeypatch.setattr(
        "qwen_experiment_metrics.publish.send_raw",
        lambda run, rows: sent.append((run, list(rows))) or True,
    )
    with Recorder.open(manifest_path, None, "arbitrary-producer", runtime) as recorder:
        recorder.record(Fact("arbitrary.key", {"b": 2, "a": 1}))
    persisted = [
        json.loads(line) for line in recorder.completed_path.read_text().splitlines()
    ]
    assert sent == [(recorder.manifest.run_id, persisted)]


def test_credential_transport_variables_are_not_snapshotted(tmp_path: Path) -> None:
    output = tmp_path / "environment.snapshot"
    capture_environment_snapshot(
        output,
        {
            b"SAFE": b"yes",
            b"EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_FILE": b"/secret/path",
            b"EXPERIMENT_METRICS_BIGQUERY_CONFIG": b"/config/path",
            b"EXPERIMENT_METRICS_DIRECT_SEND": b"1",
        },
    )
    assert output.read_bytes() == b"SAFE\0yes\0"
