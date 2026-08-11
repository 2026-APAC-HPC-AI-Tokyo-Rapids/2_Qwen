from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conftest import RUN_ID

from qwen_experiment_metrics.canonical import jcs_bytes

ROOT = Path(__file__).resolve().parents[2]


def test_overhead_harness_is_alternating_and_gates_primary_time() -> None:
    pjm = (ROOT / "scripts/v2-overhead-check.pjm").read_text(encoding="utf-8")
    node = (ROOT / "scripts/v2-overhead-node.sh").read_text(encoding="utf-8")

    assert "#PJM -L jobenv=singularity" in pjm
    assert "arms=(off on off on)" in pjm
    assert 'primary_measure:"generation_wall_sec"' in pjm
    assert "on_generation / off_generation" in pjm
    assert "<= 1.05" in pjm
    assert "workload-warmup.json" in pjm
    assert "workload-measured.json" in pjm
    assert "reference_workload_identity" in pjm
    assert "QWEN_EXPERIMENT_METRICS_BENCH_ARGS_BLOB_OID" in node
    assert "QWEN_EXPERIMENT_METRICS_SERVER_ARGS_BLOB_OID" in node
    assert "QWEN_EXPERIMENT_METRICS_SOURCE_BLOB_OID" not in node
    assert 'singularity exec "${CHECK_DIR}/sglang.sif"' not in node
    assert "control/instrumentation-source-blob-oids.json" in node
    assert "for _ in {1..60}" in node
    assert '[[ ! -s "${completion_file}" ]] || break' in node
    assert "qwen-experiment-metrics/.venv/bin/qwen-experiment-metrics" in node
    assert "serve-events" in node
    assert "producer-finalize-wall-sec-node-" in node


def test_overhead_workflow_archives_direct_send_evidence_without_credentials() -> None:
    workflow = (ROOT / ".github/workflows/v2-overhead-check.yml").read_text(
        encoding="utf-8"
    )

    upload = workflow.index("Upload fixed comparison evidence")
    discard = workflow.index("Discard comparison data")
    assert upload < discard
    assert "if: always()" in workflow[upload:discard]
    assert "offline-throughput-source-blob-oid.txt" in workflow[:upload]
    assert "server-args-source-blob-oid.txt" in workflow[:upload]
    for member in (
        "comparison-input.json",
        "instrumentation-source-blob-oids.json",
        "comparison-order.txt",
        "observations.tsv",
        "summary.json",
        "primary-result.json",
        "event-summary-node-*.json",
        "domain/fragments/producer--*.jsonl",
        "producer-finalize-wall-sec-node-*",
    ):
        assert member in workflow[upload:discard]
    assert "workload-warmup.json" not in workflow[upload:discard]
    lowered = workflow.lower()
    assert "bigquery-credential.json" not in workflow[upload:discard]
    assert "EXPERIMENT_METRICS_DIRECT_SEND=1" in workflow[:upload]
    assert "EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_FILE" in workflow[:upload]
    assert "qwen-experiment-metrics publish" not in lowered


def test_event_summary_reports_recorder_archive_counts(
    tmp_path: Path,
) -> None:
    script = ROOT / "scripts/v2-overhead-event-summary.py"
    archive = tmp_path / "events.jsonl"
    summary = tmp_path / "summary.json"
    values = []
    for sequence, event_type in enumerate(("bench_args_effective", "primary_result")):
        values.append(
            {
                "schema_version": "qwen-domain-event/v1",
                "run_id": RUN_ID,
                "producer_id": "sglang-node-0",
                "node_rank": 0,
                "sequence": sequence,
                "event_type": event_type,
                "payload": {},
            }
        )
    archive.write_bytes(b"".join(jcs_bytes(value) + b"\n" for value in values))
    subprocess.run(
        [sys.executable, str(script), str(archive), str(summary), RUN_ID, "0"],
        check=True,
    )
    value = json.loads(summary.read_text(encoding="utf-8"))
    assert value["schema_version"] == "overhead-event-summary/v2"
    assert set(value) == {
        "schema_version",
        "run_id",
        "node_rank",
        "total_events",
        "total_payload_bytes",
        "event_type_counts",
        "producer_ids",
    }
    assert value["total_events"] == 2
    assert value["event_type_counts"] == {
        "bench_args_effective": 1,
        "primary_result": 1,
    }
    assert value["producer_ids"] == ["sglang-node-0"]
