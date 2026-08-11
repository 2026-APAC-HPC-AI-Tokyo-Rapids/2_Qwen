from __future__ import annotations

import json
import signal
import socket
import struct
import subprocess
import sys
import time
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
    assert 'control/instrumentation-source-blob-oids.json' in node
    assert "for _ in {1..60}" in node
    assert '[[ ! -s "${completion_file}" ]] || break' in node


def test_overhead_workflow_archives_fixed_evidence_and_has_no_bq_boundary() -> None:
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
    ):
        assert member in workflow[upload:discard]
    assert "workload-warmup.json" not in workflow[upload:discard]
    lowered = workflow.lower()
    assert "bigquery" not in lowered
    assert "google_credentials" not in lowered
    assert "qwen-experiment-metrics publish" not in lowered


def test_event_drain_validates_envelopes_and_writes_closed_counts(
    tmp_path: Path,
) -> None:
    script = ROOT / "scripts/v2-overhead-event-drain.py"
    socket_path = tmp_path / "events.sock"
    ready = tmp_path / "ready"
    summary = tmp_path / "summary.json"
    process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            str(socket_path),
            str(ready),
            str(summary),
            RUN_ID,
            "0",
        ]
    )
    try:
        deadline = time.monotonic() + 3
        while not ready.exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        for sequence, event_type in enumerate(
            ("bench_args_effective", "primary_result")
        ):
            value = {
                "schema_version": "qwen-domain-event/v1",
                "run_id": RUN_ID,
                "producer_id": "sglang-node-0",
                "node_rank": 0,
                "sequence": sequence,
                "event_type": event_type,
                "payload": {},
            }
            payload = jcs_bytes(value)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(socket_path))
                client.sendall(struct.pack(">I", len(payload)) + payload)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=3) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    value = json.loads(summary.read_text(encoding="utf-8"))
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
