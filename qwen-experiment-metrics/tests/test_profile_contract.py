from __future__ import annotations

import gzip
import hashlib
from decimal import Decimal
from pathlib import Path

import pytest

from qwen_experiment_metrics.canonical import jcs_bytes
from qwen_experiment_metrics.profile_contract import (
    DCGM_OPTIONAL_FIELDS,
    NVIDIA_SMI_SAMPLE_ARGV,
    PROFILE_ACTIVITIES,
    PROFILE_ID,
    PROFILE_STEPS,
    ProfilerEventRow,
    ProfileValidationError,
    dcgm_supported_fields,
    dedupe_profiler_rows,
    parse_phase_summary,
    profiler_environment,
    profiler_overlay_args,
    project_profiler_events,
    validate_profiler_overlay,
)

RUN_ID = "r-0123456789abcdef0123456789abcdef"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jcs_bytes(value) + b"\n")


def _phase(kind: str = "measured") -> dict[str, object]:
    return {
        "schema_version": "phase-summary/v1",
        "run_id": RUN_ID,
        "invocation_id": "inv-1",
        "kind": kind,
        "rank0": {
            "clock_id": "boot:mono",
            "start_broadcast_ns": 10,
            "all_start_acks_received_ns": 20,
            "generation_start_ns": 30,
            "generation_end_ns": 80,
            "end_broadcast_ns": 90,
            "all_end_acks_received_ns": 100,
        },
        "nodes": [
            {
                "node_rank": 0,
                "clock_id": "boot:mono",
                "start_sequence": 1,
                "end_sequence": 2,
                "start_received_ns": 21,
                "start_ack_ns": 22,
                "end_received_ns": 91,
                "end_ack_ns": 92,
            }
        ],
    }


def _trace_bytes(*events: dict[str, object]) -> bytes:
    return gzip.compress(jcs_bytes({"traceEvents": list(events)}))


def _write_trace(runtime: Path, rank: int, events: tuple[dict[str, object], ...]) -> None:
    torch = runtime / "profile/raw/torch"
    torch.mkdir(parents=True, exist_ok=True)
    trace = torch / f"qwen-profile-TP-{rank}.trace.json.gz"
    trace.write_bytes(_trace_bytes(*events))
    _write_json(
        torch / f"qwen-profile-TP-{rank}.capture.json",
        {
            "schema_version": "profiler-capture/v1",
            "run_id": RUN_ID,
            "global_rank": rank,
            "invocation_id": "inv-1",
            "source_member": f"profile/raw/torch/{trace.name}",
            "size_bytes": trace.stat().st_size,
            "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        },
    )


def test_phase_summary_is_closed_and_orders_windows(tmp_path: Path) -> None:
    path = tmp_path / "control/phase/summary.json"
    _write_json(path, _phase())
    summary = parse_phase_summary(path, run_id=RUN_ID, expected_nodes=1)
    assert summary.kind == "measured"
    assert summary.nodes[0].start_received_ns == 21
    assert summary.as_dict()["schema_version"] == "phase-summary-projection/v1"

    invalid = _phase()
    invalid["rank0"]["generation_end_ns"] = 15  # type: ignore[index]
    _write_json(path, invalid)
    with pytest.raises(ProfileValidationError):
        parse_phase_summary(path, run_id=RUN_ID, expected_nodes=1)


def test_phase_summary_keeps_each_remote_node_clock_domain(tmp_path: Path) -> None:
    value = _phase()
    value["nodes"].append(  # type: ignore[union-attr]
        {
            "node_rank": 1,
            "clock_id": "remote-boot:mono",
            "start_sequence": 1,
            "end_sequence": 2,
            "start_received_ns": 15,
            "start_ack_ns": 16,
            "end_received_ns": 70,
            "end_ack_ns": 71,
        }
    )
    path = tmp_path / "control/phase/summary.json"
    _write_json(path, value)
    summary = parse_phase_summary(path, run_id=RUN_ID, expected_nodes=2)
    assert summary.nodes[1].clock_id == "remote-boot:mono"


def test_profiler_overlay_and_dcgm_optional_contract() -> None:
    assert PROFILE_ID == "qwen-profile"
    assert PROFILE_ACTIVITIES == ("CPU", "GPU")
    assert PROFILE_STEPS == 5
    assert NVIDIA_SMI_SAMPLE_ARGV == (
        "nvidia-smi",
        "--query-gpu=uuid,index,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm",
        "--format=csv,noheader,nounits",
    )
    validate_profiler_overlay(profiler_overlay_args(), profiler_environment())
    with pytest.raises(ProfileValidationError):
        validate_profiler_overlay(("--profile",), profiler_environment())
    assert dcgm_supported_fields(("engine_active_ratio",)) == (
        "engine_active_ratio",
    )
    assert dcgm_supported_fields(()) == ()
    with pytest.raises(ProfileValidationError):
        dcgm_supported_fields(("unknown",))
    assert "xid_error_count" in DCGM_OPTIONAL_FIELDS


def test_profiler_projection_preserves_capture_lineage_and_partial_rank(
    tmp_path: Path,
) -> None:
    _write_trace(
        tmp_path,
        0,
        (
            {"ph": "X", "cat": "cpu_op", "name": "prepare", "dur": 2},
            {"ph": "X", "cat": "cuda_kernel", "name": "matmul", "dur": 3.5},
            {"ph": "M", "name": "thread_name"},
        ),
    )
    _write_trace(tmp_path, 1, ())
    projection = project_profiler_events(
        tmp_path,
        run_id=RUN_ID,
        invocation_id="inv-1",
        expected_ranks=3,
    )
    assert projection.missing_ranks == (2,)
    assert projection.partial_ranks == (1,)
    assert not projection.collection_complete
    assert {(row.global_rank, row.activity) for row in projection.profiler_rows} == {
        (0, "CPU"),
        (0, "GPU"),
    }
    gpu = next(row for row in projection.profiler_rows if row.activity == "GPU")
    assert gpu.inclusive_duration_sum_ns == Decimal(3500)
    assert gpu.as_dict()["schema_version"] == "profiler-event-projection/v1"
    assert gpu.source_member.endswith("qwen-profile-TP-0.trace.json.gz")
    assert len(projection.captures) == 2


def test_profiler_projection_dedupes_equal_rows_and_keeps_conflict() -> None:
    row = ProfilerEventRow(
        RUN_ID,
        0,
        "GPU",
        "cuda_kernel",
        "matmul",
        1,
        Decimal(2),
        "profile/raw/torch/qwen-profile-TP-0.trace.json.gz",
        "a" * 64,
    )
    duplicate = ProfilerEventRow(
        RUN_ID,
        0,
        "GPU",
        "cuda_kernel",
        "matmul",
        1,
        Decimal(2),
        row.source_member,
        row.source_sha256,
    )
    changed = ProfilerEventRow(
        RUN_ID,
        0,
        "GPU",
        "cuda_kernel",
        "matmul",
        2,
        Decimal(2),
        row.source_member,
        row.source_sha256,
    )
    rows, conflicts = dedupe_profiler_rows((row, duplicate))
    assert rows == (row,)
    assert conflicts == ()
    rows, conflicts = dedupe_profiler_rows((row, changed))
    assert rows == ()
    assert len(conflicts) == 1
