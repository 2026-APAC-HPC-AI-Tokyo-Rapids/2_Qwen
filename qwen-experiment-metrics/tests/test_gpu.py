from __future__ import annotations

from decimal import Decimal

import pytest

from qwen_experiment_metrics.gpu import (
    GpuSampleWriter,
    parse_gpu_samples_jsonl,
    parse_nvidia_smi_csv,
    reduce_measured_generation,
)

RUN_ID = "r-" + "b" * 32
EXPECTED = {"GPU-a": 0, "GPU-b": 1}


def _measurement(index: int, utilization: object, power: object) -> dict[str, object]:
    return {
        "logical_index": index,
        "utilization_percent": utilization,
        "memory_used_bytes": 1_000 + index,
        "power_w": power,
        "temperature_c": 40 + index,
        "sm_clock_mhz": 1_500 - index,
    }


def _write_samples(tmp_path):
    path = tmp_path / "profile/raw/gpu/node-0.jsonl"
    with GpuSampleWriter(
        path,
        run_id=RUN_ID,
        purpose="profile",
        node_rank=0,
        clock_id="boot:monotonic",
        expected_gpus=EXPECTED,
        flush_every_frames=2,
    ) as writer:
        for sequence, timestamp, utilization, power in (
            (0, 0, 10, 100),
            (1, 1_000_000_000, 20, None),
            (3, 3_000_000_000, 100, 100),
        ):
            writer.append_frame(
                sequence=sequence,
                monotonic_ns=timestamp,
                wall_time_utc=f"2026-08-10T00:00:0{sequence}Z",
                measurements={
                    "GPU-a": _measurement(0, utilization, power),
                    "GPU-b": _measurement(1, utilization / 2, power),
                },
            )
    return path


def test_gpu_writer_is_profile_only_and_round_trips_complete_frames(tmp_path) -> None:
    with pytest.raises(ValueError, match="profile-only"):
        GpuSampleWriter(
            tmp_path / "forbidden.jsonl",
            run_id=RUN_ID,
            purpose="bench",
            node_rank=0,
            clock_id="clock",
            expected_gpus=EXPECTED,
        )

    path = _write_samples(tmp_path)
    samples = parse_gpu_samples_jsonl(
        path.read_bytes(),
        run_id=RUN_ID,
        node_rank=0,
        clock_id="boot:monotonic",
        expected_gpus=EXPECTED,
    )

    assert len(samples) == 6
    assert {sample.sequence for sample in samples} == {0, 1, 3}
    assert samples[0].utilization_percent == Decimal(10)


def test_gpu_reader_ignores_incomplete_frame_and_unterminated_tail(tmp_path) -> None:
    data = _write_samples(tmp_path).read_bytes()
    lines = data.splitlines(keepends=True)
    # Remove GPU-b from sequence 1, then leave a kill-truncated tail.
    damaged = b"".join(lines[:2] + lines[2:3] + lines[4:]) + b'{"schema_version"'
    samples = parse_gpu_samples_jsonl(
        damaged,
        run_id=RUN_ID,
        node_rank=0,
        clock_id="boot:monotonic",
        expected_gpus=EXPECTED,
    )

    assert {sample.sequence for sample in samples} == {0, 3}


def test_measured_generation_reducer_uses_bounded_step_intervals(tmp_path) -> None:
    path = _write_samples(tmp_path)
    samples = parse_gpu_samples_jsonl(
        path.read_bytes(),
        run_id=RUN_ID,
        node_rank=0,
        clock_id="boot:monotonic",
        expected_gpus=EXPECTED,
    )
    rows = reduce_measured_generation(
        samples,
        node_rank=0,
        clock_id="boot:monotonic",
        start_ns=500_000_000,
        end_ns=4_000_000_000,
        expected_gpus=EXPECTED,
        nominal_interval_sec=Decimal(1),
    )

    assert len(rows) == 2
    first = rows[0]
    assert first.phase == "measured_generation"
    assert first.phase_duration_sec == Decimal("3.5")
    assert first.valid_sample_count == 3
    assert first.valid_utilization_duration_sec == Decimal("3.5")
    assert first.utilization_coverage == Decimal(1)
    assert first.utilization_mean == Decimal(145) / Decimal("3.5")
    assert first.utilization_p95 == Decimal(100)
    assert first.valid_power_duration_sec == Decimal("1.5")
    assert first.energy_j == Decimal(150)
    assert first.power_mean == Decimal(100)
    assert first.valid_memory_sample_count == 3
    assert first.memory_max_bytes == 1_000
    assert first.temperature_max_c == Decimal(40)
    assert first.sm_clock_min_mhz == Decimal(1500)


def test_reducer_requires_both_phase_markers_and_all_expected_gpus(tmp_path) -> None:
    samples = parse_gpu_samples_jsonl(
        _write_samples(tmp_path).read_bytes(),
        run_id=RUN_ID,
        node_rank=0,
        clock_id="boot:monotonic",
        expected_gpus=EXPECTED,
    )
    assert (
        reduce_measured_generation(
            samples,
            node_rank=0,
            clock_id="boot:monotonic",
            start_ns=None,
            end_ns=4_000_000_000,
            expected_gpus=EXPECTED,
        )
        == ()
    )
    assert (
        reduce_measured_generation(
            tuple(sample for sample in samples if sample.gpu_uuid == "GPU-a"),
            node_rank=0,
            clock_id="boot:monotonic",
            start_ns=0,
            end_ns=4_000_000_000,
            expected_gpus=EXPECTED,
        )
        == ()
    )


def test_nvidia_smi_csv_maps_units_and_nullable_fields() -> None:
    parsed = parse_nvidia_smi_csv(
        "GPU-a, 0, 99.5, 2.5, N/A, 42, 1500\n"
        "GPU-b, 1, 50, 4, 300.25, [N/A], Not Supported\n"
    )

    assert parsed["GPU-a"]["memory_used_bytes"] == 2_621_440
    assert parsed["GPU-a"]["power_w"] is None
    assert parsed["GPU-b"]["power_w"] == Decimal("300.25")
    assert parsed["GPU-b"]["sm_clock_mhz"] is None
