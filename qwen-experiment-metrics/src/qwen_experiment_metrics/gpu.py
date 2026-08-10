from __future__ import annotations

import csv
import os
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path
from typing import Self

from .canonical import CanonicalizationError, jcs_bytes, load_json_strict

_SAMPLE_FIELDS = {
    "schema_version",
    "run_id",
    "node_rank",
    "sequence",
    "clock_id",
    "monotonic_ns",
    "wall_time_utc",
    "gpu_uuid",
    "logical_index",
    "utilization_percent",
    "memory_used_bytes",
    "power_w",
    "temperature_c",
    "sm_clock_mhz",
}
_NVIDIA_FIELDS = (
    "uuid",
    "index",
    "utilization.gpu",
    "memory.used",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
)
_MISSING = frozenset({"", "N/A", "[N/A]", "Not Supported", "[Not Supported]"})
_NANOSECONDS = Decimal(1_000_000_000)


@dataclass(frozen=True, slots=True)
class GpuSample:
    schema_version: str
    run_id: str
    node_rank: int
    sequence: int
    clock_id: str
    monotonic_ns: int
    wall_time_utc: str
    gpu_uuid: str
    logical_index: int
    utilization_percent: Decimal | None
    memory_used_bytes: int | None
    power_w: Decimal | None
    temperature_c: Decimal | None
    sm_clock_mhz: Decimal | None

    def to_object(self) -> dict[str, object]:
        return _json_numbers(asdict(self))


@dataclass(frozen=True, slots=True)
class GpuPhaseV1:
    phase: str
    node_rank: int
    gpu_uuid: str
    phase_duration_sec: Decimal
    valid_sample_count: int
    valid_utilization_duration_sec: Decimal
    utilization_coverage: Decimal
    utilization_mean: Decimal | None
    utilization_p95: Decimal | None
    valid_power_duration_sec: Decimal
    power_coverage: Decimal
    energy_j: Decimal | None
    power_mean: Decimal | None
    valid_memory_sample_count: int
    memory_max_bytes: int | None
    valid_temperature_sample_count: int
    temperature_max_c: Decimal | None
    valid_sm_clock_sample_count: int
    sm_clock_min_mhz: Decimal | None

    def to_object(self) -> dict[str, object]:
        return asdict(self)


class GpuSampleWriter:
    """Buffered, node-owned raw JSONL writer available only to profile runs."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        purpose: str,
        node_rank: int,
        clock_id: str,
        expected_gpus: Mapping[str, int],
        flush_every_frames: int = 10,
    ) -> None:
        if purpose != "profile":
            raise ValueError("GPU sampling is profile-only")
        if not run_id or not clock_id or node_rank < 0 or not expected_gpus:
            raise ValueError("invalid GPU writer identity")
        if set(expected_gpus.values()) != set(range(len(expected_gpus))):
            raise ValueError("logical GPU indices must be contiguous and unique")
        if flush_every_frames <= 0:
            raise ValueError("flush interval must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("xb")
        self.run_id = run_id
        self.node_rank = node_rank
        self.clock_id = clock_id
        self.expected_gpus = dict(expected_gpus)
        self.flush_every_frames = flush_every_frames
        self._frames_since_flush = 0
        self._closed = False

    def append_frame(
        self,
        *,
        sequence: int,
        monotonic_ns: int,
        wall_time_utc: str,
        measurements: Mapping[str, Mapping[str, object | None]],
    ) -> None:
        if self._closed:
            raise ValueError("GPU sample writer is closed")
        if set(measurements) != set(self.expected_gpus):
            raise ValueError("poll frame does not contain the expected GPU UUIDs")
        samples = tuple(
            _sample_from_measurement(
                run_id=self.run_id,
                node_rank=self.node_rank,
                sequence=sequence,
                clock_id=self.clock_id,
                monotonic_ns=monotonic_ns,
                wall_time_utc=wall_time_utc,
                gpu_uuid=uuid,
                logical_index=self.expected_gpus[uuid],
                values=measurements[uuid],
            )
            for uuid in sorted(measurements, key=self.expected_gpus.__getitem__)
        )
        for sample in samples:
            self._stream.write(jcs_bytes(sample.to_object()) + b"\n")
        self._frames_since_flush += 1
        if self._frames_since_flush >= self.flush_every_frames:
            self.flush()

    def flush(self) -> None:
        if not self._closed:
            self._stream.flush()
            self._frames_since_flush = 0

    def close(self) -> None:
        if not self._closed:
            self.flush()
            self._stream.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class NvidiaSmiSampler(threading.Thread):
    """Poll nvidia-smi locally and append only complete expected-GPU frames."""

    def __init__(
        self,
        writer: GpuSampleWriter,
        *,
        interval_sec: float = 1.0,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if interval_sec <= 0:
            raise ValueError("sampling interval must be positive")
        super().__init__(daemon=True)
        self.writer = writer
        self.interval_sec = interval_sec
        self.runner = runner
        self.stop_requested = threading.Event()

    def run(self) -> None:
        deadline = time.monotonic()
        sequence = 0
        while not self.stop_requested.is_set():
            self.poll_once(sequence)
            sequence += 1
            deadline += self.interval_sec
            self.stop_requested.wait(max(0.0, deadline - time.monotonic()))

    def poll_once(self, sequence: int) -> bool:
        try:
            completed = self.runner(
                [
                    "nvidia-smi",
                    f"--query-gpu={','.join(_NVIDIA_FIELDS)}",
                    "--format=csv,noheader,nounits",
                ],
                env={**os.environ, "LC_ALL": "C"},
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=max(1.0, self.interval_sec),
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if completed.returncode != 0:
            return False
        measurements = parse_nvidia_smi_csv(completed.stdout)
        if set(measurements) != set(self.writer.expected_gpus):
            return False
        now_ns = time.monotonic_ns()
        wall = (
            datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
        )
        self.writer.append_frame(
            sequence=sequence,
            monotonic_ns=now_ns,
            wall_time_utc=wall,
            measurements=measurements,
        )
        return True

    def stop(self) -> None:
        self.stop_requested.set()


def parse_nvidia_smi_csv(text: str) -> dict[str, dict[str, object | None]]:
    result: dict[str, dict[str, object | None]] = {}
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if len(row) != len(_NVIDIA_FIELDS):
            return {}
        values = [item.strip() for item in row]
        uuid = values[0]
        try:
            logical_index = int(values[1])
            if not uuid or uuid in result or logical_index < 0:
                return {}
            result[uuid] = {
                "logical_index": logical_index,
                "utilization_percent": _nullable_decimal(values[2]),
                "memory_used_bytes": _nullable_mib(values[3]),
                "power_w": _nullable_decimal(values[4]),
                "temperature_c": _nullable_decimal(values[5]),
                "sm_clock_mhz": _nullable_decimal(values[6]),
            }
        except (ValueError, InvalidOperation):
            return {}
    return result


def parse_gpu_samples_jsonl(
    data: bytes | str,
    *,
    run_id: str,
    node_rank: int,
    clock_id: str,
    expected_gpus: Mapping[str, int],
) -> tuple[GpuSample, ...]:
    """Read valid complete frames; an unterminated final row is safely ignored."""

    raw = data.encode() if isinstance(data, str) else data
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith((b"\n", b"\r")):
        lines.pop()
    frames: dict[int, list[GpuSample]] = defaultdict(list)
    try:
        for line in lines:
            if not line.strip():
                raise ValueError("blank GPU JSONL row")
            sample = _parse_sample(load_json_strict(line))
            if (
                sample.run_id != run_id
                or sample.node_rank != node_rank
                or sample.clock_id != clock_id
            ):
                raise ValueError("GPU sample identity mismatch")
            frames[sample.sequence].append(sample)
    except (CanonicalizationError, ValueError, InvalidOperation, TypeError):
        return ()

    complete: list[GpuSample] = []
    prior_time: int | None = None
    for sequence in sorted(frames):
        frame = frames[sequence]
        by_uuid = {sample.gpu_uuid: sample for sample in frame}
        if len(by_uuid) != len(frame) or set(by_uuid) != set(expected_gpus):
            continue
        if any(
            sample.logical_index != expected_gpus[sample.gpu_uuid] for sample in frame
        ):
            continue
        timestamps = {sample.monotonic_ns for sample in frame}
        walls = {sample.wall_time_utc for sample in frame}
        if len(timestamps) != 1 or len(walls) != 1:
            continue
        frame_time = next(iter(timestamps))
        if prior_time is not None and frame_time <= prior_time:
            return ()
        prior_time = frame_time
        complete.extend(sorted(frame, key=lambda sample: sample.logical_index))
    return tuple(complete)


def reduce_measured_generation(
    samples: Sequence[GpuSample],
    *,
    node_rank: int,
    clock_id: str,
    start_ns: int | None,
    end_ns: int | None,
    expected_gpus: Mapping[str, int],
    nominal_interval_sec: Decimal = Decimal(1),
) -> tuple[GpuPhaseV1, ...]:
    """Reduce the node-local measured-generation window as step functions."""

    if (
        start_ns is None
        or end_ns is None
        or start_ns < 0
        or end_ns <= start_ns
        or nominal_interval_sec <= 0
        or not expected_gpus
    ):
        return ()
    if any(
        sample.node_rank != node_rank
        or sample.clock_id != clock_id
        or sample.gpu_uuid not in expected_gpus
        or sample.logical_index != expected_gpus[sample.gpu_uuid]
        for sample in samples
    ):
        return ()
    by_gpu: dict[str, list[GpuSample]] = defaultdict(list)
    for sample in samples:
        by_gpu[sample.gpu_uuid].append(sample)
    if set(by_gpu) != set(expected_gpus):
        return ()
    phase_duration = Decimal(end_ns - start_ns) / _NANOSECONDS
    max_step_ns = int(nominal_interval_sec * _NANOSECONDS * 2)
    rows: list[GpuPhaseV1] = []
    for uuid in sorted(expected_gpus, key=expected_gpus.__getitem__):
        ordered = sorted(by_gpu[uuid], key=lambda sample: sample.monotonic_ns)
        if any(
            right.monotonic_ns <= left.monotonic_ns for left, right in pairwise(ordered)
        ):
            return ()
        intervals: list[tuple[GpuSample, Decimal]] = []
        for index, sample in enumerate(ordered):
            next_ns = (
                ordered[index + 1].monotonic_ns if index + 1 < len(ordered) else end_ns
            )
            step_end = min(next_ns, end_ns, sample.monotonic_ns + max_step_ns)
            overlap_ns = max(0, step_end - max(sample.monotonic_ns, start_ns))
            if overlap_ns > 0:
                intervals.append((sample, Decimal(overlap_ns) / _NANOSECONDS))
        rows.append(_reduce_gpu_intervals(uuid, node_rank, phase_duration, intervals))
    return tuple(rows)


def _reduce_gpu_intervals(
    uuid: str,
    node_rank: int,
    phase_duration: Decimal,
    intervals: list[tuple[GpuSample, Decimal]],
) -> GpuPhaseV1:
    utilization = [
        (sample.utilization_percent, duration)
        for sample, duration in intervals
        if sample.utilization_percent is not None
    ]
    power = [
        (sample.power_w, duration)
        for sample, duration in intervals
        if sample.power_w is not None
    ]
    utilization_duration = sum((duration for _, duration in utilization), Decimal(0))
    power_duration = sum((duration for _, duration in power), Decimal(0))
    utilization_integral = sum(
        (value * duration for value, duration in utilization), Decimal(0)
    )
    energy = sum((value * duration for value, duration in power), Decimal(0))
    memory = [
        sample.memory_used_bytes
        for sample, _ in intervals
        if sample.memory_used_bytes is not None
    ]
    temperature = [
        sample.temperature_c
        for sample, _ in intervals
        if sample.temperature_c is not None
    ]
    clocks = [
        sample.sm_clock_mhz
        for sample, _ in intervals
        if sample.sm_clock_mhz is not None
    ]
    return GpuPhaseV1(
        phase="measured_generation",
        node_rank=node_rank,
        gpu_uuid=uuid,
        phase_duration_sec=phase_duration,
        valid_sample_count=len(intervals),
        valid_utilization_duration_sec=utilization_duration,
        utilization_coverage=utilization_duration / phase_duration,
        utilization_mean=(
            utilization_integral / utilization_duration
            if utilization_duration
            else None
        ),
        utilization_p95=_weighted_nearest_rank(utilization, Decimal("0.95")),
        valid_power_duration_sec=power_duration,
        power_coverage=power_duration / phase_duration,
        energy_j=energy if power_duration else None,
        power_mean=energy / power_duration if power_duration else None,
        valid_memory_sample_count=len(memory),
        memory_max_bytes=max(memory) if memory else None,
        valid_temperature_sample_count=len(temperature),
        temperature_max_c=max(temperature) if temperature else None,
        valid_sm_clock_sample_count=len(clocks),
        sm_clock_min_mhz=min(clocks) if clocks else None,
    )


def _weighted_nearest_rank(
    values: Sequence[tuple[Decimal, Decimal]], quantile: Decimal
) -> Decimal | None:
    if not values:
        return None
    total = sum((weight for _, weight in values), Decimal(0))
    threshold = total * quantile
    cumulative = Decimal(0)
    for value, weight in sorted(values, key=lambda item: item[0]):
        cumulative += weight
        if cumulative >= threshold:
            return value
    return max(value for value, _ in values)


def _sample_from_measurement(
    *,
    run_id: str,
    node_rank: int,
    sequence: int,
    clock_id: str,
    monotonic_ns: int,
    wall_time_utc: str,
    gpu_uuid: str,
    logical_index: int,
    values: Mapping[str, object | None],
) -> GpuSample:
    expected = {
        "logical_index",
        "utilization_percent",
        "memory_used_bytes",
        "power_w",
        "temperature_c",
        "sm_clock_mhz",
    }
    if set(values) != expected or values["logical_index"] != logical_index:
        raise ValueError("measurement is not a closed object")
    return _validated_sample(
        GpuSample(
            schema_version="gpu-sample/v1",
            run_id=run_id,
            node_rank=node_rank,
            sequence=sequence,
            clock_id=clock_id,
            monotonic_ns=monotonic_ns,
            wall_time_utc=wall_time_utc,
            gpu_uuid=gpu_uuid,
            logical_index=logical_index,
            utilization_percent=_optional_decimal(values["utilization_percent"]),
            memory_used_bytes=_optional_uint(values["memory_used_bytes"]),
            power_w=_optional_decimal(values["power_w"]),
            temperature_c=_optional_decimal(values["temperature_c"]),
            sm_clock_mhz=_optional_decimal(values["sm_clock_mhz"]),
        )
    )


def _parse_sample(value: object) -> GpuSample:
    if not isinstance(value, dict) or set(value) != _SAMPLE_FIELDS:
        raise ValueError("GPU sample is not a closed object")
    return _validated_sample(
        GpuSample(
            schema_version=value["schema_version"],
            run_id=value["run_id"],
            node_rank=_required_uint(value["node_rank"]),
            sequence=_required_uint(value["sequence"]),
            clock_id=value["clock_id"],
            monotonic_ns=_required_uint(value["monotonic_ns"]),
            wall_time_utc=value["wall_time_utc"],
            gpu_uuid=value["gpu_uuid"],
            logical_index=_required_uint(value["logical_index"]),
            utilization_percent=_optional_decimal(value["utilization_percent"]),
            memory_used_bytes=_optional_uint(value["memory_used_bytes"]),
            power_w=_optional_decimal(value["power_w"]),
            temperature_c=_optional_decimal(value["temperature_c"]),
            sm_clock_mhz=_optional_decimal(value["sm_clock_mhz"]),
        )
    )


def _validated_sample(sample: GpuSample) -> GpuSample:
    if (
        sample.schema_version != "gpu-sample/v1"
        or not isinstance(sample.run_id, str)
        or not sample.run_id
        or not isinstance(sample.clock_id, str)
        or not sample.clock_id
        or not isinstance(sample.gpu_uuid, str)
        or not sample.gpu_uuid
        or not isinstance(sample.wall_time_utc, str)
        or not _is_utc_timestamp(sample.wall_time_utc)
    ):
        raise ValueError("invalid GPU sample identity")
    if sample.utilization_percent is not None and not (
        0 <= sample.utilization_percent <= 100
    ):
        raise ValueError("GPU utilization is outside 0..100")
    for value in (sample.power_w, sample.temperature_c, sample.sm_clock_mhz):
        if value is not None and value < 0:
            raise ValueError("negative GPU measurement")
    return sample


def _is_utc_timestamp(value: str) -> bool:
    if not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo == UTC


def _required_uint(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("unsigned integer required")
    return value


def _optional_uint(value: object | None) -> int | None:
    return None if value is None else _required_uint(value)


def _optional_decimal(value: object | None) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise TypeError("decimal number required")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("finite decimal required")
    return result


def _nullable_decimal(value: str) -> Decimal | None:
    if value in _MISSING:
        return None
    result = Decimal(value)
    if not result.is_finite() or result < 0:
        raise ValueError("invalid NVIDIA decimal")
    return result


def _nullable_mib(value: str) -> int | None:
    result = _nullable_decimal(value)
    if result is None:
        return None
    bytes_value = result * 1024 * 1024
    if bytes_value != bytes_value.to_integral_value():
        raise ValueError("NVIDIA memory value is not an integral byte count")
    return int(bytes_value)


def _json_numbers(value: object) -> object:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {key: _json_numbers(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_numbers(child) for child in value]
    return value
