from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import Any

from .canonical import CanonicalizationError, load_json_strict

_IMB_BENCHMARKS = ("PingPong", "Sendrecv", "Allreduce")
_IMB_SECTION = re.compile(r"^#\s*Benchmarking\s+(PingPong|Sendrecv|Allreduce)\s*$")
_IMB_PROCESSES = re.compile(r"^#\s*#?processes\s*=\s*([0-9]+)\s*$", re.IGNORECASE)
_NCCL_COLLECTIVES = frozenset({"all_reduce", "all_gather", "reduce_scatter"})
_NCCL_TOP_LEVEL = {
    "version",
    "start_time",
    "args",
    "env",
    "nccl_version",
    "config",
    "results",
    "out_of_bounds",
    "average_bus_bandwidth",
    "errors",
    "end_time",
}
_NCCL_CONFIG = {
    "nthreads",
    "ngpus",
    "minimum_bytes",
    "maximum_bytes",
    "step_factor",
    "warmup_iters",
    "iterations",
    "aggregated_iterations",
    "validation",
    "graph",
    "blocking_collectives",
    "per_iter_timing",
    "per_iter_skip",
    "parallel_init",
    "devices",
}
_NCCL_DEVICE = {
    "rank",
    "group",
    "pid",
    "hostname",
    "device",
    "device_hex",
    "device_info",
}
_NCCL_RESULT = {
    "size",
    "count",
    "type",
    "redop",
    "root",
    "out_of_place",
    "in_place",
    "actual_iterations",
    "experiment_name",
}
_NCCL_MEASUREMENT = {"time", "alg_bw", "bus_bw", "nwrong"}


@dataclass(frozen=True, slots=True)
class ImbResult:
    benchmark: str
    process_count: int
    message_bytes: int
    iterations: int
    time_min_us: Decimal | None
    time_avg_us: Decimal | None
    time_max_us: Decimal | None
    bandwidth_mbytes_per_sec: Decimal | None


@dataclass(frozen=True, slots=True)
class MpiNetworkResultV2:
    run_id: str
    benchmark: str
    process_count: int
    message_bytes: int
    iterations: int
    time_min_us: Decimal | None
    time_avg_us: Decimal | None
    time_max_us: Decimal | None
    bandwidth_mbytes_per_sec: Decimal | None
    source_member: str

    def to_object(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NcclNetworkResultV2:
    run_id: str
    collective: str
    message_bytes: int
    element_count: int
    data_type: str
    reduction_op: str | None
    buffer_mode: str
    mpi_process_count: int
    gpu_count: int
    time_us: Decimal
    algorithm_bandwidth_gbps: Decimal
    bus_bandwidth_gbps: Decimal
    error_count: int
    source_member: str

    def to_object(self) -> dict[str, object]:
        return asdict(self)


def parse_imb_mpi1(text: str) -> tuple[ImbResult, ...]:
    """Return all complete target sections, or no rows for any bad capture."""

    try:
        return _parse_imb_mpi1_strict(text)
    except (ValueError, InvalidOperation):
        return ()


def project_mpi_network_results_v2(
    run_id: str, results: tuple[ImbResult, ...], source_member: str
) -> tuple[MpiNetworkResultV2, ...]:
    try:
        _validate_binding(run_id, source_member)
        rows = tuple(
            MpiNetworkResultV2(
                run_id=run_id, source_member=source_member, **asdict(row)
            )
            for row in results
        )
        grains = {
            (row.run_id, row.benchmark, row.process_count, row.message_bytes)
            for row in rows
        }
        if len(grains) != len(rows):
            return ()
        return rows
    except ValueError:
        return ()


def parse_nccl_tests_json(
    data: bytes | str,
    *,
    run_id: str,
    collective: str,
    source_member: str,
) -> tuple[NcclNetworkResultV2, ...]:
    """Project a complete nccl-tests v3 ``-J`` document, otherwise zero rows."""

    try:
        document = load_json_strict(data)
        _validate_binding(run_id, source_member)
        return _parse_nccl_document(document, run_id, collective, source_member)
    except (CanonicalizationError, ValueError, InvalidOperation, TypeError):
        return ()


def _parse_imb_mpi1_strict(text: str) -> tuple[ImbResult, ...]:
    lines = text.splitlines()
    starts = [
        (index, match.group(1))
        for index, line in enumerate(lines)
        if (match := _IMB_SECTION.fullmatch(line)) is not None
    ]
    counts = {name: 0 for name in _IMB_BENCHMARKS}
    rows: list[ImbResult] = []
    for position, (start, benchmark) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        rows.extend(_parse_imb_section(benchmark, lines[start:end]))
        counts[benchmark] += 1
    if not starts or any(count == 0 for count in counts.values()):
        raise ValueError("missing target IMB section")
    grains = {(row.benchmark, row.process_count, row.message_bytes) for row in rows}
    if len(grains) != len(rows):
        raise ValueError("duplicate IMB result grain")
    return tuple(rows)


def _parse_imb_section(benchmark: str, lines: list[str]) -> list[ImbResult]:
    process_count: int | None = None
    header: tuple[str, ...] | None = None
    rows: list[ImbResult] = []
    expected_header = (
        ("bytes", "repetitions", "t_min[usec]", "t_max[usec]", "t_avg[usec]")
        if benchmark == "Allreduce"
        else ("bytes", "repetitions", "t[usec]", "mbytes/sec")
    )
    for line in lines[1:]:
        if match := _IMB_PROCESSES.fullmatch(line):
            if process_count is not None:
                raise ValueError("duplicate process count")
            process_count = int(match.group(1))
            continue
        stripped = line.strip()
        if stripped.startswith("#") and "bytes" in stripped.lower():
            candidate = tuple(token.lstrip("#").lower() for token in stripped.split())
            if candidate != expected_header or header is not None:
                raise ValueError("unexpected IMB header")
            header = candidate
            continue
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if not tokens[0].isdecimal():
            continue  # IMB warnings and progress text are not result records.
        if process_count is None or process_count <= 0 or header is None:
            raise ValueError("IMB data before metadata")
        if len(tokens) != len(header):
            raise ValueError("truncated IMB row")
        columns = dict(zip(header, tokens, strict=True))
        message_bytes = _uint_token(columns["bytes"])
        iterations = _uint_token(columns["repetitions"])
        if iterations == 0:
            raise ValueError("zero iterations")
        if benchmark == "Allreduce":
            minimum = _finite_decimal(columns["t_min[usec]"])
            maximum = _finite_decimal(columns["t_max[usec]"])
            average = _finite_decimal(columns["t_avg[usec]"])
            if min(minimum, average, maximum) < 0 or not minimum <= average <= maximum:
                raise ValueError("inconsistent Allreduce times")
            rows.append(
                ImbResult(
                    benchmark,
                    process_count,
                    message_bytes,
                    iterations,
                    minimum,
                    average,
                    maximum,
                    None,
                )
            )
        else:
            elapsed = _finite_decimal(columns["t[usec]"])
            bandwidth = _finite_decimal(columns["mbytes/sec"])
            if elapsed < 0 or bandwidth < 0:
                raise ValueError("negative IMB measurement")
            rows.append(
                ImbResult(
                    benchmark,
                    process_count,
                    message_bytes,
                    iterations,
                    None,
                    None,
                    elapsed,
                    bandwidth,
                )
            )
    if process_count is None or header is None or not rows:
        raise ValueError("incomplete IMB section")
    return rows


def _parse_nccl_document(
    document: Any, run_id: str, collective: str, source_member: str
) -> tuple[NcclNetworkResultV2, ...]:
    if collective not in _NCCL_COLLECTIVES:
        raise ValueError("unknown collective")
    root = _object_with_keys(document, _NCCL_TOP_LEVEL, "NCCL document")
    if root["version"] != 3:
        raise ValueError("unsupported nccl-tests JSON version")
    _string(root["start_time"])
    _string(root["end_time"])
    _string_list(root["args"])
    _string_list(root["env"])
    _uint(root["nccl_version"], positive=True)
    if _string_list(root["errors"]):
        raise ValueError("nccl-tests reported runtime errors")
    summary = _object_with_keys(root["out_of_bounds"], {"count", "okay"}, "summary")
    _uint(summary["count"])
    if summary["okay"] not in {"true", "false"}:
        raise ValueError("invalid nccl-tests boolean encoding")
    average = _object_with_keys(
        root["average_bus_bandwidth"], {"bandwidth", "okay"}, "bandwidth summary"
    )
    _decimal_number(average["bandwidth"], nonnegative=True)
    if average["okay"] not in {"true", "false", "unchecked"}:
        raise ValueError("invalid bandwidth check state")

    config = _object_with_keys(root["config"], _NCCL_CONFIG, "NCCL config")
    nthreads = _uint(config["nthreads"], positive=True)
    ngpus = _uint(config["ngpus"], positive=True)
    minimum = _uint(config["minimum_bytes"], positive=True)
    maximum = _uint(config["maximum_bytes"], positive=True)
    factor = _uint(config["step_factor"], positive=True)
    if minimum < 8 or maximum > 1 << 30 or minimum > maximum or factor != 2:
        raise ValueError("unexpected NCCL size schedule")
    expected_config = {
        "warmup_iters": 5,
        "iterations": 20,
        "aggregated_iterations": 1,
        "validation": 1,
        "graph": 0,
        "per_iter_skip": 0,
    }
    if any(_uint(config[name]) != expected for name, expected in expected_config.items()):
        raise ValueError("NCCL run config differs from the fixed protocol")
    for name in ("blocking_collectives", "per_iter_timing", "parallel_init"):
        if config[name] not in {"true", "false"}:
            raise ValueError("invalid nccl-tests boolean encoding")
    if config["per_iter_timing"] != "false":
        raise ValueError("unexpected per-iteration payload")

    devices = config["devices"]
    if not isinstance(devices, list) or not devices:
        raise ValueError("missing NCCL devices")
    ranks: set[int] = set()
    for value in devices:
        device = _object_with_keys(value, _NCCL_DEVICE, "NCCL device")
        rank = _uint(device["rank"])
        ranks.add(rank)
        _uint(device["group"])
        _uint(device["pid"], positive=True)
        _uint(device["device"])
        for name in ("hostname", "device_hex", "device_info"):
            _string(device[name])
    gpu_count = len(devices)
    if ranks != set(range(gpu_count)) or gpu_count % (nthreads * ngpus):
        raise ValueError("inconsistent NCCL device grid")
    mpi_process_count = gpu_count // (nthreads * ngpus)

    results = root["results"]
    if not isinstance(results, list) or not results:
        raise ValueError("missing NCCL results")
    expected_sizes: list[int] = []
    size = minimum
    while size <= maximum:
        expected_sizes.append(size)
        size *= factor
    if size // factor != maximum:
        raise ValueError("maximum is not in the factor-two schedule")
    if [
        row.get("size") if isinstance(row, dict) else None for row in results
    ] != expected_sizes:
        raise ValueError("partial, duplicated, or unordered NCCL result schedule")

    rows: list[NcclNetworkResultV2] = []
    expected_op = "none" if collective == "all_gather" else "sum"
    for value in results:
        result = _object_with_keys(value, _NCCL_RESULT, "NCCL result")
        message_bytes = _uint(result["size"], positive=True)
        if not _is_power_of_two(message_bytes) or not 8 <= message_bytes <= 1 << 30:
            raise ValueError("invalid NCCL result size")
        element_count = _uint(result["count"], positive=True)
        if element_count * 4 != message_bytes:
            raise ValueError("NCCL float count does not match message size")
        if result["type"] != "float" or result["redop"] != expected_op:
            raise ValueError("NCCL datatype or reduction op mismatch")
        _string(result["root"])
        if result["experiment_name"] != "":
            raise ValueError("unexpected experiment name")
        if _uint(result["actual_iterations"], positive=True) != 20:
            raise ValueError("NCCL result has incomplete iterations")
        for buffer_mode in ("out_of_place", "in_place"):
            measurement = _object_with_keys(
                result[buffer_mode], _NCCL_MEASUREMENT, "NCCL measurement"
            )
            wrong = _decimal_number(measurement["nwrong"], nonnegative=True)
            if wrong != wrong.to_integral_value():
                raise ValueError("non-integral NCCL error count")
            rows.append(
                NcclNetworkResultV2(
                    run_id=run_id,
                    collective=collective,
                    message_bytes=message_bytes,
                    element_count=element_count,
                    data_type="float",
                    reduction_op=None if collective == "all_gather" else "sum",
                    buffer_mode=buffer_mode,
                    mpi_process_count=mpi_process_count,
                    gpu_count=gpu_count,
                    time_us=_decimal_number(measurement["time"], nonnegative=True),
                    algorithm_bandwidth_gbps=_decimal_number(
                        measurement["alg_bw"], nonnegative=True
                    ),
                    bus_bandwidth_gbps=_decimal_number(
                        measurement["bus_bw"], nonnegative=True
                    ),
                    error_count=int(wrong),
                    source_member=source_member,
                )
            )
    grains = {
        (row.run_id, row.collective, row.message_bytes, row.buffer_mode) for row in rows
    }
    if len(grains) != len(rows):
        raise ValueError("duplicate NCCL result grain")
    total_errors = sum(row.error_count for row in rows)
    if summary["count"] != total_errors or summary["okay"] != (
        "true" if total_errors == 0 else "false"
    ):
        raise ValueError("NCCL error summary does not match result rows")
    return tuple(rows)


def _object_with_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} is not a closed object")
    return value


def _uint(value: Any, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("unsigned integer required")
    if positive and value == 0:
        raise ValueError("positive integer required")
    return value


def _uint_token(value: str) -> int:
    if not value.isdecimal():
        raise ValueError("unsigned decimal token required")
    return int(value)


def _finite_decimal(value: str) -> Decimal:
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError("finite decimal required")
    return result


def _decimal_number(value: Any, *, nonnegative: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("JSON number required")
    result = Decimal(str(value))
    if not result.is_finite() or (nonnegative and result < 0):
        raise ValueError("invalid finite number")
    return result


def _string(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("non-empty string required")
    return value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("string array required")
    return value


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _validate_binding(run_id: str, source_member: str) -> None:
    if not run_id:
        raise ValueError("run_id must be non-empty")
    path = PurePosixPath(source_member)
    if (
        not source_member
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != source_member
    ):
        raise ValueError("source_member must be a canonical relative POSIX path")
