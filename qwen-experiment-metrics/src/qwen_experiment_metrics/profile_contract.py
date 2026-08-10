"""Closed, profile-only phase, GPU, and Torch-profiler projections.

This module is a pure artifact boundary.  It does not start a sampler, import
cloud credentials, or publish facts.  The existing node sampler remains the
owner of GPU JSONL production; this module validates its files and projects
their lineage together with phase and profiler artifacts.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .canonical import jcs_bytes, load_json_strict
from .gpu import GpuPhaseV1, parse_gpu_samples_jsonl, reduce_measured_generation

PROFILE_ID = "qwen-profile"
PROFILE_ACTIVITIES = ("CPU", "GPU")
PROFILE_STEPS = 5
PROFILE_WITH_STACK = False
PROFILE_RECORD_SHAPES = False
NVIDIA_SMI_SAMPLE_ARGV = (
    "nvidia-smi",
    "--query-gpu=uuid,index,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm",
    "--format=csv,noheader,nounits",
)
DCGM_OPTIONAL_FIELDS = (
    "engine_active_ratio",
    "tensor_active_ratio",
    "dram_active_ratio",
    "pcie_rx_bytes_per_sec",
    "pcie_tx_bytes_per_sec",
    "nvlink_bytes_per_sec",
    "xid_error_count",
)
_RUN_ID = re.compile(r"^r-[a-z0-9][a-z0-9._-]{31}$")
_MEMBER = re.compile(r"^[^/\\.][^/\\]*(?:/[^/\\.][^/\\]*)*$")
_TRACE = re.compile(r"^qwen-profile-TP-([0-9]+)\.trace\.json\.gz$")
_CAPTURE = re.compile(r"^qwen-profile-TP-([0-9]+)\.capture\.json$")
_PHASE_KINDS = ("warmup", "measured")


class ProfileValidationError(ValueError):
    """A profile source is malformed, partial, or identity-inconsistent."""


class ProfileConflictError(ProfileValidationError):
    def __init__(self, conflicts: tuple[ProjectionConflict, ...]) -> None:
        self.conflicts = conflicts
        super().__init__(f"{len(conflicts)} profile projection conflict(s)")


@dataclass(frozen=True, slots=True)
class PhaseNode:
    node_rank: int
    clock_id: str
    start_sequence: int
    end_sequence: int
    start_received_ns: int
    start_ack_ns: int
    end_received_ns: int
    end_ack_ns: int


@dataclass(frozen=True, slots=True)
class PhaseRank0:
    clock_id: str
    start_broadcast_ns: int
    all_start_acks_received_ns: int
    generation_start_ns: int
    generation_end_ns: int
    end_broadcast_ns: int
    all_end_acks_received_ns: int


@dataclass(frozen=True, slots=True)
class PhaseSummary:
    run_id: str
    invocation_id: str
    kind: str
    rank0: PhaseRank0
    nodes: tuple[PhaseNode, ...]
    source_member: str
    source_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "phase-summary-projection/v1",
            "run_id": self.run_id,
            "invocation_id": self.invocation_id,
            "kind": self.kind,
            "rank0": {
                "clock_id": self.rank0.clock_id,
                "start_broadcast_ns": self.rank0.start_broadcast_ns,
                "all_start_acks_received_ns": self.rank0.all_start_acks_received_ns,
                "generation_start_ns": self.rank0.generation_start_ns,
                "generation_end_ns": self.rank0.generation_end_ns,
                "end_broadcast_ns": self.rank0.end_broadcast_ns,
                "all_end_acks_received_ns": self.rank0.all_end_acks_received_ns,
            },
            "nodes": [
                {
                    "node_rank": node.node_rank,
                    "clock_id": node.clock_id,
                    "start_sequence": node.start_sequence,
                    "end_sequence": node.end_sequence,
                    "start_received_ns": node.start_received_ns,
                    "start_ack_ns": node.start_ack_ns,
                    "end_received_ns": node.end_received_ns,
                    "end_ack_ns": node.end_ack_ns,
                }
                for node in self.nodes
            ],
            "source_member": self.source_member,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProfilerCapture:
    run_id: str
    global_rank: int
    invocation_id: str
    source_member: str
    size_bytes: int
    sha256: str
    capture_member: str
    capture_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "profiler-capture/v1",
            "run_id": self.run_id,
            "global_rank": self.global_rank,
            "invocation_id": self.invocation_id,
            "source_member": self.source_member,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "capture_member": self.capture_member,
            "capture_sha256": self.capture_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProfilerEventRow:
    run_id: str
    global_rank: int
    activity: str
    event_category: str
    event_name: str
    call_count: int
    inclusive_duration_sum_ns: Decimal
    source_member: str
    source_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "profiler-event-projection/v1",
            "run": self.run_id,
            "global_rank": self.global_rank,
            "activity": self.activity,
            "event_category": self.event_category,
            "event_name": self.event_name,
            "call_count": self.call_count,
            "inclusive_duration_sum_ns": str(self.inclusive_duration_sum_ns),
            "source_member": self.source_member,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class GpuProjectionRow:
    run_id: str
    phase: GpuPhaseV1
    source_member: str
    source_sha256: str
    phase_member: str
    phase_sha256: str

    def as_dict(self) -> dict[str, object]:
        values = self.phase.to_object()
        return {
            "projection_relation": "qwen_gpu_phase_v2",
            "run": self.run_id,
            **values,
            "source_member": self.source_member,
            "source_sha256": self.source_sha256,
            "phase_member": self.phase_member,
            "phase_sha256": self.phase_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProjectionConflict:
    grain: tuple[object, ...]
    rows: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class ProfileProjection:
    run_id: str
    expected_ranks: int
    phase: tuple[PhaseSummary, ...]
    captures: tuple[ProfilerCapture, ...]
    profiler_rows: tuple[ProfilerEventRow, ...]
    gpu_rows: tuple[GpuProjectionRow, ...]
    missing_ranks: tuple[int, ...]
    partial_ranks: tuple[int, ...]
    conflicts: tuple[ProjectionConflict, ...] = ()

    @property
    def collection_complete(self) -> bool:
        return not self.missing_ranks and not self.conflicts

    def require_conflict_free(self) -> ProfileProjection:
        if self.conflicts:
            raise ProfileConflictError(self.conflicts)
        return self


def profiler_environment() -> dict[str, str]:
    """Return the fixed profile overlay environment, without side effects."""

    return {
        "SGLANG_PROFILE_ID": PROFILE_ID,
        "SGLANG_PROFILE_WITH_STACK": "0",
        "SGLANG_PROFILE_RECORD_SHAPES": "0",
    }


def dcgm_supported_fields(supported: Iterable[str]) -> tuple[str, ...]:
    """Return only explicitly probed optional DCGM fields; no fallback is made."""

    values = tuple(supported)
    if any(field not in DCGM_OPTIONAL_FIELDS for field in values):
        raise ProfileValidationError("unknown DCGM field")
    return tuple(field for field in DCGM_OPTIONAL_FIELDS if field in values)


def profiler_overlay_args() -> tuple[str, ...]:
    """Exact built-in legacy profiler flags fixed by D-004."""

    return (
        "--profile",
        "--profile-activities",
        *PROFILE_ACTIVITIES,
        "--profile-steps",
        str(PROFILE_STEPS),
    )


def validate_profiler_overlay(argv: Sequence[str], environment: Mapping[str, str]) -> None:
    expected = profiler_overlay_args()
    if tuple(argv[-len(expected) :]) != expected:
        raise ProfileValidationError("profiler argv does not use the fixed overlay")
    expected_env = profiler_environment()
    if {name: environment.get(name) for name in expected_env} != expected_env:
        raise ProfileValidationError("profiler environment is not fixed")


def parse_phase_summary(
    path: Path, *, run_id: str, expected_nodes: int
) -> PhaseSummary:
    _run_id(run_id)
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "run_id",
        "invocation_id",
        "kind",
        "rank0",
        "nodes",
    }:
        raise ProfileValidationError("phase summary is not a closed object")
    if value["schema_version"] != "phase-summary/v1" or value["run_id"] != run_id:
        raise ProfileValidationError("phase summary identity differs")
    invocation_id = _string(value["invocation_id"], "invocation_id")
    kind = value["kind"]
    if kind not in _PHASE_KINDS:
        raise ProfileValidationError("phase kind is invalid")
    rank0 = _phase_rank0(value["rank0"])
    nodes_value = value["nodes"]
    if not isinstance(nodes_value, list) or len(nodes_value) != expected_nodes:
        raise ProfileValidationError("phase node cardinality differs")
    nodes = tuple(_phase_node(item) for item in nodes_value)
    if tuple(node.node_rank for node in nodes) != tuple(range(expected_nodes)):
        raise ProfileValidationError("phase node order differs")
    _validate_phase_order(rank0, nodes)
    member = _relative_member(path, path.parent.parent.parent)
    return PhaseSummary(run_id, invocation_id, kind, rank0, nodes, member, _sha256_file(path))


def project_phase_summary(
    runtime_dir: Path, *, run_id: str, expected_nodes: int
) -> tuple[PhaseSummary, ...]:
    path = runtime_dir / "control/phase/summary.json"
    if not path.is_file():
        return ()
    return (parse_phase_summary(path, run_id=run_id, expected_nodes=expected_nodes),)


def parse_profiler_capture(
    capture_path: Path,
    trace_path: Path,
    *,
    run_id: str,
    global_rank: int,
    invocation_id: str,
) -> ProfilerCapture:
    _run_id(run_id)
    value = load_json_strict(capture_path.read_bytes())
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "run_id",
        "global_rank",
        "invocation_id",
        "source_member",
        "size_bytes",
        "sha256",
    }:
        raise ProfileValidationError("profiler capture is not a closed object")
    member = _relative_member(trace_path, trace_path.parent.parent.parent.parent)
    if (
        value["schema_version"] != "profiler-capture/v1"
        or value["run_id"] != run_id
        or value["global_rank"] != global_rank
        or value["invocation_id"] != invocation_id
        or value["source_member"] != member
        or value["size_bytes"] != trace_path.stat().st_size
        or value["sha256"] != _sha256_file(trace_path)
    ):
        raise ProfileValidationError("profiler capture lineage differs")
    if (
        isinstance(value["global_rank"], bool)
        or not isinstance(value["global_rank"], int)
        or isinstance(value["size_bytes"], bool)
        or not isinstance(value["size_bytes"], int)
        or value["size_bytes"] < 0
    ):
        raise ProfileValidationError("profiler capture numeric fields are invalid")
    _sha256(value["sha256"])
    return ProfilerCapture(
        run_id,
        global_rank,
        invocation_id,
        member,
        trace_path.stat().st_size,
        value["sha256"],
        _relative_member(capture_path, capture_path.parent.parent.parent.parent),
        _sha256_file(capture_path),
    )


def parse_chrome_trace(
    trace_path: Path,
    *,
    run_id: str,
    global_rank: int,
    source_member: str,
    source_sha256: str,
) -> tuple[ProfilerEventRow, ...]:
    """Project only complete-duration Chrome events; metadata is ignored."""

    _run_id(run_id)
    _sha256(source_sha256)
    try:
        document = json.loads(gzip.decompress(trace_path.read_bytes()))
    except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProfileValidationError("profiler trace is not valid gzip JSON") from error
    if not isinstance(document, dict) or not isinstance(document.get("traceEvents"), list):
        raise ProfileValidationError("profiler trace has no traceEvents array")
    grouped: dict[tuple[str, str, str], list[Decimal]] = defaultdict(list)
    for event in document["traceEvents"]:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        name = event.get("name")
        category = event.get("cat", "")
        if not isinstance(name, str) or not name:
            continue
        if isinstance(category, list):
            category = ",".join(str(item) for item in category)
        if not isinstance(category, str):
            continue
        duration = event.get("dur")
        try:
            duration_value = Decimal(str(duration))
        except (InvalidOperation, ValueError):
            continue
        if not duration_value.is_finite() or duration_value < 0:
            continue
        activity = _trace_activity(category, name)
        # Chrome trace ``dur`` is expressed in microseconds; the closed
        # projection contract stores the sum in nanoseconds.
        grouped[(activity, category, name)].append(duration_value * Decimal(1000))
    return tuple(
        ProfilerEventRow(
            run_id,
            global_rank,
            activity,
            category,
            name,
            len(durations),
            sum(durations, Decimal(0)),
            source_member,
            source_sha256,
        )
        for activity, category, name in sorted(grouped)
        for durations in (grouped[(activity, category, name)],)
    )


def project_profiler_events(
    runtime_dir: Path,
    *,
    run_id: str,
    invocation_id: str,
    expected_ranks: int,
) -> ProfileProjection:
    """Validate fixed trace/capture paths and project observed event rows."""

    _run_id(run_id)
    if expected_ranks <= 0:
        raise ProfileValidationError("expected rank count must be positive")
    torch_dir = runtime_dir / "profile/raw/torch"
    traces: dict[int, Path] = {}
    captures: dict[int, Path] = {}
    if torch_dir.is_dir():
        for path in torch_dir.iterdir():
            trace_match = _TRACE.fullmatch(path.name)
            capture_match = _CAPTURE.fullmatch(path.name)
            if trace_match:
                traces[_parse_rank(trace_match.group(1))] = path
            elif capture_match:
                captures[_parse_rank(capture_match.group(1))] = path
            else:
                raise ProfileValidationError(
                    f"unexpected profiler member: {path.name}"
                )
    for rank in (*traces, *captures):
        if rank >= expected_ranks:
            raise ProfileValidationError("profiler rank is outside expected range")
    missing = tuple(
        rank
        for rank in range(expected_ranks)
        if rank not in traces or rank not in captures
    )
    rows: list[ProfilerEventRow] = []
    capture_rows: list[ProfilerCapture] = []
    partial: list[int] = []
    for rank in sorted(set(traces) & set(captures)):
        trace = traces[rank]
        capture = parse_profiler_capture(
            captures[rank],
            trace,
            run_id=run_id,
            global_rank=rank,
            invocation_id=invocation_id,
        )
        capture_rows.append(capture)
        trace_rows = parse_chrome_trace(
            trace,
            run_id=run_id,
            global_rank=rank,
            source_member=capture.source_member,
            source_sha256=capture.sha256,
        )
        if not trace_rows:
            partial.append(rank)
        rows.extend(trace_rows)
    unique, conflicts = dedupe_profiler_rows(rows)
    return ProfileProjection(
        run_id,
        expected_ranks,
        (),
        tuple(capture_rows),
        unique,
        (),
        missing,
        tuple(partial),
        conflicts,
    )


def dedupe_profiler_rows(
    rows: Iterable[ProfilerEventRow],
) -> tuple[tuple[ProfilerEventRow, ...], tuple[ProjectionConflict, ...]]:
    unique, conflicts = _dedupe(
        rows,
        key=lambda row: (
            row.run_id,
            row.global_rank,
            row.activity,
            row.event_category,
            row.event_name,
        ),
    )
    return unique, conflicts  # type: ignore[return-value]


def dedupe_gpu_rows(
    rows: Iterable[GpuProjectionRow],
) -> tuple[tuple[GpuProjectionRow, ...], tuple[ProjectionConflict, ...]]:
    unique, conflicts = _dedupe(
        rows,
        key=lambda row: (row.run_id, row.phase.node_rank, row.phase.gpu_uuid, row.phase.phase),
    )
    return unique, conflicts  # type: ignore[return-value]


def project_gpu_phase(
    runtime_dir: Path,
    *,
    run_id: str,
    phase: PhaseSummary,
    expected_gpus: Mapping[int, Mapping[str, int]],
    nominal_interval_sec: Decimal = Decimal(1),
) -> tuple[GpuProjectionRow, ...]:
    """Use the existing GPU reducer with phase-window and source lineage."""

    _run_id(run_id)
    if phase.run_id != run_id or phase.kind != "measured":
        raise ProfileValidationError("GPU phase must be the measured invocation")
    rows: list[GpuProjectionRow] = []
    for node in phase.nodes:
        if node.node_rank not in expected_gpus:
            raise ProfileValidationError("phase node has no expected GPU grid")
        path = runtime_dir / f"profile/raw/gpu/node-{node.node_rank}.jsonl"
        if not path.is_file():
            continue
        raw = path.read_bytes()
        try:
            first = load_json_strict(raw.splitlines()[0])
            clock_id = first["clock_id"]
            if clock_id != node.clock_id:
                raise ProfileValidationError("GPU sample and phase clocks differ")
            samples = parse_gpu_samples_jsonl(
                raw,
                run_id=run_id,
                node_rank=node.node_rank,
                clock_id=clock_id,
                expected_gpus=expected_gpus[node.node_rank],
            )
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ProfileValidationError("GPU sample source is invalid") from error
        reduced = reduce_measured_generation(
            samples,
            node_rank=node.node_rank,
            clock_id=clock_id,
            start_ns=node.start_received_ns,
            end_ns=node.end_received_ns,
            expected_gpus=expected_gpus[node.node_rank],
            nominal_interval_sec=nominal_interval_sec,
        )
        member = f"profile/raw/gpu/node-{node.node_rank}.jsonl"
        digest = _sha256_file(path)
        for value in reduced:
            rows.append(
                GpuProjectionRow(
                    run_id,
                    value,
                    member,
                    digest,
                    phase.source_member,
                    phase.source_sha256,
                )
            )
    return rows


def _dedupe(
    rows: Iterable[object],
    *,
    key: object,
) -> tuple[tuple[object, ...], tuple[ProjectionConflict, ...]]:
    groups: dict[tuple[object, ...], list[object]] = defaultdict(list)
    key_fn = key
    for row in rows:
        groups[key_fn(row)].append(row)  # type: ignore[operator]
    unique: list[object] = []
    conflicts: list[ProjectionConflict] = []
    for grain in sorted(groups, key=lambda item: jcs_bytes(list(item))):
        values: dict[bytes, object] = {}
        for row in groups[grain]:
            values.setdefault(jcs_bytes(_as_dict(row)), row)
        ordered = tuple(values[item] for item in sorted(values))
        if len(ordered) == 1:
            unique.append(ordered[0])
        else:
            conflicts.append(ProjectionConflict(grain, ordered))
    return tuple(unique), tuple(conflicts)


def _as_dict(value: object) -> dict[str, object]:
    if not hasattr(value, "as_dict"):
        raise ProfileValidationError("projection row has no closed representation")
    return value.as_dict()  # type: ignore[no-any-return]


def _phase_rank0(value: object) -> PhaseRank0:
    if not isinstance(value, dict) or set(value) != {
        "clock_id", "start_broadcast_ns", "all_start_acks_received_ns",
        "generation_start_ns", "generation_end_ns", "end_broadcast_ns",
        "all_end_acks_received_ns",
    }:
        raise ProfileValidationError("phase rank0 object is not closed")
    return PhaseRank0(
        _string(value["clock_id"], "clock_id"),
        *(_uint(value[name], name) for name in (
            "start_broadcast_ns", "all_start_acks_received_ns", "generation_start_ns",
            "generation_end_ns", "end_broadcast_ns", "all_end_acks_received_ns",
        )),
    )


def _phase_node(value: object) -> PhaseNode:
    names = {
        "node_rank", "clock_id", "start_sequence", "end_sequence",
        "start_received_ns", "start_ack_ns", "end_received_ns", "end_ack_ns",
    }
    if not isinstance(value, dict) or set(value) != names:
        raise ProfileValidationError("phase node object is not closed")
    return PhaseNode(
        _uint(value["node_rank"], "node_rank"),
        _string(value["clock_id"], "clock_id"),
        _uint(value["start_sequence"], "start_sequence"),
        _uint(value["end_sequence"], "end_sequence"),
        _uint(value["start_received_ns"], "start_received_ns"),
        _uint(value["start_ack_ns"], "start_ack_ns"),
        _uint(value["end_received_ns"], "end_received_ns"),
        _uint(value["end_ack_ns"], "end_ack_ns"),
    )


def _validate_phase_order(rank0: PhaseRank0, nodes: Sequence[PhaseNode]) -> None:
    if not (
        rank0.start_broadcast_ns <= rank0.all_start_acks_received_ns
        <= rank0.generation_start_ns <= rank0.generation_end_ns
        <= rank0.end_broadcast_ns <= rank0.all_end_acks_received_ns
    ):
        raise ProfileValidationError("rank0 phase timestamps are not ordered")
    for node in nodes:
        if node.start_sequence >= node.end_sequence or not (
            node.start_received_ns <= node.start_ack_ns
            <= node.end_received_ns <= node.end_ack_ns
        ):
            raise ProfileValidationError("node phase window is invalid")
        if node.node_rank == 0 and node.clock_id != rank0.clock_id:
            raise ProfileValidationError("rank zero phase clock IDs differ")


def _trace_activity(category: str, name: str) -> str:
    text = f"{category} {name}".lower()
    return "GPU" if any(token in text for token in ("cuda", "gpu", "kernel")) else "CPU"


def _relative_member(path: Path, root: Path) -> str:
    if path.is_symlink():
        raise ProfileValidationError("source member must not be a symlink")
    resolved = path.resolve(strict=True)
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ProfileValidationError("source path escapes runtime root")
    member = resolved.relative_to(root_resolved).as_posix()
    if _MEMBER.fullmatch(member) is None:
        raise ProfileValidationError("source member is unsafe")
    return member


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ProfileValidationError("SHA-256 is invalid")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProfileValidationError(f"{label} must be nonempty")
    return value


def _uint(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProfileValidationError(f"{label} must be uint")
    return value


def _parse_rank(value: str) -> int:
    if str(int(value)) != value:
        raise ProfileValidationError("rank filename is not canonical")
    return _uint(int(value), "global_rank")


def _run_id(value: object) -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise ProfileValidationError("run ID is not canonical")
    return value
