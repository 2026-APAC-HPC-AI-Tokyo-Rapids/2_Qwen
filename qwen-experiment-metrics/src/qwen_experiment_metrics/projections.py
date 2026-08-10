"""Pure, closed artifact projections for the Qwen v2 contract.

The module is intentionally independent of the registry/publisher.  A source
member is parsed once, projected to a declared relation, and carries its
member SHA-256 as lineage.  Rows are closed (unknown columns are rejected),
and append callers must handle non-identical grain conflicts explicitly.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath

from .artifact_store import ROLE_FILENAMES
from .canonical import jcs_bytes, load_json_strict
from .gpu import parse_gpu_samples_jsonl
from .manifest import load_manifest
from .network import (
    MpiNetworkResultV2,
    NcclNetworkResultV2,
    parse_imb_mpi1,
    parse_nccl_tests_json,
    project_mpi_network_results_v2,
)
from .profile_contract import (
    ProfileValidationError,
    project_gpu_phase,
    project_profiler_events,
)
from .sources import (
    validate_cache_manifest,
    validate_dataset_manifest,
    validate_model_manifest,
    validate_runtime_launchers,
    validate_sif_sidecar,
)
from .system_sources import (
    validate_environment_snapshot,
    validate_lifecycle_summary,
    validate_minimal_hardware,
    validate_node_lifecycle,
    validate_node_topology,
)
from .workload import validate_ordered_workload

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^r-[0-9a-f]{32}$")


class ProjectionError(ValueError):
    """A source or projection row violates the v2 contract."""


class ProjectionConflictError(ProjectionError):
    def __init__(self, conflicts: tuple[ProjectionConflict, ...]) -> None:
        self.conflicts = conflicts
        super().__init__(f"{len(conflicts)} projection grain conflict(s)")


@dataclass(frozen=True, slots=True)
class RelationSpec:
    relation_id: str
    schema_version: str
    grain: tuple[str, ...]
    applicability: tuple[str, ...]
    fields: tuple[str, ...]
    lineage: bool = True


def _spec(
    relation_id: str,
    schema_version: str,
    grain: tuple[str, ...],
    applicability: tuple[str, ...],
    *fields: str,
    lineage: bool = True,
) -> RelationSpec:
    return RelationSpec(
        relation_id,
        schema_version,
        grain,
        applicability,
        tuple(sorted(fields, key=lambda item: item.encode())),
        lineage,
    )


_ALL = ("bench", "profile")
_PROFILE = ("profile",)
_GENERATION = ("bench", "profile")

# This is the code-owned projection registry.  The schema document is the
# authority for these names, grains, and applicability; it is not inferred
# from a BigQuery table at runtime.
RELATION_REGISTRY: dict[str, RelationSpec] = {
    "qwen_source_track_v2": _spec(
        "qwen_source_track_v2", "source-track-projection/v1", ("run",), _ALL,
        "run_purpose", "repository_id", "source_track", "source_track_sha256",
    ),
    "qwen_sif_provenance_v2": _spec(
        "qwen_sif_provenance_v2", "qwen-sif-provenance-projection/v1", ("run",), _ALL,
        "sif_artifact_sha256", "source_commits", "score_source_tree", "recipe",
        "applied_patches", "base_oci", "lock_manifest", "build_command",
        "toolchain_observations", "embedded_manifest_identity", "runtime_files",
    ),
    "qwen_runtime_launchers_v2": _spec(
        "qwen_runtime_launchers_v2", "runtime-launcher-projection/v1", ("run", "role"), _ALL,
        "executables", "versions",
    ),
    "qwen_model_manifests_v2": _spec(
        "qwen_model_manifests_v2", "model-manifest-projection/v1", ("run", "kind"), _ALL,
        "source_kind", "repository", "revision", "prebuilt_shards", "small_metadata",
        "manifest_sha256",
    ),
    "qwen_dataset_manifests_v2": _spec(
        "qwen_dataset_manifests_v2", "dataset-manifest-projection/v1", ("run",), _ALL,
        "repository", "full_revision", "relative_path", "size_bytes", "content_sha256",
        "manifest_sha256",
    ),
    "qwen_cache_manifests_v2": _spec(
        "qwen_cache_manifests_v2", "cache-manifest-projection/v1", ("run", "node_rank"), _ALL,
        "node_rank", "semantic_manifest_sha256", "members",
    ),
    "qwen_workload_manifests_raw_v2": _spec(
        "qwen_workload_manifests_raw_v2", "workload-manifest-projection/v1",
        ("run", "workload_kind"), _ALL,
        "workload_kind", "dataset_manifest_sha256", "warmup_mode",
    ),
    "qwen_workload_requests_raw_v2": _spec(
        "qwen_workload_requests_raw_v2", "workload-request-projection/v1",
        ("run", "workload_kind", "ordinal"), _ALL,
        "workload_kind", "ordinal", "request", lineage=False,
    ),
    "qwen_minimal_hardware_v2": _spec(
        "qwen_minimal_hardware_v2", "minimal-hardware-projection/v1",
        ("run", "node_rank"), _ALL,
        "node_rank", "hostname", "cpu_identity", "platform_settings",
    ),
    "qwen_node_launch_v2": _spec(
        "qwen_node_launch_v2", "node-launch-projection/v1",
        ("run", "node_rank", "global_rank"), _ALL,
        "node_rank", "global_rank", "local_rank", "size", "hostname",
        "node_wrapper_argv", "singularity_argv", "sglang_argv", "cpu_affinity", "numa_mems",
    ),
    "qwen_topology_raw_v2": _spec(
        "qwen_topology_raw_v2", "topology-projection/v1",
        ("run", "node_rank", "scope", "index"), _ALL,
        "node_rank", "scope", "index", "name", "uuid", "pci_bus_id",
        "memory_total_bytes", "driver_version", "persistence_mode", "compute_mode",
        "power_limit_w", "max_sm_clock_mhz", "max_memory_clock_mhz", "mig_mode", "ecc_mode",
    ),
    "qwen_environment_v2": _spec(
        "qwen_environment_v2", "environment-projection/v1",
        ("run", "scope", "name"), _ALL,
        "scope", "name", "value_base64",
    ),
    "qwen_lifecycle_v2": _spec(
        "qwen_lifecycle_v2", "lifecycle-summary-projection/v1", ("run", "kind"), _ALL,
        "kind", "summary",
    ),
    "qwen_node_lifecycle_v2": _spec(
        "qwen_node_lifecycle_v2", "node-lifecycle-projection/v1",
        ("run", "node_rank"), _ALL,
        "node_rank", "lifecycle",
    ),
    "qwen_primary_results_v2": _spec(
        "qwen_primary_results_v2", "primary-result-projection/v1",
        ("run", "invocation_id"), _ALL,
        "invocation_id", "generation_wall_sec", "returned_count", "error_count",
        "observed_input_tokens", "observed_output_tokens", "ordered_output_token_ids_sha256",
    ),
    "qwen_phase_summaries_v2": _spec(
        "qwen_phase_summaries_v2", "phase-summary-projection/v1",
        ("run", "invocation_id"), _PROFILE,
        "invocation_id", "kind", "rank0", "nodes",
    ),
    "qwen_gpu_samples_raw_v2": _spec(
        "qwen_gpu_samples_raw_v2", "gpu-sample-projection/v1",
        ("run", "node_rank", "sequence", "gpu_uuid"), _PROFILE,
        "node_rank", "sequence", "clock_id", "monotonic_ns", "wall_time_utc",
        "gpu_uuid", "logical_index", "utilization_percent", "memory_used_bytes",
        "power_w", "temperature_c", "sm_clock_mhz",
    ),
    "qwen_gpu_phases_v2": _spec(
        "qwen_gpu_phases_v2", "gpu-phase/v1",
        ("run", "phase", "node_rank", "gpu_uuid"), _PROFILE,
        "phase", "node_rank", "gpu_uuid", "phase_duration_sec", "valid_sample_count",
        "valid_utilization_duration_sec", "utilization_coverage", "utilization_mean",
        "utilization_p95", "valid_power_duration_sec", "power_coverage", "energy_j",
        "power_mean", "valid_memory_sample_count", "memory_max_bytes",
        "valid_temperature_sample_count", "temperature_max_c", "valid_sm_clock_sample_count",
        "sm_clock_min_mhz",
    ),
    "qwen_profile_hardware_raw_v2": _spec(
        "qwen_profile_hardware_raw_v2", "profile-hardware-projection/v1",
        ("run", "node_rank", "record_kind", "numa_node_id"), _PROFILE,
        "node_rank", "hostname", "record_kind", "numa_node_id",
        "postrun_memory_available_bytes", "cpulist", "mem_total_bytes", "mem_free_bytes",
        "source_members",
    ),
    "qwen_profiler_events_raw_v2": _spec(
        "qwen_profiler_events_raw_v2", "profiler-event-projection/v1",
        ("run", "global_rank", "activity", "event_category", "event_name"), _PROFILE,
        "global_rank", "activity", "event_category", "event_name", "call_count",
        "inclusive_duration_sum_ns",
    ),
    "qwen_config_snapshots_v2": _spec(
        "qwen_config_snapshots_v2", "config-snapshot-projection/v1",
        ("run", "config_type"), _ALL,
        "config_type", "definition", "fields",
    ),
    "qwen_placement_v2": _spec(
        "qwen_placement_v2", "placement-projection/v1",
        ("run", "node_rank", "global_rank"), _ALL,
        "node_rank", "global_rank", "server_args", "worker_ready", "topology",
    ),
    "qwen_artifacts_v2": _spec(
        "qwen_artifacts_v2", "artifact-index-projection/v1", ("run", "role"), _ALL,
        "role", "path", "size_bytes", "sha256", "run_purpose", "logical_artifact_name",
    ),
    "qwen_mpi_network_results_v2": _spec(
        "qwen_mpi_network_results_v2", "mpi-network-result-projection/v1",
        ("run", "benchmark", "process_count", "message_bytes"), _GENERATION,
        "benchmark", "process_count", "message_bytes", "iterations", "time_min_us",
        "time_avg_us", "time_max_us", "bandwidth_mbytes_per_sec",
    ),
    "qwen_nccl_network_results_v2": _spec(
        "qwen_nccl_network_results_v2", "nccl-network-result-projection/v1",
        ("run", "collective", "message_bytes", "buffer_mode"), _GENERATION,
        "collective", "message_bytes", "element_count", "data_type", "reduction_op",
        "buffer_mode", "mpi_process_count", "gpu_count", "time_us",
        "algorithm_bandwidth_gbps", "bus_bandwidth_gbps", "error_count",
    ),
}

# Backward-compatible names used by the first append adapter.  They point to
# the same closed schemas, never to a weaker six-field schema.
RELATION_ALIASES = {
    "qwen_mpi_network_results_v2": "qwen_mpi_network_results_v2",
    "qwen_nccl_network_results_v2": "qwen_nccl_network_results_v2",
}
RELATION_FIELDS = {name: spec.fields for name, spec in RELATION_REGISTRY.items()}
RELATION_GRAIN = {name: spec.grain for name, spec in RELATION_REGISTRY.items()}


@dataclass(frozen=True, slots=True)
class ProjectionRow:
    relation_id: str
    run_id: str
    source_member: str
    source_sha256: str
    columns: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        spec = _relation(self.relation_id)
        _run_id(self.run_id)
        _safe_member(self.source_member)
        _sha256(self.source_sha256, "source_sha256")
        names = tuple(name for name, _ in self.columns)
        if names != tuple(sorted(names, key=lambda item: item.encode())):
            raise ProjectionError("projection columns are not byte sorted")
        if names != spec.fields:
            raise ProjectionError(f"{self.relation_id} columns are not closed")
        for name, value in self.columns:
            _json_value(value, name)

    @classmethod
    def from_mapping(
        cls,
        relation_id: str,
        run_id: str,
        source_member: str,
        source_sha256: str,
        columns: Mapping[str, object],
    ) -> ProjectionRow:
        spec = _relation(relation_id)
        values = dict(columns)
        if "schema_version" in values:
            raise ProjectionError("schema_version is relation metadata, not a column")
        expected = set(spec.fields)
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise ProjectionError(f"{relation_id} closed fields differ: missing={missing}, extra={extra}")
        return cls(relation_id, run_id, source_member, source_sha256, tuple(sorted(values.items(), key=lambda item: item[0].encode())))

    def as_dict(self) -> dict[str, object]:
        """Append-adapter row; relation and lineage remain explicit."""

        spec = _relation(self.relation_id)
        row: dict[str, object] = {
            "projection_relation": self.relation_id,
            "schema_version": spec.schema_version,
            "run": self.run_id,
            "run_id": self.run_id,
            "source_member": self.source_member,
            "source_sha256": self.source_sha256,
        }
        row.update({name: _json_value(value, name, normalise=True) for name, value in self.columns})
        return row

    def to_projection_object(self) -> dict[str, object]:
        spec = _relation(self.relation_id)
        row: dict[str, object] = {
            "schema_version": spec.schema_version,
            "run_id": self.run_id,
        }
        row.update({name: _json_value(value, name, normalise=True) for name, value in self.columns})
        if spec.lineage:
            row["source_member"] = self.source_member
            row["source_sha256"] = self.source_sha256
        return row

    def to_bigquery_row(self) -> dict[str, object]:
        return self.as_dict()

    def grain(self) -> tuple[object, ...]:
        spec = _relation(self.relation_id)
        row = self.as_dict()
        return tuple(row.get(name, self.run_id) for name in spec.grain)


@dataclass(frozen=True, slots=True)
class ProjectionConflict:
    relation_id: str
    run_id: str
    grain: tuple[object, ...]
    rows: tuple[ProjectionRow, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "projection_relation": self.relation_id,
            "run": self.run_id,
            "grain": list(self.grain),
            "rows": [row.as_dict() for row in self.rows],
        }


@dataclass(frozen=True, slots=True)
class ProjectionBatch:
    rows: tuple[ProjectionRow, ...]
    conflicts: tuple[ProjectionConflict, ...] = ()

    @property
    def collection_complete(self) -> bool:
        return not self.conflicts

    def require_conflict_free(self) -> tuple[ProjectionRow, ...]:
        if self.conflicts:
            raise ProjectionConflictError(self.conflicts)
        return self.rows


def deduplicate_rows(rows: Iterable[ProjectionRow]) -> ProjectionBatch:
    groups: dict[tuple[str, str, tuple[object, ...]], list[ProjectionRow]] = defaultdict(list)
    for row in rows:
        groups[(row.relation_id, row.run_id, row.grain())].append(row)
    unique: list[ProjectionRow] = []
    conflicts: list[ProjectionConflict] = []
    for key in sorted(groups, key=lambda item: jcs_bytes([item[0], item[1], list(item[2])])):
        by_bytes: dict[bytes, ProjectionRow] = {}
        for row in groups[key]:
            by_bytes.setdefault(jcs_bytes(row.as_dict()), row)
        ordered = tuple(by_bytes[identity] for identity in sorted(by_bytes))
        if len(ordered) == 1:
            unique.append(ordered[0])
        else:
            conflicts.append(ProjectionConflict(key[0], key[1], key[2], ordered))
    return ProjectionBatch(tuple(unique), tuple(conflicts))


dedupe_and_detect_conflicts = deduplicate_rows


def registry_specs() -> tuple[RelationSpec, ...]:
    """Return deterministic registry metadata for SQL/publish adapters."""

    return tuple(RELATION_REGISTRY[name] for name in sorted(RELATION_REGISTRY, key=lambda item: item.encode()))


def project_closed_record(
    runtime_dir: Path,
    relation_id: str,
    source_member: str,
    *,
    run_id: str,
    expected_schema_version: str | None = None,
    required_run_field: str = "run_id",
    transform: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
) -> ProjectionRow:
    """Project one already validated closed JSON source.

    ``transform`` may select/rename fields but must return exactly the relation
    data fields.  It cannot add lineage or silently drop unknown source keys.
    """

    path, member, digest = _source_file(runtime_dir, source_member)
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict):
        raise ProjectionError("source record is not an object")
    spec = _relation(relation_id)
    if expected_schema_version is not None and value.get("schema_version") != expected_schema_version:
        raise ProjectionError("source schema version differs")
    if required_run_field in value and value[required_run_field] != run_id:
        raise ProjectionError("source run identity differs")
    if transform is not None:
        expected_source = set(spec.fields) | {"schema_version", required_run_field}
        if set(value) != expected_source:
            raise ProjectionError("source record has unknown or missing fields")
    selected = transform(value) if transform else _select_fields(value, spec.fields)
    return ProjectionRow.from_mapping(relation_id, run_id, member, digest, selected)


def project_sif_provenance(runtime_dir: Path, run_id: str) -> ProjectionRow:
    path, member, digest = _source_file(runtime_dir, "source/sif-sidecar.json")
    value = validate_sif_sidecar(path, member_root=runtime_dir / "source")
    artifact = value["sif_artifact"]
    if not isinstance(artifact, dict):
        raise ProjectionError("SIF artifact record is not an object")
    sif_path = _member_path(runtime_dir, artifact["artifact_relative_path"])
    if _file_digest(sif_path) != artifact["sha256"] or sif_path.stat().st_size != artifact["size_bytes"]:
        raise ProjectionError("SIF artifact lineage does not match sidecar")
    commits = value["source_commits"]
    by_role = {row["commit_role"]: row for row in commits}
    def observation_member(observation: Mapping[str, object], name: str) -> tuple[str, str]:
        locator = observation[name]
        if not isinstance(locator, dict) or set(locator) != {"path", "size_bytes", "sha256"}:
            raise ProjectionError("SIF observation member is not closed")
        relative = locator["path"]
        if not isinstance(relative, str):
            raise ProjectionError("SIF observation member path is invalid")
        payload = _member_path(runtime_dir, f"source/{relative}").read_bytes()
        if len(payload) != locator["size_bytes"] or _file_digest(_member_path(runtime_dir, f"source/{relative}")) != locator["sha256"]:
            raise ProjectionError("SIF observation member lineage differs")
        return relative, base64.b64encode(payload).decode("ascii")

    projected = {
        "sif_artifact_sha256": artifact["sha256"],
        "source_commits": commits,
        "score_source_tree": value["score_source_tree"],
        "recipe": value["recipe"],
        "applied_patches": value["applied_patches"],
        "base_oci": {name: value["base_oci"][name] for name in ("reference", "manifest_digest", "config_digest", "layer_digests")},
        "lock_manifest": {name: value["lock_manifest"][name] for name in ("path", "sha256")},
        "build_command": value["build_command"],
        "toolchain_observations": [
            {
                "role": item["role"], "outcome": item["outcome"],
                "stdout_member": observation_member(item, "stdout")[0], "stdout_b64": observation_member(item, "stdout")[1],
                "stderr_member": observation_member(item, "stderr")[0], "stderr_b64": observation_member(item, "stderr")[1],
            }
            for item in value["toolchain_commands"]
        ],
        "embedded_manifest_identity": value["embedded_manifest"]["manifest_identity"],
        "runtime_files": value["runtime_files"],
    }
    if not by_role:
        raise ProjectionError("SIF source commits are empty")
    return ProjectionRow.from_mapping("qwen_sif_provenance_v2", run_id, member, digest, projected)


def project_model_manifest(runtime_dir: Path, run_id: str) -> ProjectionRow:
    path, member, digest = _source_file(runtime_dir, "source/model-manifest.json")
    value = validate_model_manifest(path)
    source = value["source"]
    return ProjectionRow.from_mapping(
        "qwen_model_manifests_v2", run_id, member, digest,
        {
            "source_kind": source["kind"],
            "repository": source.get("repository"),
            "revision": source.get("revision"),
            "prebuilt_shards": source.get("shards"),
            "small_metadata": value["small_metadata"],
            "manifest_sha256": digest,
        },
    )


def project_dataset_manifest(runtime_dir: Path, run_id: str) -> ProjectionRow:
    path, member, digest = _source_file(runtime_dir, "source/dataset-manifest.json")
    value = validate_dataset_manifest(path)
    return ProjectionRow.from_mapping(
        "qwen_dataset_manifests_v2", run_id, member, digest,
        {"repository": value["repository"], "full_revision": value["full_revision"],
         "relative_path": value["relative_path"], "size_bytes": value["size_bytes"],
         "content_sha256": value["sha256"], "manifest_sha256": digest},
    )


def project_cache_manifests(runtime_dir: Path, run_id: str) -> tuple[ProjectionRow, ...]:
    rows: list[ProjectionRow] = []
    for path in sorted((runtime_dir / "source/cache").glob("node-*-input-manifest.json")):
        match = re.fullmatch(r"node-([0-9]+)-input-manifest\.json", path.name)
        if match is None or str(int(match.group(1))) != match.group(1):
            raise ProjectionError("cache manifest filename is invalid")
        node_rank = int(match.group(1))
        _, member, digest = _source_file(runtime_dir, path.relative_to(runtime_dir).as_posix())
        value = validate_cache_manifest(path, run_id, node_rank)
        semantic = {"schema_version": "cache-members/v1", "members": value["members"]}
        rows.append(ProjectionRow.from_mapping(
            "qwen_cache_manifests_v2", run_id, member, digest,
            {"node_rank": node_rank, "semantic_manifest_sha256": _sha256_bytes(semantic), "members": value["members"]},
        ))
    return tuple(rows)


def project_runtime_launchers(runtime_dir: Path, run_id: str) -> tuple[ProjectionRow, ...]:
    path, member, digest = _source_file(runtime_dir, "source/runtime-launchers.json")
    value = validate_runtime_launchers(path, runtime_dir)
    return (ProjectionRow.from_mapping(
        "qwen_runtime_launchers_v2", run_id, member, digest,
        {"executables": value["executables"], "versions": value["versions"]},
    ),)


def project_source_track(runtime_dir: Path, run_id: str) -> ProjectionRow:
    """Project the run-owned source-track identity once."""

    path, member, digest = _source_file(runtime_dir, "control/run-manifest.json")
    manifest = load_manifest(path)
    if manifest.run_id != run_id:
        raise ProjectionError("source-track run identity differs")
    source_track = manifest.source_track.to_object()
    return ProjectionRow.from_mapping(
        "qwen_source_track_v2", run_id, member, digest,
        {"run_purpose": manifest.run_purpose, "repository_id": manifest.repository_id,
         "source_track": source_track,
         "source_track_sha256": _sha256_bytes(source_track)},
    )


def project_minimal_hardware(runtime_dir: Path, run_id: str, node_rank: int) -> ProjectionRow:
    relative = f"source/hardware/node-{node_rank}.json"
    path, member, digest = _source_file(runtime_dir, relative)
    try:
        validate_minimal_hardware(path, runtime_dir, node_rank)
    except (TypeError, ValueError) as error:
        raise ProjectionError("minimal hardware validation failed") from error
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict):
        raise ProjectionError("minimal hardware is not an object")
    lscpu = value["lscpu_json"]
    labels: dict[str, object] = {}
    if isinstance(lscpu, dict):
        rows = lscpu.get("lscpu", [])
        if isinstance(rows, list):
            for item in rows:
                if isinstance(item, dict) and isinstance(item.get("field"), str):
                    labels[item["field"].rstrip(":").strip()] = item.get("data")
    identity = {
        "architecture": labels.get("Architecture"), "model_name": labels.get("Model name"),
        "vendor_id": labels.get("Vendor ID"), "cpu_family": labels.get("CPU family"),
        "model": labels.get("Model"), "stepping": labels.get("Stepping"),
        "physical_cores": None, "numa_nodes": None,
    }
    return ProjectionRow.from_mapping(
        "qwen_minimal_hardware_v2", run_id, member, digest,
        {"node_rank": node_rank, "hostname": value["hostname"], "cpu_identity": identity,
         "platform_settings": value["platform_settings"]},
    )


def project_node_launch(runtime_dir: Path, run_id: str, node_rank: int) -> ProjectionRow:
    relative = f"source/topology/node-{node_rank}.json"
    path, member, digest = _source_file(runtime_dir, relative)
    try:
        validate_node_topology(path, runtime_dir, run_id, node_rank)
    except (TypeError, ValueError) as error:
        raise ProjectionError("node topology validation failed") from error
    value = load_json_strict(path.read_bytes())
    outer = value["outer_mpi"]
    return ProjectionRow.from_mapping(
        "qwen_node_launch_v2", run_id, member, digest,
        {"node_rank": node_rank, "global_rank": outer["global_rank"],
         "local_rank": outer["local_rank"], "size": outer["size"],
         "hostname": outer["hostname"], "node_wrapper_argv": value["node_wrapper_argv"],
         "singularity_argv": value["singularity_argv"], "sglang_argv": value["sglang_argv"],
         "cpu_affinity": value["cpu_affinity"], "numa_mems": value["numa_mems"]},
    )


def project_topology(runtime_dir: Path, run_id: str, node_rank: int) -> tuple[ProjectionRow, ...]:
    """Project parsed GPU identity from both host/container command records."""

    relative = f"source/topology/node-{node_rank}.json"
    path, member, digest = _source_file(runtime_dir, relative)
    try:
        validate_node_topology(path, runtime_dir, run_id, node_rank)
    except (TypeError, ValueError) as error:
        raise ProjectionError("node topology validation failed") from error
    value = load_json_strict(path.read_bytes())
    rows: list[ProjectionRow] = []
    for scope in ("host", "container"):
        command = value["gpu_commands"][scope]
        payload = _member_path(runtime_dir, command["stdout_member"]).read_text(encoding="utf-8")
        for fields in _csv_rows(payload, 13):
            try:
                index = int(fields[0])
                memory = int(Decimal(fields[4]) * 1024 * 1024)
            except (ValueError, ArithmeticError) as error:
                raise ProjectionError("topology GPU CSV is invalid") from error
            rows.append(ProjectionRow.from_mapping(
                "qwen_topology_raw_v2", run_id, member, digest,
                {"node_rank": node_rank, "scope": scope, "index": index, "name": fields[1],
                 "uuid": fields[2], "pci_bus_id": fields[3], "memory_total_bytes": memory,
                 "driver_version": fields[5], "persistence_mode": fields[6], "compute_mode": fields[7],
                 "power_limit_w": _nullable_decimal(fields[8]), "max_sm_clock_mhz": _nullable_decimal(fields[9]),
                 "max_memory_clock_mhz": _nullable_decimal(fields[10]), "mig_mode": fields[11], "ecc_mode": fields[12]},
            ))
    return tuple(rows)


def project_lifecycle(runtime_dir: Path, run_id: str, expected_nodes: int) -> tuple[ProjectionRow, ...]:
    relative = "control/lifecycle/summary.json"
    path, member, digest = _source_file(runtime_dir, relative)
    try:
        validate_lifecycle_summary(path, run_id, expected_nodes)
    except (TypeError, ValueError) as error:
        raise ProjectionError("lifecycle summary validation failed") from error
    value = load_json_strict(path.read_bytes())
    return (ProjectionRow.from_mapping(
        "qwen_lifecycle_v2", run_id, member, digest,
        {"kind": "summary", "summary": value},
    ),)


def project_node_lifecycle(runtime_dir: Path, run_id: str, node_rank: int) -> ProjectionRow:
    relative = f"source/lifecycle/node-{node_rank}.json"
    path, member, digest = _source_file(runtime_dir, relative)
    try:
        validate_node_lifecycle(path, node_rank=node_rank)
    except (TypeError, ValueError) as error:
        raise ProjectionError("node lifecycle validation failed") from error
    value = load_json_strict(path.read_bytes())
    return ProjectionRow.from_mapping(
        "qwen_node_lifecycle_v2", run_id, member, digest,
        {"node_rank": node_rank, "lifecycle": value},
    )


def project_workload(runtime_dir: Path, run_id: str, kind: str) -> tuple[ProjectionRow, ...]:
    if kind not in {"warmup", "measured"}:
        raise ProjectionError("workload kind is invalid")
    path = runtime_dir / f"source/workload-{kind}.json"
    if not path.is_file():
        return ()
    _, member, digest = _source_file(runtime_dir, path.relative_to(runtime_dir).as_posix())
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict):
        raise ProjectionError("workload is not an object")
    try:
        validate_ordered_workload(path, kind=kind, dataset_manifest_sha256=value["dataset_manifest_sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectionError("workload validation failed") from error
    manifest = ProjectionRow.from_mapping(
        "qwen_workload_manifests_raw_v2", run_id, member, digest,
        {"workload_kind": kind, "dataset_manifest_sha256": value["dataset_manifest_sha256"],
         "warmup_mode": value.get("mode")},
    )
    request_rows = tuple(
        ProjectionRow.from_mapping(
            "qwen_workload_requests_raw_v2", run_id, member, digest,
            {"workload_kind": kind, "ordinal": ordinal, "request": request},
        )
        for ordinal, request in enumerate(value["requests"])
    )
    return (manifest, *request_rows)


def project_environment(runtime_dir: Path, run_id: str, scope: str) -> tuple[ProjectionRow, ...]:
    path = runtime_dir / f"source/environment/{scope}.snapshot"
    _, member, digest = _source_file(runtime_dir, path.relative_to(runtime_dir).as_posix())
    rows = []
    for name, raw in validate_environment_snapshot(path):
        decoded_name = name.decode("utf-8", errors="strict")
        rows.append(ProjectionRow.from_mapping(
            "qwen_environment_v2", run_id, member, digest,
            {"scope": scope, "name": decoded_name, "value_base64": base64.b64encode(raw).decode("ascii")},
        ))
    return tuple(rows)


def project_primary_result(runtime_dir: Path, run_id: str) -> ProjectionRow:
    return project_closed_record(
        runtime_dir, "qwen_primary_results_v2", "source/primary-result.json",
        run_id=run_id, expected_schema_version="primary-result/v1",
        transform=lambda value: {name: value[name] for name in RELATION_FIELDS["qwen_primary_results_v2"]},
    )


def project_artifact_manifest(runtime_dir: Path, run_id: str) -> tuple[ProjectionRow, ...]:
    path, member, digest = _source_file(runtime_dir, "artifact-manifest.json")
    value = _validate_artifact_manifest(path, runtime_dir, run_id)
    return tuple(ProjectionRow.from_mapping(
        "qwen_artifacts_v2", run_id, member, digest,
        {"role": item["role"], "path": item["path"], "size_bytes": item["size_bytes"],
         "sha256": item["sha256"], "run_purpose": value["run_purpose"],
         "logical_artifact_name": value["logical_artifact_name"]},
    ) for item in value["files"])


def project_imb_results(runtime_dir: Path, run_id: str, *, source_member: str = "source/network/imb-mpi1/stdout.txt") -> tuple[ProjectionRow, ...]:
    path, member, digest = _source_file(runtime_dir, source_member)
    results = project_mpi_network_results_v2(run_id, parse_imb_mpi1(path.read_text(encoding="utf-8")), member)
    return tuple(_mpi_row(row, digest) for row in results)


def project_nccl_result(runtime_dir: Path, run_id: str, collective: str) -> tuple[ProjectionRow, ...]:
    source_member = f"source/network/nccl-tests/{collective}/result.json"
    path, member, digest = _source_file(runtime_dir, source_member)
    results = parse_nccl_tests_json(path.read_bytes(), run_id=run_id, collective=collective, source_member=member)
    return tuple(_nccl_row(row, digest) for row in results)


def project_gpu_samples(runtime_dir: Path, run_id: str, node_rank: int, expected_gpus: Mapping[str, int]) -> tuple[ProjectionRow, ...]:
    relative = f"profile/raw/gpu/node-{node_rank}.jsonl"
    path, member, digest = _source_file(runtime_dir, relative)
    samples = parse_gpu_samples_jsonl(path.read_bytes(), run_id=run_id, node_rank=node_rank,
                                      clock_id=_first_clock(path), expected_gpus=expected_gpus)
    return tuple(ProjectionRow.from_mapping(
        "qwen_gpu_samples_raw_v2", run_id, member, digest,
        {name: sample.to_object()[name] for name in RELATION_FIELDS["qwen_gpu_samples_raw_v2"]},
    ) for sample in samples)


def project_gpu_phases(runtime_dir: Path, run_id: str, phase_path: Path, expected_gpus: Mapping[int, Mapping[str, int]]) -> tuple[ProjectionRow, ...]:
    from .profile_contract import parse_phase_summary

    phase = parse_phase_summary(phase_path, run_id=run_id, expected_nodes=len(expected_gpus))
    try:
        values = project_gpu_phase(runtime_dir, run_id=run_id, phase=phase, expected_gpus=expected_gpus)
    except ProfileValidationError as error:
        raise ProjectionError("GPU phase projection failed") from error
    rows = []
    for value in values:
        data = value.phase.to_object()
        rows.append(ProjectionRow.from_mapping("qwen_gpu_phases_v2", run_id,
            value.source_member, value.source_sha256,
            {name: data[name] for name in RELATION_FIELDS["qwen_gpu_phases_v2"]}))
    return tuple(rows)


def project_profiler_events_raw(runtime_dir: Path, run_id: str, invocation_id: str, expected_ranks: int) -> ProjectionBatch:
    try:
        projection = project_profiler_events(runtime_dir, run_id=run_id, invocation_id=invocation_id, expected_ranks=expected_ranks)
    except ProfileValidationError as error:
        raise ProjectionError("profiler projection failed") from error
    rows = tuple(ProjectionRow.from_mapping(
        "qwen_profiler_events_raw_v2", run_id, value.source_member, value.source_sha256,
        {name: value.as_dict()[name] for name in RELATION_FIELDS["qwen_profiler_events_raw_v2"]},
    ) for value in projection.profiler_rows)
    return deduplicate_rows(rows)


def project_profile_hardware(runtime_dir: Path, run_id: str, node_rank: int) -> tuple[ProjectionRow, ...]:
    """Project profile-only /proc and NUMA captures without placeholders."""

    root = runtime_dir / f"profile/raw/hardware/node-{node_rank}"
    capture_path = root / "capture.json"
    capture_member = capture_path.relative_to(runtime_dir).as_posix()
    _, _, capture_digest = _source_file(runtime_dir, capture_member)
    capture = load_json_strict(capture_path.read_bytes())
    expected = {"schema_version", "run_id", "node_rank", "hostname", "members"}
    if not isinstance(capture, dict) or set(capture) != expected or capture["schema_version"] != "profile-hardware-capture/v1" or capture["run_id"] != run_id or capture["node_rank"] != node_rank:
        raise ProjectionError("profile hardware capture is not closed")
    if not isinstance(capture["hostname"], str) or not capture["hostname"]:
        raise ProjectionError("profile hardware hostname is invalid")
    members = capture["members"]
    if not isinstance(members, list):
        raise ProjectionError("profile hardware members are not an array")
    bound: dict[tuple[str, int | None], tuple[str, Path, str]] = {}
    for item in members:
        if not isinstance(item, dict) or set(item) != {"role", "numa_node_id", "path", "size_bytes", "sha256"}:
            raise ProjectionError("profile hardware member is not closed")
        role = item["role"]
        numa_id = item["numa_node_id"]
        if role not in {"proc_meminfo", "numa_cpulist", "numa_meminfo"}:
            raise ProjectionError("profile hardware member role is invalid")
        if role == "proc_meminfo" and numa_id is not None:
            raise ProjectionError("proc_meminfo NUMA identity differs")
        if role != "proc_meminfo" and (isinstance(numa_id, bool) or not isinstance(numa_id, int) or numa_id < 0):
            raise ProjectionError("NUMA member identity is invalid")
        path, member, digest = _source_file(runtime_dir, item["path"])
        if path.stat().st_size != item["size_bytes"] or digest != item["sha256"]:
            raise ProjectionError("profile hardware member lineage differs")
        key = (role, numa_id)
        if key in bound:
            raise ProjectionError("duplicate profile hardware member")
        bound[key] = (member, path, digest)
    role_order = {"proc_meminfo": 0, "numa_cpulist": 1, "numa_meminfo": 2}
    member_order = [(role, -1 if node is None else node, value[0]) for (role, node), value in bound.items()]
    if member_order != sorted(member_order, key=lambda item: (role_order[item[0]], item[1], item[2].encode())):
        raise ProjectionError("profile hardware members are not role/node/path sorted")
    if ("proc_meminfo", None) not in bound:
        raise ProjectionError("profile hardware proc_meminfo member is missing")
    capture_ref = {"role": "capture", "path": capture_member, "sha256": capture_digest}
    rows: list[ProjectionRow] = []
    proc = bound.get(("proc_meminfo", None))
    if proc is not None:
        available = _meminfo_value(proc[1], "MemAvailable")
        source_members = [capture_ref, {"role": "proc_meminfo", "path": proc[0], "sha256": proc[2]}]
        rows.append(ProjectionRow.from_mapping(
            "qwen_profile_hardware_raw_v2", run_id, capture_member, capture_digest,
            {"node_rank": node_rank, "hostname": capture["hostname"], "record_kind": "node_summary",
             "numa_node_id": None, "postrun_memory_available_bytes": available,
             "cpulist": None, "mem_total_bytes": None, "mem_free_bytes": None,
             "source_members": source_members},
        ))
    numa_ids = sorted({numa_id for role, numa_id in bound if role != "proc_meminfo" and numa_id is not None})
    for numa_id in numa_ids:
        cpulist = bound.get(("numa_cpulist", numa_id))
        meminfo = bound.get(("numa_meminfo", numa_id))
        if cpulist is None or meminfo is None:
            raise ProjectionError("NUMA capture pair is incomplete")
        source_members = [capture_ref,
            {"role": "numa_cpulist", "path": cpulist[0], "sha256": cpulist[2]},
            {"role": "numa_meminfo", "path": meminfo[0], "sha256": meminfo[2]}]
        total = _meminfo_value(meminfo[1], "MemTotal", node_id=numa_id)
        free = _meminfo_value(meminfo[1], "MemFree", node_id=numa_id)
        rows.append(ProjectionRow.from_mapping(
            "qwen_profile_hardware_raw_v2", run_id, capture_member, capture_digest,
            {"node_rank": node_rank, "hostname": capture["hostname"], "record_kind": "numa_node",
             "numa_node_id": numa_id, "postrun_memory_available_bytes": None,
             "cpulist": _canonical_cpulist(cpulist[1].read_text(encoding="ascii")),
             "mem_total_bytes": total, "mem_free_bytes": free, "source_members": source_members},
        ))
    return tuple(rows)


def project_validated_sources(runtime_dir: Path, run_id: str) -> ProjectionBatch:
    """Project all present validated source families without placeholders."""

    manifest = load_manifest(runtime_dir / "control/run-manifest.json")
    if manifest.run_id != run_id:
        raise ProjectionError("run manifest identity differs")
    rows: list[ProjectionRow] = [project_source_track(runtime_dir, run_id)]
    optional: list[tuple[str, object]] = [
        ("source/sif-sidecar.json", project_sif_provenance),
        ("source/model-manifest.json", project_model_manifest),
        ("source/dataset-manifest.json", project_dataset_manifest),
        ("source/runtime-launchers.json", project_runtime_launchers),
        ("artifact-manifest.json", project_artifact_manifest),
    ]
    for relative, function in optional:
        if (runtime_dir / relative).is_file():
            result = function(runtime_dir, run_id)
            rows.extend(result if isinstance(result, tuple) else (result,))
    rows.extend(project_cache_manifests(runtime_dir, run_id))
    for kind in ("warmup", "measured"):
        rows.extend(project_workload(runtime_dir, run_id, kind))
    primary = runtime_dir / "source/primary-result.json"
    if primary.is_file():
        rows.append(project_primary_result(runtime_dir, run_id))
    for path in sorted((runtime_dir / "source/hardware").glob("node-*.json")):
        match = re.fullmatch(r"node-([0-9]+)\.json", path.name)
        if match:
            rows.append(project_minimal_hardware(runtime_dir, run_id, int(match.group(1))))
    for path in sorted((runtime_dir / "source/topology").glob("node-*.json")):
        match = re.fullmatch(r"node-([0-9]+)\.json", path.name)
        if match:
            node_rank = int(match.group(1))
            rows.append(project_node_launch(runtime_dir, run_id, node_rank))
            rows.extend(project_topology(runtime_dir, run_id, node_rank))
    lifecycle = runtime_dir / "control/lifecycle/summary.json"
    if lifecycle.is_file():
        summary = load_json_strict(lifecycle.read_bytes())
        if isinstance(summary, dict) and isinstance(summary.get("nodes"), list):
            rows.append(project_lifecycle(runtime_dir, run_id, len(summary["nodes"])))
    for path in sorted((runtime_dir / "source/lifecycle").glob("node-*.json")):
        match = re.fullmatch(r"node-([0-9]+)\.json", path.name)
        if match:
            rows.append(project_node_lifecycle(runtime_dir, run_id, int(match.group(1))))
    for path in sorted((runtime_dir / "profile/raw/hardware").glob("node-*/capture.json")):
        match = re.fullmatch(r"node-([0-9]+)", path.parent.name)
        if match:
            rows.extend(project_profile_hardware(runtime_dir, run_id, int(match.group(1))))
    for scope_path in sorted((runtime_dir / "source/environment").glob("*.snapshot")):
        rows.extend(project_environment(runtime_dir, run_id, scope_path.stem))
    imb = runtime_dir / "source/network/imb-mpi1/stdout.txt"
    if imb.is_file():
        rows.extend(project_imb_results(runtime_dir, run_id))
    for collective in ("all_reduce", "all_gather", "reduce_scatter"):
        if (runtime_dir / f"source/network/nccl-tests/{collective}/result.json").is_file():
            rows.extend(project_nccl_result(runtime_dir, run_id, collective))
    return deduplicate_rows(rows)


def _mpi_row(row: MpiNetworkResultV2, digest: str) -> ProjectionRow:
    values = row.to_object()
    return ProjectionRow.from_mapping("qwen_mpi_network_results_v2", row.run_id, row.source_member, digest,
        {name: values[name] for name in RELATION_FIELDS["qwen_mpi_network_results_v2"]})


def _nccl_row(row: NcclNetworkResultV2, digest: str) -> ProjectionRow:
    values = row.to_object()
    return ProjectionRow.from_mapping("qwen_nccl_network_results_v2", row.run_id, row.source_member, digest,
        {name: values[name] for name in RELATION_FIELDS["qwen_nccl_network_results_v2"]})


def _validate_artifact_manifest(path: Path, runtime_dir: Path, run_id: str) -> dict[str, object]:
    value = load_json_strict(path.read_bytes())
    expected = {"schema_version", "run_id", "logical_artifact_name", "run_purpose", "files"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ProjectionError("artifact manifest is not closed")
    if value["schema_version"] != "artifact-manifest/v1" or value["run_id"] != run_id:
        raise ProjectionError("artifact manifest identity differs")
    roles = {"bench": ("run_bundle", "metrics"), "profile": ("run_bundle", "metrics", "profile_raw")}
    if value["run_purpose"] not in roles or not isinstance(value["files"], list):
        raise ProjectionError("artifact manifest purpose/files differ")
    files = value["files"]
    if [item.get("role") for item in files] != list(roles[value["run_purpose"]]):
        raise ProjectionError("artifact manifest role order differs")
    for item in files:
        if not isinstance(item, dict) or set(item) != {"role", "path", "size_bytes", "sha256"}:
            raise ProjectionError("artifact member is not closed")
        if item["role"] not in ROLE_FILENAMES or item["path"] != ROLE_FILENAMES[item["role"]]:
            raise ProjectionError("artifact role/path differs")
        member = _member_path(runtime_dir, item["path"])
        if member.stat().st_size != item["size_bytes"] or _file_digest(member) != item["sha256"]:
            raise ProjectionError("artifact member lineage differs")
    return value


def _relation(relation_id: str) -> RelationSpec:
    try:
        return RELATION_REGISTRY[RELATION_ALIASES.get(relation_id, relation_id)]
    except KeyError as error:
        raise ProjectionError(f"unknown relation: {relation_id!r}") from error


def _select_fields(value: Mapping[str, object], fields: Sequence[str]) -> dict[str, object]:
    # Generic source projection is intentionally strict: source schemas may
    # have metadata not belonging to the target relation, but a caller must
    # explicitly transform it rather than silently leaking/losing fields.
    return {name: value[name] for name in fields}


def _source_file(runtime_dir: Path, relative: str) -> tuple[Path, str, str]:
    path = _member_path(runtime_dir, relative)
    return path, relative, _file_digest(path)


def _member_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ProjectionError("member path must be a string")
    _safe_member(relative)
    root_resolved = root.resolve()
    candidate = (root_resolved / Path(*PurePosixPath(relative).parts))
    if candidate.is_symlink():
        raise ProjectionError("member must not be a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root_resolved):
        raise ProjectionError("member is not a regular file under runtime root")
    return resolved


def _safe_member(relative: str) -> None:
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or any(not part for part in path.parts) or str(path) != relative:
        raise ProjectionError(f"unsafe member path: {relative!r}")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProjectionError(f"{label} must be lowercase SHA-256")
    return value


def _sha256_bytes(value: object) -> str:
    return hashlib.sha256(jcs_bytes(value)).hexdigest()


def _run_id(value: object) -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise ProjectionError("run ID is not canonical")
    return value


def _first_clock(path: Path) -> str:
    first = path.read_bytes().splitlines()[0]
    value = load_json_strict(first)
    if not isinstance(value, dict) or not isinstance(value.get("clock_id"), str):
        raise ProjectionError("GPU source has no clock ID")
    return value["clock_id"]


def _csv_rows(payload: str, width: int) -> tuple[tuple[str, ...], ...]:
    rows = tuple(tuple(item.strip() for item in row) for row in csv.reader(payload.splitlines()))
    if any(len(row) != width for row in rows):
        raise ProjectionError("CSV column count differs")
    return rows


def _nullable_decimal(value: str) -> Decimal | None:
    if value in {"", "N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
        return None
    try:
        decimal = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ProjectionError("numeric CSV value is invalid") from error
    if not decimal.is_finite() or decimal < 0:
        raise ProjectionError("numeric CSV value is invalid")
    return decimal


def _meminfo_value(path: Path, key: str, *, node_id: int | None = None) -> int | None:
    pattern = re.compile(
        rf"^Node {node_id} {re.escape(key)}:[ \t]+([0-9]+)[ \t]+kB$"
        if node_id is not None
        else rf"^{re.escape(key)}:[ \t]+([0-9]+)[ \t]+kB$"
    )
    found: int | None = None
    for line in path.read_text(encoding="ascii").splitlines():
        match = pattern.fullmatch(line)
        if match:
            if found is not None:
                raise ProjectionError(f"duplicate {key} in meminfo")
            found = int(match.group(1)) * 1024
    return found


def _canonical_cpulist(text: str) -> str:
    values: set[int] = set()
    stripped = text.strip()
    if not stripped:
        return ""
    for token in stripped.split(","):
        if re.fullmatch(r"[0-9]+", token):
            values.add(int(token))
        elif re.fullmatch(r"[0-9]+-[0-9]+", token):
            start, end = (int(item) for item in token.split("-"))
            if end < start:
                raise ProjectionError("CPU list range is reversed")
            values.update(range(start, end + 1))
        else:
            raise ProjectionError("CPU list grammar is invalid")
    if not values:
        return ""
    ordered = sorted(values)
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _json_value(value: object, label: str, *, normalise: bool = False) -> object:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ProjectionError(f"{label} must be finite")
        return str(value) if normalise else value
    if isinstance(value, float) and not math.isfinite(value):
        raise ProjectionError(f"{label} must be finite")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list | tuple):
        return [_json_value(item, label, normalise=normalise) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ProjectionError(f"{label} has non-string object key")
        return {key: _json_value(child, label, normalise=normalise) for key, child in value.items()}
    raise ProjectionError(f"{label} is not JSON-compatible")
