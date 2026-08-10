from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .canonical import load_json_strict
from .event_sources import ArchivedDomainEvent, validate_event_archive
from .manifest import load_manifest
from .projections import (
    ProjectionBatch,
    ProjectionError,
    deduplicate_rows,
    project_config_snapshot,
    project_gpu_phases,
    project_gpu_samples,
    project_placement,
    project_profiler_events_raw,
    project_topology,
    project_validated_sources,
)
from .sources import load_execution_settings
from .system_sources import topology_gpu_bindings


def project_run_archive(runtime_dir: Path, run_id: str) -> ProjectionBatch:
    """Route a complete run archive into all applicable closed v2 relations."""

    manifest = load_manifest(runtime_dir / "control/run-manifest.json")
    settings = load_execution_settings(runtime_dir / "control/execution-settings.json")
    if manifest.run_id != run_id or settings.run_purpose != manifest.run_purpose:
        raise ProjectionError("run archive identities differ")
    base = project_validated_sources(runtime_dir, run_id)
    rows = list(base.require_conflict_free())

    archives = {
        node: validate_event_archive(
            runtime_dir / f"source/events/node-{node}.jsonl",
            runtime_dir,
            run_id=run_id,
            node_rank=node,
        )
        for node in range(settings.nodes)
    }
    configurations = _configuration_events(archives)
    for event_type in ("bench_args_effective", "server_args_effective"):
        selected = configurations[0][event_type]
        rows.append(project_config_snapshot(
            run_id,
            selected.event.payload,
            source_member=selected.source_member,
            source_sha256=selected.source_sha256,
        ))

    workers = _worker_events(archives, settings.tensor_parallel_size)
    server_by_node = {
        node: values["server_args_effective"].event.payload
        for node, values in configurations.items()
    }
    topology_by_node: dict[int, list[dict[str, object]]] = {}
    expected_gpus: dict[int, dict[str, int]] = {}
    for node in range(settings.nodes):
        topology_path = runtime_dir / f"source/topology/node-{node}.json"
        expected_gpus[node] = topology_gpu_bindings(topology_path, runtime_dir)
        topology_by_node[node] = [
            row.to_projection_object()
            for row in project_topology(runtime_dir, run_id, node)
            if dict(row.columns)["scope"] == "container"
        ]
    for global_rank in range(settings.tensor_parallel_size):
        archived = workers[global_rank]
        node_rank = archived.event.node_rank
        rows.append(project_placement(
            run_id,
            node_rank=node_rank,
            global_rank=global_rank,
            server_args=server_by_node[node_rank],
            worker_ready=archived.event.payload,
            topology=topology_by_node[node_rank],
            source_member=archived.source_member,
            source_sha256=archived.source_sha256,
        ))

    if manifest.run_purpose == "profile":
        phase_path = runtime_dir / "control/phase/summary.json"
        phase = load_json_strict(phase_path.read_bytes())
        if not isinstance(phase, dict) or phase.get("invocation_id") != "measured-generation":
            raise ProjectionError("profile phase invocation identity differs")
        for node in range(settings.nodes):
            rows.extend(project_gpu_samples(
                runtime_dir, run_id, node, expected_gpus[node]
            ))
        rows.extend(project_gpu_phases(
            runtime_dir, run_id, phase_path, expected_gpus
        ))
        profiler = project_profiler_events_raw(
            runtime_dir,
            run_id,
            "measured-generation",
            settings.tensor_parallel_size,
        )
        rows.extend(profiler.require_conflict_free())
    return deduplicate_rows(rows)


def _configuration_events(
    archives: Mapping[int, tuple[ArchivedDomainEvent, ...]],
) -> dict[int, dict[str, ArchivedDomainEvent]]:
    result: dict[int, dict[str, ArchivedDomainEvent]] = {}
    for node_rank, events in archives.items():
        selected: dict[str, ArchivedDomainEvent] = {}
        for item in events:
            if item.event.event_type not in {
                "bench_args_effective",
                "server_args_effective",
            }:
                continue
            if item.event.event_type in selected:
                raise ProjectionError("node has duplicate configuration event")
            selected[item.event.event_type] = item
        if set(selected) != {"bench_args_effective", "server_args_effective"}:
            raise ProjectionError("node configuration event set is incomplete")
        result[node_rank] = selected
    # A relation has run/config_type grain. Node zero owns that archival row;
    # every node-specific ServerArgs remains available in placement rows.
    if 0 not in result:
        raise ProjectionError("node zero configuration archive is missing")
    return result


def _worker_events(
    archives: Mapping[int, tuple[ArchivedDomainEvent, ...]], expected_ranks: int
) -> dict[int, ArchivedDomainEvent]:
    workers: dict[int, ArchivedDomainEvent] = {}
    for events in archives.values():
        for item in events:
            if item.event.event_type != "worker_ready":
                continue
            rank = item.event.global_rank
            if rank is None or rank in workers:
                raise ProjectionError("worker-ready rank is absent or duplicated")
            workers[rank] = item
    if set(workers) != set(range(expected_ranks)):
        raise ProjectionError("worker-ready rank grid is incomplete")
    return workers
