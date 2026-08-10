from __future__ import annotations

from pathlib import Path

import pytest
from conftest import RUN_ID, make_manifest

from qwen_experiment_metrics.canonical import jcs_bytes
from qwen_experiment_metrics.projection_router import project_run_archive
from qwen_experiment_metrics.projections import ProjectionBatch, ProjectionRow


def _configuration(node: int, sequence: int, event_type: str) -> dict[str, object]:
    config_type = "BenchArgs" if event_type == "bench_args_effective" else "ServerArgs"
    return {
        "schema_version": "qwen-domain-event/v1",
        "run_id": RUN_ID,
        "producer_id": f"sglang-node-{node}",
        "node_rank": node,
        "sequence": sequence,
        "event_type": event_type,
        "payload": {
            "schema_version": "config-snapshot/v1",
            "config_type": config_type,
            "definition": {
                "source_path": "python/source.py",
                "class_name": config_type,
                "source_blob_oid": "a" * 40,
            },
            "fields": [
                {"name": "node_rank", "declared_type": "int", "value": node}
            ],
        },
    }


def _worker(node: int, rank: int) -> dict[str, object]:
    return {
        "schema_version": "qwen-domain-event/v1",
        "run_id": RUN_ID,
        "producer_id": f"sglang-worker-{rank}",
        "node_rank": node,
        "global_rank": rank,
        "sequence": 0,
        "event_type": "worker_ready",
        "payload": {
            "tp_rank": rank,
            "gpu_id": rank % 4,
            "gpu_uuid": f"GPU-{rank}",
            "cpu_affinity": [rank],
            "numa_mems": str(node),
            "prefill_attention_backend": "flashinfer",
            "decode_attention_backend": "flashinfer",
            "torch_distributed_backend": "nccl",
            "transport_backend": "nccl",
        },
    }


def _write_event_archives(runtime: Path) -> None:
    event_dir = runtime / "source/events"
    event_dir.mkdir(parents=True)
    for node in range(2):
        values = [
            _configuration(node, 0, "bench_args_effective"),
            _configuration(node, 1, "server_args_effective"),
            *(_worker(node, rank) for rank in range(node * 4, node * 4 + 4)),
        ]
        (event_dir / f"node-{node}.jsonl").write_bytes(
            b"".join(jcs_bytes(value) + b"\n" for value in values)
        )


def test_run_archive_routes_config_and_placement_from_closed_events(
    manifest_path: Path,
    runtime: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del manifest_path
    _write_event_archives(runtime)

    monkeypatch.setattr(
        "qwen_experiment_metrics.projection_router.project_validated_sources",
        lambda *_: ProjectionBatch(()),
    )
    monkeypatch.setattr(
        "qwen_experiment_metrics.projection_router.topology_gpu_bindings",
        lambda _path, _root: {f"GPU-{index}": index for index in range(4)},
    )

    def topology(_root: Path, _run: str, node: int) -> tuple[ProjectionRow, ...]:
        return (
            ProjectionRow.from_mapping(
                "qwen_topology_raw_v2",
                RUN_ID,
                f"source/topology/node-{node}.json",
                "b" * 64,
                {
                    "node_rank": node,
                    "scope": "container",
                    "index": 0,
                    "name": "GPU",
                    "uuid": f"GPU-{node * 4}",
                    "pci_bus_id": "0000:01:00.0",
                    "memory_total_bytes": 1,
                    "driver_version": "1",
                    "persistence_mode": "Enabled",
                    "compute_mode": "Default",
                    "power_limit_w": "1",
                    "max_sm_clock_mhz": "1",
                    "max_memory_clock_mhz": "1",
                    "mig_mode": "Disabled",
                    "ecc_mode": "Enabled",
                },
            ),
        )

    monkeypatch.setattr(
        "qwen_experiment_metrics.projection_router.project_topology", topology
    )
    batch = project_run_archive(runtime, RUN_ID)
    rows = batch.require_conflict_free()
    assert sum(row.relation_id == "qwen_config_snapshots_v2" for row in rows) == 2
    placements = [row for row in rows if row.relation_id == "qwen_placement_v2"]
    assert len(placements) == 8
    assert {dict(row.columns)["global_rank"] for row in placements} == set(range(8))
    assert all(row.source_member.startswith("source/events/node-") for row in rows)


def test_profile_archive_router_invokes_gpu_phase_and_all_rank_profiler(
    runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_manifest(runtime, purpose="profile")
    _write_event_archives(runtime)
    phase = runtime / "control/phase/summary.json"
    phase.parent.mkdir(parents=True, exist_ok=True)
    phase.write_bytes(jcs_bytes({"invocation_id": "measured-generation"}) + b"\n")
    monkeypatch.setattr(
        "qwen_experiment_metrics.projection_router.project_validated_sources",
        lambda *_: ProjectionBatch(()),
    )
    monkeypatch.setattr(
        "qwen_experiment_metrics.projection_router.topology_gpu_bindings",
        lambda _path, _root: {"GPU-0": 0},
    )
    monkeypatch.setattr(
        "qwen_experiment_metrics.projection_router.project_topology", lambda *_: ()
    )
    calls: list[object] = []
    monkeypatch.setattr(
        "qwen_experiment_metrics.projection_router.project_gpu_samples",
        lambda _root, _run, node, _gpus: calls.append(("samples", node)) or (),
    )
    monkeypatch.setattr(
        "qwen_experiment_metrics.projection_router.project_gpu_phases",
        lambda _root, _run, _phase, _gpus: calls.append("phases") or (),
    )
    monkeypatch.setattr(
        "qwen_experiment_metrics.projection_router.project_profiler_events_raw",
        lambda _root, _run, invocation, ranks: (
            calls.append(("profiler", invocation, ranks)) or ProjectionBatch(())
        ),
    )

    project_run_archive(runtime, RUN_ID).require_conflict_free()
    assert calls == [
        ("samples", 0),
        ("samples", 1),
        "phases",
        ("profiler", "measured-generation", 8),
    ]
