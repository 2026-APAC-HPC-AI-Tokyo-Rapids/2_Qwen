from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qwen_experiment_metrics.canonical import jcs_bytes
from qwen_experiment_metrics.projections import (
    RELATION_REGISTRY,
    ProjectionConflictError,
    ProjectionRow,
    deduplicate_rows,
    project_artifact_manifest,
    project_cache_manifests,
    project_dataset_manifest,
    project_imb_results,
    project_model_manifest,
    project_nccl_result,
    registry_specs,
)

RUN_ID = "r-0123456789abcdef0123456789abcdef"
FIXTURES = Path(__file__).parent / "fixtures"


def test_registry_is_closed_sorted_and_declares_contract_applicability() -> None:
    expected = {
        "qwen_source_track_v2",
        "qwen_sif_provenance_v2",
        "qwen_runtime_launchers_v2",
        "qwen_model_manifests_v2",
        "qwen_dataset_manifests_v2",
        "qwen_cache_manifests_v2",
        "qwen_workload_manifests_raw_v2",
        "qwen_workload_requests_raw_v2",
        "qwen_minimal_hardware_v2",
        "qwen_node_launch_v2",
        "qwen_topology_raw_v2",
        "qwen_environment_v2",
        "qwen_lifecycle_v2",
        "qwen_node_lifecycle_v2",
        "qwen_primary_results_v2",
        "qwen_phase_summaries_v2",
        "qwen_gpu_samples_raw_v2",
        "qwen_gpu_phases_v2",
        "qwen_profile_hardware_raw_v2",
        "qwen_profiler_events_raw_v2",
        "qwen_config_snapshots_v2",
        "qwen_placement_v2",
        "qwen_artifacts_v2",
        "qwen_mpi_network_results_v2",
        "qwen_nccl_network_results_v2",
    }
    assert set(RELATION_REGISTRY) == expected
    specs = registry_specs()
    assert tuple(spec.relation_id for spec in specs) == tuple(
        sorted(expected, key=lambda item: item.encode())
    )
    for spec in specs:
        assert spec.schema_version.endswith("/v1")
        assert spec.fields == tuple(sorted(spec.fields, key=lambda item: item.encode()))
        assert spec.fields == tuple(dict.fromkeys(spec.fields))
        assert spec.applicability in (("bench", "profile"), ("profile",))
        assert spec.grain
        assert all(name == "run" or name in spec.fields for name in spec.grain)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jcs_bytes(value) + b"\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_projection_rows_dedupe_exact_values_and_report_conflicts() -> None:
    columns = {
        "benchmark": "PingPong",
        "process_count": 2,
        "message_bytes": 8,
        "iterations": 100,
        "time_min_us": None,
        "time_avg_us": None,
        "time_max_us": "1.0",
        "bandwidth_mbytes_per_sec": "2.0",
    }
    first = ProjectionRow.from_mapping(
        "qwen_mpi_network_results_v2", RUN_ID, "source/imb.txt", "a" * 64, columns
    )
    duplicate = ProjectionRow.from_mapping(
        "qwen_mpi_network_results_v2", RUN_ID, "source/imb.txt", "a" * 64, columns
    )
    batch = deduplicate_rows((first, duplicate))
    assert batch.conflicts == ()
    assert batch.rows == (first,)
    assert first.grain() == (RUN_ID, "PingPong", 2, 8)
    assert "run" not in first.to_bigquery_row()
    assert first.to_projection_object()["run_id"] == RUN_ID

    conflict = ProjectionRow.from_mapping(
        "qwen_mpi_network_results_v2",
        RUN_ID,
        "source/imb.txt",
        "a" * 64,
        {**columns, "bandwidth_mbytes_per_sec": "3.0"},
    )
    conflict_batch = deduplicate_rows((first, conflict))
    assert conflict_batch.rows == ()
    assert len(conflict_batch.conflicts) == 1
    with pytest.raises(ProjectionConflictError):
        conflict_batch.require_conflict_free()


def test_manifest_projections_preserve_source_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    model = {
        "schema_version": "model-manifest/v1",
        "source": {
            "kind": "hub_revision",
            "repository": "example/model",
            "revision": "a" * 40,
        },
        "small_metadata": [
            {
                "path": "config.json",
                "size_bytes": 2,
                "sha256": "b" * 64,
                "role": "configuration",
            },
            {
                "path": "model.safetensors.index.json",
                "size_bytes": 2,
                "sha256": "c" * 64,
                "role": "index",
            },
        ],
    }
    dataset = {
        "schema_version": "dataset-manifest/v1",
        "repository": "example/dataset",
        "full_revision": "d" * 40,
        "relative_path": "data.jsonl",
        "size_bytes": 12,
        "sha256": "e" * 64,
    }
    _write_json(source / "model-manifest.json", model)
    _write_json(source / "dataset-manifest.json", dataset)

    model_row = project_model_manifest(tmp_path, RUN_ID)
    dataset_row = project_dataset_manifest(tmp_path, RUN_ID)
    assert model_row.source_member == "source/model-manifest.json"
    assert model_row.source_sha256 == _sha(source / "model-manifest.json")
    assert dataset_row.source_sha256 == _sha(source / "dataset-manifest.json")
    assert model_row.to_bigquery_row()["repository"] == "example/model"
    assert dataset_row.to_bigquery_row()["content_sha256"] == "e" * 64


def test_cache_projection_uses_subject_free_semantic_identity(tmp_path: Path) -> None:
    path = tmp_path / "source/cache/node-3-input-manifest.json"
    _write_json(
        path,
        {
            "schema_version": "cache-manifest/v1",
            "run_id": RUN_ID,
            "node_rank": 3,
            "members": [],
        },
    )
    rows = project_cache_manifests(tmp_path, RUN_ID)
    assert len(rows) == 1
    assert rows[0].to_bigquery_row()["node_rank"] == 3
    assert len(rows[0].to_bigquery_row()["semantic_manifest_sha256"]) == 64


def test_network_projections_reuse_validated_parsers(tmp_path: Path) -> None:
    imb = tmp_path / "source/network/imb-mpi1/stdout.txt"
    imb.parent.mkdir(parents=True)
    imb.write_bytes((FIXTURES / "imb-mpi1-complete.txt").read_bytes())
    imb_rows = project_imb_results(tmp_path, RUN_ID)
    assert imb_rows
    assert {row.to_bigquery_row()["benchmark"] for row in imb_rows} == {
        "PingPong",
        "Sendrecv",
        "Allreduce",
    }
    assert all(row.source_sha256 == _sha(imb) for row in imb_rows)

    nccl = tmp_path / "source/network/nccl-tests/all_reduce/result.json"
    nccl.parent.mkdir(parents=True)
    nccl.write_bytes((FIXTURES / "nccl-all-reduce.json").read_bytes())
    nccl_rows = project_nccl_result(tmp_path, RUN_ID, "all_reduce")
    assert len(nccl_rows) == 4
    assert {row.to_bigquery_row()["buffer_mode"] for row in nccl_rows} == {
        "in_place",
        "out_of_place",
    }


def test_artifact_manifest_projection_validates_fixed_members(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    bundle = tmp_path / "run-artifacts.tar.zst"
    metrics.write_text(
        json.dumps({"run": RUN_ID, "key": "collection_complete", "value": "true"})
        + "\n",
        encoding="utf-8",
    )
    bundle.write_bytes(b"bundle")
    manifest = tmp_path / "artifact-manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "artifact-manifest/v1",
            "run_id": RUN_ID,
            "logical_artifact_name": RUN_ID,
            "run_purpose": "bench",
            "files": [
                {
                    "role": "run_bundle",
                    "path": "run-artifacts.tar.zst",
                    "size_bytes": bundle.stat().st_size,
                    "sha256": _sha(bundle),
                },
                {
                    "role": "metrics",
                    "path": "metrics.jsonl",
                    "size_bytes": metrics.stat().st_size,
                    "sha256": _sha(metrics),
                },
            ],
        },
    )
    rows = project_artifact_manifest(tmp_path, RUN_ID)
    assert [row.to_bigquery_row()["role"] for row in rows] == [
        "run_bundle",
        "metrics",
    ]
    assert all(row.source_sha256 == _sha(manifest) for row in rows)
