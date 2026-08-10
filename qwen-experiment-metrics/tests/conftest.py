from __future__ import annotations

from pathlib import Path

import pytest

from qwen_experiment_metrics.canonical import jcs_bytes
from qwen_experiment_metrics.manifest import SourceTrack, create_run_manifest

RUN_ID = "r-0123456789abcdef0123456789abcdef"
EXECUTION_SETTINGS = {
    "schema_version": "qwen-execution-settings/v1",
    "run_purpose": "bench",
    "nodes": 2,
    "mpi_processes": 2,
    "gpus_per_node": 4,
    "tensor_parallel_size": 8,
    "resource_group": "b-batch",
    "transport_policy": "genkai-openmpi-default",
    "advertise_source": "mpi-rank-zero-hostname",
    "sglang_port": 25000,
    "node_agent_port": 25001,
    "rendezvous_timeout_sec": 300,
    "ready_timeout_sec": 300,
}


@pytest.fixture
def runtime(tmp_path: Path) -> Path:
    return tmp_path / "run"


@pytest.fixture
def manifest_path(runtime: Path) -> Path:
    return make_manifest(runtime)


def make_manifest(runtime: Path, purpose: str = "bench") -> Path:
    settings_path = runtime / "control/execution-settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_bytes(
        jcs_bytes(dict(EXECUTION_SETTINGS, run_purpose=purpose)) + b"\n"
    )
    return create_run_manifest(
        runtime,
        run_purpose=purpose,
        source_track=SourceTrack(
            object_hash_algorithm="sha1",
            base_role="score_reference" if purpose == "bench" else "profile_base",
            base_commit="a" * 40,
            head_commit="b" * 40,
        ),
        observed_control_plane_sha256="c" * 64,
        instrumentation_bundle_sha256="d" * 64,
        run_id=RUN_ID,
    )
