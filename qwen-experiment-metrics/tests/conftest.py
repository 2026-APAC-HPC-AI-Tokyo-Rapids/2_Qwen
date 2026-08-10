from __future__ import annotations

from pathlib import Path

import pytest

from qwen_experiment_metrics.manifest import SourceTrack, create_run_manifest

RUN_ID = "r-0123456789abcdef0123456789abcdef"


@pytest.fixture
def runtime(tmp_path: Path) -> Path:
    return tmp_path / "run"


@pytest.fixture
def manifest_path(runtime: Path) -> Path:
    return make_manifest(runtime)


def make_manifest(runtime: Path, purpose: str = "bench") -> Path:
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
