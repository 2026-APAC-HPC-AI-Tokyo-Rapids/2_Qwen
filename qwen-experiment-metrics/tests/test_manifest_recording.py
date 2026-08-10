from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_experiment_metrics.canonical import load_json_strict
from qwen_experiment_metrics.domain import Fact
from qwen_experiment_metrics.manifest import (
    ManifestError,
    RunManifest,
    SourceTrack,
    create_run_manifest,
    load_manifest,
    manifest_facts,
    new_run_id,
)
from qwen_experiment_metrics.recording import Recorder, RecorderError


def test_run_id_uses_exactly_128_csprng_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("secrets.token_bytes", lambda size: b"\xab" * size)
    assert new_run_id() == "r-" + "ab" * 16


def test_manifest_is_closed_and_exact(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    assert manifest.to_object() == load_json_strict(manifest_path.read_bytes())
    document = json.loads(manifest_path.read_text())
    document["identities"] = {}
    manifest_path.write_text(json.dumps(document))
    with pytest.raises(ManifestError, match="extra=.*identities"):
        load_manifest(manifest_path)


def test_manifest_cannot_be_replaced(runtime: Path, manifest_path: Path) -> None:
    existing = manifest_path.read_bytes()
    with pytest.raises(ManifestError, match="immutable"):
        create_run_manifest(
            runtime,
            run_purpose="bench",
            source_track=SourceTrack("sha1", "score_reference", "e" * 40, "f" * 40),
            observed_control_plane_sha256="1" * 64,
            instrumentation_bundle_sha256="2" * 64,
            run_id="r-" + "9" * 32,
        )
    assert manifest_path.read_bytes() == existing


def test_profile_manifest_requires_profile_base() -> None:
    manifest = RunManifest(
        run_id="r-" + "0" * 32,
        run_purpose="profile",
        source_track=SourceTrack("sha1", "score_reference", "a" * 40, "b" * 40),
        observed_control_plane_sha256="c" * 64,
        instrumentation_bundle_sha256="d" * 64,
    )
    from qwen_experiment_metrics.manifest import validate_manifest

    with pytest.raises(ManifestError, match="profile_base"):
        validate_manifest(manifest)


def test_recorder_only_publishes_normal_close(
    manifest_path: Path, runtime: Path
) -> None:
    recorder = Recorder.open(manifest_path, None, "runner", runtime)
    recorder.record_many(manifest_facts(load_manifest(manifest_path)))
    assert recorder.temporary_path.exists()
    assert not recorder.completed_path.exists()
    completed = recorder.close()
    assert completed.exists()
    assert not recorder.temporary_path.exists()
    rows = [load_json_strict(line) for line in completed.read_bytes().splitlines()]
    assert all(list(row) == ["run", "key", "value"] for row in rows)
    assert {row["key"] for row in rows} == {
        "run.identity.run_id",
        "run.identity.run_purpose",
        "run.identity.repository_id",
        "run.identity.instrumentation_bundle_sha256",
        "source.track.head_commit",
        "source.track.manifest_sha256",
        "source.track.score_reference_commit",
        "source.track.source_sha256",
    }


def test_recorder_leaves_aborted_fragment_unmergeable(
    manifest_path: Path, runtime: Path
) -> None:
    recorder = Recorder.open(manifest_path, None, "runner", runtime)
    recorder.record(Fact("run.identity.run_id", load_manifest(manifest_path).run_id))
    partial = recorder.abort()
    assert partial.suffix == ".tmp"
    assert partial.exists()
    assert not recorder.completed_path.exists()


def test_recorder_enforces_manifest_binding_and_ownership(
    manifest_path: Path, runtime: Path
) -> None:
    recorder = Recorder.open(manifest_path, None, "runner", runtime)
    with pytest.raises(RecorderError, match="persisted manifest"):
        recorder.record(Fact("run.identity.run_id", "r-" + "f" * 32))
    with pytest.raises(ValueError, match="cannot own"):
        recorder.record(Fact("result.request.returned_count", 1))
    recorder.abort()
