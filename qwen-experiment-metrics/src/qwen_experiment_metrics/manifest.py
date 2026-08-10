from __future__ import annotations

import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

from .canonical import jcs_bytes, load_json_strict
from .domain import Fact

RUN_ID_PATTERN = re.compile(r"^r-[0-9a-f]{32}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REPOSITORY_ID = "apac2026-qwen"


class ManifestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceTrack:
    object_hash_algorithm: str
    base_role: str
    base_commit: str
    head_commit: str

    def to_object(self) -> dict[str, str]:
        return {
            "schema_version": "source-track/v1",
            "object_hash_algorithm": self.object_hash_algorithm,
            "base_role": self.base_role,
            "base_commit": self.base_commit,
            "head_commit": self.head_commit,
        }


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    run_purpose: str
    source_track: SourceTrack
    observed_control_plane_sha256: str
    instrumentation_bundle_sha256: str
    repository_id: str = REPOSITORY_ID
    schema_version: str = "run-manifest/v1"

    def to_object(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_purpose": self.run_purpose,
            "repository_id": self.repository_id,
            "source_track": self.source_track.to_object(),
            "observed_control_plane_sha256": self.observed_control_plane_sha256,
            "instrumentation_bundle_sha256": self.instrumentation_bundle_sha256,
        }


def new_run_id() -> str:
    return f"r-{secrets.token_bytes(16).hex()}"


def create_run_manifest(
    runtime_dir: Path,
    *,
    run_purpose: str,
    source_track: SourceTrack,
    observed_control_plane_sha256: str,
    instrumentation_bundle_sha256: str,
    run_id: str | None = None,
) -> Path:
    manifest = RunManifest(
        run_id=run_id or new_run_id(),
        run_purpose=run_purpose,
        source_track=source_track,
        observed_control_plane_sha256=observed_control_plane_sha256,
        instrumentation_bundle_sha256=instrumentation_bundle_sha256,
    )
    validate_manifest(manifest)
    path = runtime_dir / "control" / "run-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ManifestError(f"run manifest is immutable and already exists: {path}")
    _atomic_write(path, jcs_bytes(manifest.to_object()) + b"\n")
    return path


def load_manifest(path: Path) -> RunManifest:
    try:
        document = load_json_strict(path.read_bytes())
    except OSError as error:
        raise ManifestError(f"cannot read manifest: {path}") from error
    if not isinstance(document, dict):
        raise ManifestError("run manifest must be an object")
    expected = {
        "schema_version",
        "run_id",
        "run_purpose",
        "repository_id",
        "source_track",
        "observed_control_plane_sha256",
        "instrumentation_bundle_sha256",
    }
    _exact_fields(document, expected, "run manifest")
    source = document["source_track"]
    if not isinstance(source, dict):
        raise ManifestError("source_track must be an object")
    _exact_fields(
        source,
        {
            "schema_version",
            "object_hash_algorithm",
            "base_role",
            "base_commit",
            "head_commit",
        },
        "source_track",
    )
    if source["schema_version"] != "source-track/v1":
        raise ManifestError("unsupported source_track schema")
    source_track = SourceTrack(
        object_hash_algorithm=_string(source["object_hash_algorithm"], "algorithm"),
        base_role=_string(source["base_role"], "base_role"),
        base_commit=_string(source["base_commit"], "base_commit"),
        head_commit=_string(source["head_commit"], "head_commit"),
    )
    manifest = RunManifest(
        schema_version=_string(document["schema_version"], "schema_version"),
        run_id=_string(document["run_id"], "run_id"),
        run_purpose=_string(document["run_purpose"], "run_purpose"),
        repository_id=_string(document["repository_id"], "repository_id"),
        source_track=source_track,
        observed_control_plane_sha256=_string(
            document["observed_control_plane_sha256"],
            "observed_control_plane_sha256",
        ),
        instrumentation_bundle_sha256=_string(
            document["instrumentation_bundle_sha256"],
            "instrumentation_bundle_sha256",
        ),
    )
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: RunManifest) -> None:
    if manifest.schema_version != "run-manifest/v1":
        raise ManifestError("unsupported run manifest schema")
    if RUN_ID_PATTERN.fullmatch(manifest.run_id) is None:
        raise ManifestError("run_id must be r- followed by 128-bit lowercase hex")
    if manifest.run_purpose not in {"bench", "profile"}:
        raise ManifestError("run_purpose must be bench or profile")
    if manifest.repository_id != REPOSITORY_ID:
        raise ManifestError(f"repository_id must be {REPOSITORY_ID}")
    if manifest.source_track.object_hash_algorithm not in {"sha1", "sha256"}:
        raise ManifestError("unsupported Git object hash algorithm")
    expected_oid_length = (
        40 if manifest.source_track.object_hash_algorithm == "sha1" else 64
    )
    for name, oid in (
        ("base_commit", manifest.source_track.base_commit),
        ("head_commit", manifest.source_track.head_commit),
    ):
        if re.fullmatch(f"[0-9a-f]{{{expected_oid_length}}}", oid) is None:
            raise ManifestError(f"{name} is not a full Git object ID")
    expected_role = (
        "score_reference" if manifest.run_purpose == "bench" else "profile_base"
    )
    if manifest.source_track.base_role != expected_role:
        raise ManifestError(f"base_role must be {expected_role} for this purpose")
    for name, value in (
        ("observed_control_plane_sha256", manifest.observed_control_plane_sha256),
        ("instrumentation_bundle_sha256", manifest.instrumentation_bundle_sha256),
    ):
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ManifestError(f"{name} must be lowercase SHA-256")


def manifest_facts(manifest: RunManifest) -> tuple[Fact, ...]:
    return (
        Fact("run.identity.run_id", manifest.run_id),
        Fact("run.identity.run_purpose", manifest.run_purpose),
        Fact("run.identity.repository_id", manifest.repository_id),
        Fact(
            "run.identity.instrumentation_bundle_sha256",
            manifest.instrumentation_bundle_sha256,
        ),
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb", buffering=64 * 1024) as stream:
            stream.write(payload)
            stream.flush()
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _exact_fields(document: dict[str, object], expected: set[str], label: str) -> None:
    actual = set(document)
    if actual != expected:
        raise ManifestError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{label} must be a string")
    return value
