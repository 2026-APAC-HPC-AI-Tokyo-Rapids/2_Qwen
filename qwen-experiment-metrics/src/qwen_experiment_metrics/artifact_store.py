from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .canonical import jcs_text
from .manifest import load_manifest

ROLE_FILENAMES = {
    "run_bundle": "run-artifacts.tar.zst",
    "metrics": "metrics.jsonl",
    "profile_raw": "profile-raw.tar.zst",
}


@dataclass(frozen=True, slots=True)
class Member:
    path: str
    size_bytes: int
    sha256: str
    role: str

    @classmethod
    def from_file(cls, run_dir: Path, relative: str, role: str) -> Member:
        path = _member_path(run_dir, relative)
        return cls(relative, path.stat().st_size, _sha256(path), role)

    def as_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class ArtifactSet:
    manifest_path: Path
    files: tuple[Path, ...]


class RunPackager:
    def package(
        self,
        manifest_path: Path,
        runtime_dir: Path,
        *,
        run_members: tuple[tuple[str, str], ...],
        profile_members: tuple[str, ...] = (),
    ) -> ArtifactSet:
        manifest = load_manifest(manifest_path)
        run_dir = runtime_dir.resolve()
        if manifest_path.resolve() != run_dir / "control/run-manifest.json":
            raise ValueError("manifest must be the runtime control manifest")
        metrics = run_dir / ROLE_FILENAMES["metrics"]
        if not metrics.is_file():
            raise FileNotFoundError(metrics)
        # metrics.jsonl is both the publishable sidecar and a required member of
        # the self-contained run bundle.  Keep a single caller-independent owner
        # for that duplication so workflows cannot accidentally omit it.
        members = tuple(
            sorted(
                (
                    *(
                        Member.from_file(run_dir, relative, role)
                        for relative, role in run_members
                    ),
                    Member.from_file(run_dir, ROLE_FILENAMES["metrics"], "metrics"),
                ),
                key=lambda item: item.path.encode(),
            )
        )
        if len({member.path for member in members}) != len(members):
            raise ValueError("run bundle members must be path-unique")
        staging = run_dir / "archive-staging"
        staging.mkdir(exist_ok=True)
        bundle_manifest = staging / "bundle.manifest.json"
        _atomic_text(
            bundle_manifest,
            jcs_text(
                {
                    "schema_version": "bundle-manifest/v1",
                    "run_id": manifest.run_id,
                    "members": [member.as_json() for member in members],
                }
            )
            + "\n",
        )
        run_bundle = run_dir / ROLE_FILENAMES["run_bundle"]
        _tar_zstd(
            run_dir,
            tuple(member.path for member in members),
            bundle_manifest,
            run_bundle,
        )
        outputs: list[tuple[str, Path]] = [
            ("run_bundle", run_bundle),
            ("metrics", metrics),
        ]
        if manifest.run_purpose == "profile":
            if not profile_members:
                raise ValueError("profile runs require profile raw members")
            profile_archive = run_dir / ROLE_FILENAMES["profile_raw"]
            for relative in profile_members:
                _member_path(run_dir, relative)
            _tar_zstd(run_dir, profile_members, None, profile_archive)
            outputs.append(("profile_raw", profile_archive))
        elif profile_members:
            raise ValueError("bench runs cannot contain profile raw members")
        artifact_manifest = run_dir / "artifact-manifest.json"
        _atomic_text(
            artifact_manifest,
            jcs_text(
                {
                    "schema_version": "artifact-manifest/v1",
                    "run_id": manifest.run_id,
                    "logical_artifact_name": manifest.run_id,
                    "run_purpose": manifest.run_purpose,
                    "files": [
                        {
                            "role": role,
                            "path": path.name,
                            "size_bytes": path.stat().st_size,
                            "sha256": _sha256(path),
                        }
                        for role, path in outputs
                    ],
                }
            )
            + "\n",
        )
        return ArtifactSet(artifact_manifest, tuple(path for _, path in outputs))


def _member_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if (
        not relative
        or posix.is_absolute()
        or ".." in posix.parts
        or any(not part for part in posix.parts)
    ):
        raise ValueError(f"unsafe artifact member path: {relative!r}")
    root = root.resolve()
    path = (root / Path(*posix.parts)).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"artifact member escapes runtime root: {relative!r}")
    return path


def _tar_zstd(
    root: Path,
    members: tuple[str, ...],
    bundle_manifest: Path | None,
    output: Path,
) -> None:
    if output.exists():
        raise FileExistsError(output)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
    temporary.unlink()
    argv = ["tar", "-I", "zstd", "-cf", str(temporary), "-C", str(root), *members]
    if bundle_manifest is not None:
        argv.extend(
            [
                "--transform",
                "s|^archive-staging/||",
                "archive-staging/bundle.manifest.json",
            ]
        )
    try:
        subprocess.run(argv, check=True)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)
