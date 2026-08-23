from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import BinaryIO, Self

from .canonical import encode_logical_value, fact_row_bytes
from .domain import Fact
from .manifest import RunManifest, load_manifest


class RecorderError(RuntimeError):
    pass


class Recorder:
    """Buffered single-producer fragment writer with atomic completion."""

    def __init__(
        self,
        manifest: RunManifest,
        producer_id: str,
        temporary_path: Path,
        completed_path: Path,
        stream: BinaryIO,
    ) -> None:
        self.manifest = manifest
        self.producer_id = producer_id
        self.temporary_path = temporary_path
        self.completed_path = completed_path
        self._stream = stream
        self._closed = False

    @classmethod
    def open(
        cls,
        manifest_path: Path,
        registry_path: Path | None,
        producer_id: str,
        runtime_dir: Path,
    ) -> Recorder:
        manifest = load_manifest(manifest_path)
        directory = runtime_dir / "domain" / "fragments"
        directory.mkdir(parents=True, exist_ok=True)
        token = _producer_token(producer_id)
        completed_path = directory / f"producer--{token}.jsonl"
        temporary_path = directory / f"producer--{token}.jsonl.tmp"
        if completed_path.exists():
            raise RecorderError(f"completed fragment already exists: {producer_id}")
        try:
            stream = temporary_path.open("xb", buffering=64 * 1024)
        except FileExistsError as error:
            raise RecorderError(
                f"partial fragment already exists: {producer_id}"
            ) from error
        return cls(
            manifest,
            producer_id,
            temporary_path,
            completed_path,
            stream,
        )

    def record(self, fact: Fact) -> None:
        if self._closed:
            raise RecorderError("recorder is closed")
        manifest_values = {
            "run.identity.run_id": self.manifest.run_id,
            "run.identity.run_purpose": self.manifest.run_purpose,
            "run.identity.repository_id": self.manifest.repository_id,
            "run.identity.instrumentation_bundle_sha256": (
                self.manifest.instrumentation_bundle_sha256
            ),
        }
        if (
            fact.key in manifest_values
            and fact.logical_value != manifest_values[fact.key]
        ):
            raise RecorderError(f"{fact.key} does not match the persisted manifest")
        encoded = encode_logical_value(fact.logical_value)
        self._stream.write(
            fact_row_bytes(self.manifest.run_id, fact.key, encoded) + b"\n"
        )

    def record_many(self, facts: object) -> None:
        for fact in facts:  # type: ignore[union-attr]
            if not isinstance(fact, Fact):
                raise TypeError("record_many accepts Fact values")
            self.record(fact)

    def close(self) -> Path:
        if self._closed:
            return self.completed_path
        self._stream.flush()
        self._stream.close()
        os.replace(self.temporary_path, self.completed_path)
        self._closed = True
        return self.completed_path

    def abort(self) -> Path:
        if not self._closed:
            self._stream.close()
            self._closed = True
        return self.temporary_path

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def producer_from_fragment(path: Path) -> str:
    prefix = "producer--"
    suffix = ".jsonl"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise RecorderError(f"not a completed producer fragment: {path.name}")
    token = path.name[len(prefix) : -len(suffix)]
    padding = "=" * (-len(token) % 4)
    try:
        return base64.urlsafe_b64decode(token + padding).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise RecorderError(f"invalid producer fragment name: {path.name}") from error


def _producer_token(producer_id: str) -> str:
    return base64.urlsafe_b64encode(producer_id.encode()).decode().rstrip("=")
