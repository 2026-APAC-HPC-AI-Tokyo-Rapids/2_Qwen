from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from urllib.request import Request

import pytest

from qwen_experiment_metrics.publish import (
    BigQueryConfig,
    publish_metrics,
    read_metrics,
)


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _artifact_manifest(metrics: Path, run_id: str) -> Path:
    bundle = metrics.parent / "run-artifacts.tar.zst"
    bundle.write_bytes(b"bundle")
    files = []
    for role, path in (("run_bundle", bundle), ("metrics", metrics)):
        files.append(
            {
                "role": role,
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = metrics.parent / "artifact-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "artifact-manifest/v1",
                "run_id": run_id,
                "logical_artifact_name": run_id,
                "run_purpose": "bench",
                "files": files,
            },
            separators=(",", ":"),
        )
    )
    return manifest


def test_publish_validates_and_chunks_immutable_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write(
        path,
        [
            {"run": "r-1", "key": f"key.{index}", "value": str(index)}
            for index in range(501)
        ],
    )
    requests: list[Request] = []

    def send(request: Request) -> bytes:
        requests.append(request)
        return b"{}"

    count = publish_metrics(
        path,
        BigQueryConfig("project", "dataset", "table", "asia-northeast1"),
        artifact_manifest=_artifact_manifest(path, "r-1"),
        expected_run="r-1",
        request=send,
    )
    assert count == 501
    assert [len(json.loads(item.data or b"{}")["rows"]) for item in requests] == [
        500,
        1,
    ]
    assert len(json.loads(requests[0].data or b"{}")["rows"][0]["insertId"]) == 64


def test_publish_rejects_foreign_or_rejected_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write(path, [{"run": "r-2", "key": "a", "value": "1"}])
    with pytest.raises(ValueError, match="foreign run"):
        read_metrics(path, expected_run="r-1")
    with pytest.raises(RuntimeError, match="rejected rows"):
        publish_metrics(
            path,
            BigQueryConfig("project", "dataset", "table", "asia-northeast1"),
            artifact_manifest=_artifact_manifest(path, "r-2"),
            request=lambda _: b'{"insertErrors":[{"index":0}]}',
        )


def test_read_metrics_rejects_duplicate_json_members(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"run":"r-one","run":"r-two","key":"x","value":"1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid metrics row 1"):
        read_metrics(path)


def test_publish_rejects_tampered_artifact_binding(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write(path, [{"run": "r-3", "key": "a", "value": "1"}])
    manifest = _artifact_manifest(path, "r-3")
    path.write_text('{"run":"r-3","key":"a","value":"2"}\n')

    with pytest.raises(ValueError, match="size/hash"):
        publish_metrics(
            path,
            BigQueryConfig("project", "dataset", "table", "asia-northeast1"),
            artifact_manifest=manifest,
            request=lambda _: b"{}",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda files: files.reverse(), "role order"),
        (
            lambda files: files[0].__setitem__("path", "renamed.tar.zst"),
            "fixed filename",
        ),
    ],
)
def test_publish_rejects_artifact_order_and_filename(
    tmp_path: Path,
    mutation: Callable[[list[dict[str, object]]], None],
    message: str,
) -> None:
    path = tmp_path / "metrics.jsonl"
    _write(path, [{"run": "r-4", "key": "a", "value": "1"}])
    manifest = _artifact_manifest(path, "r-4")
    document = json.loads(manifest.read_text())
    mutation(document["files"])
    manifest.write_text(json.dumps(document, separators=(",", ":")))

    with pytest.raises(ValueError, match=message):
        publish_metrics(
            path,
            BigQueryConfig("project", "dataset", "table", "asia-northeast1"),
            artifact_manifest=manifest,
            request=lambda _: b"{}",
        )


def test_publish_rejects_symlinked_artifact(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write(path, [{"run": "r-5", "key": "a", "value": "1"}])
    manifest = _artifact_manifest(path, "r-5")
    bundle = tmp_path / "run-artifacts.tar.zst"
    real_bundle = tmp_path / "real-run-artifacts.tar.zst"
    bundle.rename(real_bundle)
    bundle.symlink_to(real_bundle.name)

    with pytest.raises(ValueError, match="escapes"):
        publish_metrics(
            path,
            BigQueryConfig("project", "dataset", "table", "asia-northeast1"),
            artifact_manifest=manifest,
            request=lambda _: b"{}",
        )
