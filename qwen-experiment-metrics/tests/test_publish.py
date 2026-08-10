from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from urllib.request import Request

import pytest

from qwen_experiment_metrics.physical_publish import PhysicalPublisherConfig
from qwen_experiment_metrics.projections import ProjectionBatch, ProjectionRow
from qwen_experiment_metrics.publish import (
    BigQueryConfig,
    publish_metrics,
    publish_metrics_with_projections,
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


def test_physical_projection_publish_precedes_raw_publish(tmp_path: Path) -> None:
    run_id = "r-0123456789abcdef0123456789abcdef"
    path = tmp_path / "metrics.jsonl"
    _write(path, [{"run": run_id, "key": "collection_complete", "value": "true"}])
    artifact = _artifact_manifest(path, run_id)
    row = ProjectionRow.from_mapping(
        "qwen_source_track_v2",
        run_id,
        "control/run-manifest.json",
        "a" * 64,
        {
            "run_purpose": "bench",
            "repository_id": "apac2026-qwen",
            "source_track": {},
            "source_track_sha256": "b" * 64,
        },
    )
    calls: list[str] = []

    report, raw_count = publish_metrics_with_projections(
        path,
        BigQueryConfig("project", "dataset", "qwen_metrics", "location"),
        ProjectionBatch((row,)),
        PhysicalPublisherConfig("project", "dataset", "location"),
        artifact_manifest=artifact,
        expected_run=run_id,
        projection_sender=lambda _: calls.append("physical") or b"{}",
        raw_request=lambda _: calls.append("raw") or b"{}",
    )
    assert calls == ["physical", "raw"]
    assert report.relation_rows == 1
    assert raw_count == 1


def test_physical_failure_prevents_raw_publish(tmp_path: Path) -> None:
    run_id = "r-0123456789abcdef0123456789abcdef"
    path = tmp_path / "metrics.jsonl"
    _write(path, [{"run": run_id, "key": "collection_complete", "value": "true"}])
    artifact = _artifact_manifest(path, run_id)
    row = ProjectionRow.from_mapping(
        "qwen_source_track_v2",
        run_id,
        "control/run-manifest.json",
        "a" * 64,
        {
            "run_purpose": "bench",
            "repository_id": "apac2026-qwen",
            "source_track": {},
            "source_track_sha256": "b" * 64,
        },
    )
    raw_called = False

    def raw_sender(_: Request) -> bytes:
        nonlocal raw_called
        raw_called = True
        return b"{}"

    with pytest.raises(RuntimeError, match="rejected projection"):
        publish_metrics_with_projections(
            path,
            BigQueryConfig("project", "dataset", "qwen_metrics", "location"),
            ProjectionBatch((row,)),
            PhysicalPublisherConfig("project", "dataset", "location"),
            artifact_manifest=artifact,
            expected_run=run_id,
            projection_sender=lambda _: b'{"insertErrors":[{"index":0}]}',
            raw_request=raw_sender,
        )
    assert not raw_called


def test_combined_publish_resolves_one_token_for_both_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "r-0123456789abcdef0123456789abcdef"
    path = tmp_path / "metrics.jsonl"
    _write(path, [{"run": run_id, "key": "collection_complete", "value": "true"}])
    artifact = _artifact_manifest(path, run_id)
    row = ProjectionRow.from_mapping(
        "qwen_source_track_v2",
        run_id,
        "control/run-manifest.json",
        "a" * 64,
        {
            "run_purpose": "bench",
            "repository_id": "apac2026-qwen",
            "source_track": {},
            "source_track_sha256": "b" * 64,
        },
    )
    tokens: list[None] = []
    requests: list[Request] = []
    monkeypatch.setattr(
        "qwen_experiment_metrics.publish._access_token",
        lambda: tokens.append(None) or "shared-token",
    )
    monkeypatch.setattr(
        "qwen_experiment_metrics.publish._urlopen",
        lambda request: requests.append(request) or b"{}",
    )

    publish_metrics_with_projections(
        path,
        BigQueryConfig("project", "dataset", "qwen_metrics", "location"),
        ProjectionBatch((row,)),
        PhysicalPublisherConfig("project", "dataset", "location"),
        artifact_manifest=artifact,
        expected_run=run_id,
    )
    assert tokens == [None]
    assert len(requests) == 2
    assert all(request.get_header("Authorization") == "Bearer shared-token" for request in requests)
