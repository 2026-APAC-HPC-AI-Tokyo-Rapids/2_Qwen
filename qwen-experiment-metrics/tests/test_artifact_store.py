from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from qwen_experiment_metrics.artifact_store import RunPackager
from qwen_experiment_metrics.manifest import load_manifest


def test_packager_creates_fixed_bench_artifact_set(
    runtime: Path, manifest_path: Path
) -> None:
    manifest = load_manifest(manifest_path)
    (runtime / "logs").mkdir()
    (runtime / "logs/benchmark.log").write_text("complete\n", encoding="utf-8")
    (runtime / "metrics.jsonl").write_text(
        json.dumps(
            {"run": manifest.run_id, "key": "collection_complete", "value": "true"},
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    result = RunPackager().package(
        manifest_path,
        runtime,
        run_members=(("logs/benchmark.log", "benchmark_log"),),
    )

    assert [path.name for path in result.files] == [
        "run-artifacts.tar.zst",
        "metrics.jsonl",
    ]
    artifact = json.loads(result.manifest_path.read_text())
    assert artifact["run_id"] == manifest.run_id
    assert [item["path"] for item in artifact["files"]] == [
        "run-artifacts.tar.zst",
        "metrics.jsonl",
    ]
    assert [item["role"] for item in artifact["files"]] == [
        "run_bundle",
        "metrics",
    ]
    members = subprocess.run(
        ["tar", "-I", "zstd", "-tf", runtime / "run-artifacts.tar.zst"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    assert members == [
        "logs/benchmark.log",
        "metrics.jsonl",
        "bundle.manifest.json",
    ]

    bundle_manifest = json.loads(
        subprocess.run(
            [
                "tar",
                "-I",
                "zstd",
                "-xOf",
                runtime / "run-artifacts.tar.zst",
                "bundle.manifest.json",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    )
    assert [item["path"] for item in bundle_manifest["members"]] == [
        "logs/benchmark.log",
        "metrics.jsonl",
    ]
    assert [item["role"] for item in bundle_manifest["members"]] == [
        "benchmark_log",
        "metrics",
    ]


def test_packager_rejects_member_escape(runtime: Path, manifest_path: Path) -> None:
    (runtime / "metrics.jsonl").write_text("not-empty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe artifact member"):
        RunPackager().package(
            manifest_path,
            runtime,
            run_members=(("../outside", "invalid"),),
        )


def test_packager_rejects_caller_supplied_metrics_member(
    runtime: Path, manifest_path: Path
) -> None:
    (runtime / "metrics.jsonl").write_text("not-empty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="path-unique"):
        RunPackager().package(
            manifest_path,
            runtime,
            run_members=(("metrics.jsonl", "unexpected-duplicate"),),
        )
