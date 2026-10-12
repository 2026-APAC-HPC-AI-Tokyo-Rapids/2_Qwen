"""Regression checks for the pre-Slurm image selection and metadata inspection."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("corruption", [None, "hash", "missing_dp_query"])
def test_metadata_extraction_supports_old_squashfs_and_rejects_changed_patches(tmp_path, corruption):
    embedded = tmp_path / "embedded"
    embedded.mkdir()
    patches = ["patches/sglang/firmus-staged-eagle-kv.patch", "patches/sglang/qwen3-sm90-megamoe.patch", "patches/sglang/firmus-dp-rank-query.patch"]
    manifest = {"applied_patches": [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()} for p in patches]}
    if corruption == "hash":
        manifest["applied_patches"][0]["sha256"] = "0" * 64
    elif corruption == "missing_dp_query":
        manifest["applied_patches"].pop()
    (embedded / "manifest.json").write_text(json.dumps(manifest))
    (embedded / "sm90-megamoe-build.json").write_text(json.dumps({"version": "0.1.5", "header_sha256": "5479807104bf1aeed4cc3ecc8ce57d2a835d68457f9cf4962ef3f15cc176e8d2", "patch_sha256": hashlib.sha256((ROOT / "patches/deep-gemm/sm90-late-topk-fp32.patch").read_bytes()).hexdigest(), "combine_accumulator": "FP32", "compiled_geometries": [[64, 128, 256, True], [128, 128, 256, False], [128, 256, 512, False]]}))
    unpack = tmp_path / "unsquashfs"
    unpack.write_text('#!/bin/bash\nset -eu\n[[ "$*" != *"-cat"* ]] || exit 1\nwhile [[ "$1" != -d ]]; do shift; done\n[[ ! -e "$2" ]] || exit 1\nmkdir -p "$2/opt/apac"\ncp "$EMBEDDED"/*.json "$2/opt/apac/"\nchmod -R a-w "$2"\n')
    unpack.chmod(0o755)
    destination = tmp_path / "metadata"
    destination.mkdir()
    result = subprocess.run(["bash", str(ROOT / "scripts/check-firmus-image.sh"), "/fixture.sqsh", str(destination), str(ROOT)], env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "EMBEDDED": str(embedded)}, capture_output=True, text=True, check=False)
    assert (result.returncode == 0) is (corruption is None), result.stderr
    assert not list(destination.glob(".image-metadata.*"))
    if corruption is None:
        assert json.loads((destination / "image-manifest.json").read_text()) == manifest


@pytest.mark.skipif(os.uname().sysname != "Linux", reason="provision uses Linux stat")
@pytest.mark.parametrize("require_build,source_commit,success", [(False, "old-build", True), (True, "new-run", True), (True, "old-build", False)])
def test_provision_uses_selected_image_for_reuse_and_keeps_fresh_build_commit_check(tmp_path, require_build, source_commit, success):
    workspace = tmp_path / "workspace"
    (workspace / "configs").mkdir(parents=True)
    config = {"framework": "sglang", "workload": "official", "concurrency": 256, "prefill": {"moe_a2a_backend": "none"}, "images": {"sglang": {"path": "/stale-image.sqsh"}}}
    (workspace / "configs/firmus-framework.json").write_text(json.dumps(config))
    archive = tmp_path / "archive"
    archive.mkdir()
    image = archive / "123.sqsh"
    image.write_bytes(b"selected-image")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    (archive / "123.container-sidecar.json").write_text(json.dumps({"source_commits": {"qwen": source_commit}, "container_artifact": {"sha256": digest}}))
    # Execute the actual image-selection/hash/metadata phase; no model staging
    # or scheduler operation is part of this regression test.
    script = (ROOT / "scripts/provision-firmus-framework.sh").read_text().split("model=/scratch/models/", 1)[0]
    script = script.replace("archive=/scratch/home/ACCOUNT/qwen/containers", f'archive="{archive}"')
    result = subprocess.run(["bash", "-c", script], env={**os.environ, "GITHUB_WORKSPACE": str(workspace), "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "new-run", "GITHUB_OUTPUT": str(tmp_path / "outputs"), "RUNTIME_ROOT": str(tmp_path / "runs"), "PROFILE": "true", "REQUIRE_BUILD": str(require_build).lower(), "CONTAINER_SHA256": digest, "CONTAINER_SIZE": str(image.stat().st_size)}, capture_output=True, text=True, check=False)
    assert (result.returncode == 0) is success, result.stderr
    if success:
        runtime = next((tmp_path / "runs").iterdir())
        assert (runtime / "sglang.sqsh").resolve() == image
        assert json.loads((runtime / "source/firmus-framework.json").read_text())["images"]["sglang"]["sha256"] == digest
