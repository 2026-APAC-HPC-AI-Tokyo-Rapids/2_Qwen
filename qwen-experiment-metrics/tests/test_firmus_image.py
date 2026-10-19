"""Regression checks for the pre-Slurm image selection and metadata inspection."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


SGLANG_PATCHES = ["firmus-staged-eagle-kv", "qwen3-sm90-megamoe", "firmus-dp-rank-query", "prefill-ready-handoff", "deepgemm-activation-layout", "rope-fp8-q", "qknorm-rope-fp8", "prefill-request-aligned-chunks", "qknorm-rope-fp8-kv", "qwen-topk-padding", "qknorm-rope-fp8-token-warps", "decode-async-admission", "dp-total-requests-rotate-ties", "megamoe-idle-prefill-geometry"]
ROUTER_PATCHES = ["prefill-load-lifetime", "no-delay-acceptor"]
CORRUPTIONS = [None, "hash", "router_hash", "router_nodelay_hash", "router_lock", "missing_router", "prefill_mode", "header_hash", "deep_gemm_first_hash", "deep_gemm_last_hash", "deep_gemm_missing_last", "deep_gemm_order"] + [f"missing_{name}" for name in SGLANG_PATCHES] + [f"missing_router_{name}" for name in ROUTER_PATCHES]


def sha256(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


# Kept on the branch but not wired into the image (changes the cross-rank protocol in every geometry; GPU-pending).
UNWIRED_DEEP_GEMM_PATCHES = {"sm90-flat-nvlink-barriers.patch"}


def test_deep_gemm_series_lists_every_wired_patch_and_pins_the_final_header() -> None:
    series = (ROOT / "patches/deep-gemm/series").read_text().split()
    assert len(series) == len(set(series))
    assert set(series) | UNWIRED_DEEP_GEMM_PATCHES == {path.name for path in (ROOT / "patches/deep-gemm").glob("*.patch")}
    assert not set(series) & UNWIRED_DEEP_GEMM_PATCHES
    header = (ROOT / "patches/deep-gemm/header.sha256").read_text().strip()
    assert len(header) == 64 and int(header, 16) >= 0


@pytest.mark.parametrize("corruption", CORRUPTIONS)
def test_metadata_extraction_supports_old_squashfs_and_rejects_changed_patches(tmp_path, corruption):
    embedded = tmp_path / "embedded"
    embedded.mkdir()
    patches = [f"patches/sglang/{name}.patch" for name in SGLANG_PATCHES] + [f"patches/sglang-router/{name}.patch" for name in ROUTER_PATCHES]
    manifest = {"applied_patches": [{"path": p, "sha256": sha256(p)} for p in patches]}
    if corruption == "hash":
        manifest["applied_patches"][0]["sha256"] = "0" * 64
    elif corruption and corruption.startswith("missing_router_"):
        manifest["applied_patches"].pop(patches.index(f"patches/sglang-router/{corruption[len('missing_router_'):]}.patch"))
    elif corruption and corruption.startswith("missing_") and corruption != "missing_router":
        manifest["applied_patches"].pop(patches.index(f"patches/sglang/{corruption[len('missing_'):]}.patch"))
    (embedded / "manifest.json").write_text(json.dumps(manifest))
    router = {"version": "0.3.2", "source_sha256": "bdbea1d54cce879fb83d2885d01ee3c006f242d0adbbac806f2fb2b91a5fe73d", "prefill_load": "dispatch_through_body_completion", "patch_sha256": sha256("patches/sglang-router/prefill-load-lifetime.patch"), "nodelay_patch_sha256": sha256("patches/sglang-router/no-delay-acceptor.patch"), "cargo_lock_sha256": sha256("patches/sglang-router/Cargo.lock")}
    if corruption == "router_hash":
        router["patch_sha256"] = "0" * 64
    if corruption == "router_nodelay_hash":
        router["nodelay_patch_sha256"] = "0" * 64
    if corruption == "router_lock":
        router["cargo_lock_sha256"] = "0" * 64
    if corruption != "missing_router":
        (embedded / "sglang-router-build.json").write_text(json.dumps(router))
    geometries = [{"role": "prefill", "block_m": 128, "block_n": 256, "epilogue_threads": 512, "stages": 3, "registers": 112, "reuse_accum": corruption != "prefill_mode", "arrival_counter": True, "full_sync": False, "swap_ab": False}, {"role": "decode", "block_m": 64, "block_n": 128, "epilogue_threads": 256, "stages": 8, "registers": 0, "reuse_accum": False, "arrival_counter": False, "full_sync": True, "swap_ab": True}]
    series = (ROOT / "patches/deep-gemm/series").read_text().split()
    deep_gemm = {name: sha256(f"patches/deep-gemm/{name}") for name in series}
    if corruption == "deep_gemm_first_hash":
        deep_gemm[series[0]] = "0" * 64
    if corruption == "deep_gemm_last_hash":
        deep_gemm[series[-1]] = "0" * 64
    if corruption == "deep_gemm_missing_last":
        del deep_gemm[series[-1]]
    if corruption == "deep_gemm_order":
        deep_gemm = {name: deep_gemm[name] for name in reversed(series)}
    header = "0" * 64 if corruption == "header_hash" else (ROOT / "patches/deep-gemm/header.sha256").read_text().strip()
    (embedded / "sm90-megamoe-build.json").write_text(json.dumps({"version": "0.1.5", "header_sha256": header, "patches": deep_gemm, "combine_accumulator": "FP32", "compiled_geometries": geometries}))
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
@pytest.mark.parametrize("require_build,source_commit,pin_matches,success", [(False, "old-build", True, True), (False, "old-build", False, False), (True, "new-run", True, True), (True, "old-build", True, False)])
def test_provision_uses_selected_image_for_reuse_and_keeps_fresh_build_commit_check(tmp_path, require_build, source_commit, pin_matches, success):
    workspace = tmp_path / "workspace"
    (workspace / "configs").mkdir(parents=True)
    # The real metadata validator is exercised above; this test isolates image selection.
    (workspace / "scripts").mkdir()
    (workspace / "scripts/check-firmus-image.sh").write_text("#!/bin/bash\nexit 0\n")
    config = {"framework": "sglang", "workload": "official", "concurrency": 256, "prefill": {"moe_a2a_backend": "none"}, "images": {"sglang": {"path": "/stale-image.sqsh"}}}
    archive = tmp_path / "archive"
    archive.mkdir()
    image = archive / "123.sqsh"
    image.write_bytes(b"selected-image")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    config["images"]["sglang"]["sha256"] = digest if pin_matches else "0" * 64
    (workspace / "configs/firmus-framework.json").write_text(json.dumps(config))
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


@pytest.mark.skipif(os.uname().sysname != "Linux", reason="provision uses Linux stat")
@pytest.mark.parametrize("applies", [True, False])
def test_provision_stages_sglang_overlay_patches_on_the_selected_image(tmp_path, applies):
    workspace = tmp_path / "workspace"
    (workspace / "configs").mkdir(parents=True)
    (workspace / "scripts").mkdir()
    (workspace / "scripts/check-firmus-image.sh").write_text("#!/bin/bash\nexit 0\n")
    (workspace / "patches/sglang").mkdir(parents=True)
    old = "x = 1\n" if applies else "x = 0\n"
    (workspace / "patches/sglang/first.patch").write_text(f"--- a/python/sglang/kernel.py\n+++ b/python/sglang/kernel.py\n@@ -1 +1 @@\n-{old}+x = 2\n")
    (workspace / "patches/sglang/second.patch").write_text("--- a/python/sglang/kernel.py\n+++ b/python/sglang/kernel.py\n@@ -1 +1 @@\n-x = 2\n+x = 3\n")
    image = tmp_path / "image.sqsh"
    image.write_bytes(b"base-image")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    (tmp_path / "image.container-sidecar.json").write_text("{}")
    overlay = ["patches/sglang/first.patch", "patches/sglang/second.patch"]
    config = {"framework": "sglang", "workload": "official", "concurrency": 192, "images": {"sglang": {"path": str(image), "sha256": digest, "size_bytes": image.stat().st_size, "provenance": str(tmp_path / "image.container-sidecar.json")}}, "sglang_overlay_patches": overlay}
    (workspace / "configs/firmus-framework.json").write_text(json.dumps(config))
    # Stand-in for the image's SGLang package, read-only like an unsquashfs extraction;
    # like unsquashfs it creates the -d directory but not its parent.
    unpack = tmp_path / "bin/unsquashfs"
    unpack.parent.mkdir()
    unpack.write_text('#!/bin/bash\nset -eu\nwhile [[ "$1" != -d ]]; do shift; done\n[[ -d "$(dirname "$2")" && ! -e "$2" ]]\nmkdir -p "$2/opt/sglang/source/python/sglang"\nprintf "x = 1\\n" >"$2/opt/sglang/source/python/sglang/kernel.py"\nchmod -R a-w "$2/opt"\n')
    unpack.chmod(0o755)
    script = (ROOT / "scripts/provision-firmus-framework.sh").read_text().split("model=/scratch/models/", 1)[0]
    result = subprocess.run(["bash", "-c", script], env={**os.environ, "PATH": f"{unpack.parent}:{os.environ['PATH']}", "GITHUB_WORKSPACE": str(workspace), "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "new-run", "GITHUB_OUTPUT": str(tmp_path / "outputs"), "RUNTIME_ROOT": str(tmp_path / "runs"), "PROFILE": "false", "REQUIRE_BUILD": "false"}, capture_output=True, text=True, check=False)
    assert (result.returncode == 0) is applies, result.stderr
    runtime = next((tmp_path / "runs").iterdir())
    if applies:
        assert (runtime / "overlay/sglang/python/sglang/kernel.py").read_text() == "x = 3\n"
        assert not (runtime / "overlay/extract").exists()
        record = json.loads((runtime / "source/sglang-overlay.json").read_text())
        assert record["base_image_sha256"] == digest and [p["path"] for p in record["patches"]] == overlay
        assert record["patches"][0]["sha256"] == hashlib.sha256((workspace / "patches/sglang/first.patch").read_bytes()).hexdigest()
        assert sorted(p.name for p in (runtime / "source/sglang-overlay").iterdir()) == ["first.patch", "second.patch"]
