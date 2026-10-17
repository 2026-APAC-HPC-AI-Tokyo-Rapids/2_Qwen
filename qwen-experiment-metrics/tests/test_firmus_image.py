"""Regression checks for the pre-Slurm image selection and metadata inspection."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("corruption", [None, "hash", "missing_dp_query", "missing_handoff", "missing_activation", "missing_rope", "missing_qknorm_rope", "router_hash", "router_lock", "missing_router", "pipeline_hash", "prefill_hash", "prefill_mode", "decode_tiles_hash", "prefill_phase_hash", "swap_ab_hoist_hash", "l1_quant_hash", "dispatch_overlap_hash"])
def test_metadata_extraction_supports_old_squashfs_and_rejects_changed_patches(tmp_path, corruption):
    embedded = tmp_path / "embedded"
    embedded.mkdir()
    patches = ["patches/sglang/firmus-staged-eagle-kv.patch", "patches/sglang/qwen3-sm90-megamoe.patch", "patches/sglang/firmus-dp-rank-query.patch", "patches/sglang/prefill-ready-handoff.patch", "patches/sglang/deepgemm-activation-layout.patch", "patches/sglang/rope-fp8-q.patch", "patches/sglang/qknorm-rope-fp8.patch"]
    manifest = {"applied_patches": [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()} for p in patches]}
    if corruption == "hash":
        manifest["applied_patches"][0]["sha256"] = "0" * 64
    elif corruption == "missing_dp_query":
        manifest["applied_patches"].pop(2)
    elif corruption == "missing_handoff":
        manifest["applied_patches"].pop(3)
    elif corruption == "missing_activation":
        manifest["applied_patches"].pop(4)
    elif corruption == "missing_rope":
        manifest["applied_patches"].pop(5)
    elif corruption == "missing_qknorm_rope":
        manifest["applied_patches"].pop()
    (embedded / "manifest.json").write_text(json.dumps(manifest))
    router = {"version": "0.3.2", "source_sha256": "bdbea1d54cce879fb83d2885d01ee3c006f242d0adbbac806f2fb2b91a5fe73d", "prefill_load": "dispatch_through_body_completion", "patch_sha256": hashlib.sha256((ROOT / "patches/sglang-router/prefill-load-lifetime.patch").read_bytes()).hexdigest(), "cargo_lock_sha256": hashlib.sha256((ROOT / "patches/sglang-router/Cargo.lock").read_bytes()).hexdigest()}
    if corruption == "router_hash":
        router["patch_sha256"] = "0" * 64
    if corruption == "router_lock":
        router["cargo_lock_sha256"] = "0" * 64
    if corruption != "missing_router":
        (embedded / "sglang-router-build.json").write_text(json.dumps(router))
    geometries = [{"role": "prefill", "block_m": 128, "block_n": 256, "epilogue_threads": 512, "stages": 3, "registers": 112, "reuse_accum": corruption != "prefill_mode", "arrival_counter": True, "full_sync": False, "swap_ab": False}, {"role": "decode", "block_m": 64, "block_n": 128, "epilogue_threads": 256, "stages": 8, "registers": 0, "reuse_accum": False, "arrival_counter": False, "full_sync": True, "swap_ab": True}]
    (embedded / "sm90-megamoe-build.json").write_text(json.dumps({"version": "0.1.5", "header_sha256": "135a43befec61e04c27a51be3ca0e4c8b928669529a830acde134042379e1a6d", "patch_sha256": hashlib.sha256((ROOT / "patches/deep-gemm/sm90-late-topk-fp32.patch").read_bytes()).hexdigest(), "fc2_pipeline_patch_sha256": "0" * 64 if corruption == "pipeline_hash" else hashlib.sha256((ROOT / "patches/deep-gemm/sm90-fc2-pipeline.patch").read_bytes()).hexdigest(), "prefill_rescale_patch_sha256": "0" * 64 if corruption == "prefill_hash" else hashlib.sha256((ROOT / "patches/deep-gemm/sm90-prefill-rescale.patch").read_bytes()).hexdigest(), "decode_tiles_patch_sha256": "0" * 64 if corruption == "decode_tiles_hash" else hashlib.sha256((ROOT / "patches/deep-gemm/sm90-decode-exact-tiles.patch").read_bytes()).hexdigest(), "prefill_phase_patch_sha256": "0" * 64 if corruption == "prefill_phase_hash" else hashlib.sha256((ROOT / "patches/deep-gemm/sm90-prefill-phase.patch").read_bytes()).hexdigest(), "swap_ab_hoist_patch_sha256": "0" * 64 if corruption == "swap_ab_hoist_hash" else hashlib.sha256((ROOT / "patches/deep-gemm/sm90-decode-swap-ab-hoist.patch").read_bytes()).hexdigest(), "l1_quant_patch_sha256": "0" * 64 if corruption == "l1_quant_hash" else hashlib.sha256((ROOT / "patches/deep-gemm/sm90-decode-swap-ab-l1-quant.patch").read_bytes()).hexdigest(), "dispatch_overlap_patch_sha256": "0" * 64 if corruption == "dispatch_overlap_hash" else hashlib.sha256((ROOT / "patches/deep-gemm/sm90-dispatch-pull-overlap.patch").read_bytes()).hexdigest(), "combine_accumulator": "FP32", "compiled_geometries": geometries}))
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
