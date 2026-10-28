"""The allocation gate must reject stale sources/provenance and failed checks."""
import importlib.util
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check-firmus-pp4-cpu.py"
spec = importlib.util.spec_from_file_location("pp4_cpu_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_exact_tree_and_stale_missing_extra(tmp_path):
    expected, staged = tmp_path / "expected", tmp_path / "staged"
    expected.mkdir()
    staged.mkdir()
    (expected / "source.py").write_text("source")
    (staged / "source.py").write_text("source")
    assert len(gate.compare_sources(expected, staged)) == 64
    (staged / "source.py").write_text("stale")
    with pytest.raises(ValueError, match="differs"):
        gate.compare_sources(expected, staged)
    (staged / "source.py").unlink()
    with pytest.raises(ValueError, match="Empty"):
        gate.compare_sources(expected, staged)
    (staged / "extra.py").write_text("extra")
    with pytest.raises(ValueError, match="differs"):
        gate.compare_sources(expected, staged)


def test_manifest_rejects_wrong_source_and_stale_build_patch(tmp_path):
    manifest = {"schema_version": "oci-embedded-manifest/v1", "source_commits": [{"repository_id": "apac2026-sglang", "oid": "wrong"}], "applied_patches": []}
    with pytest.raises(ValueError, match="revision"):
        gate.verify_manifest(manifest, tmp_path)
    manifest["source_commits"][0]["oid"] = gate.PINNED_SOURCE
    patch_file = tmp_path / "patches/sglang/test.patch"
    patch_file.parent.mkdir(parents=True)
    patch_file.write_text("stale")
    manifest["applied_patches"] = [{"path": "patches/sglang/test.patch", "sha256": "wrong"}]
    with pytest.raises(ValueError, match="build patch differs"):
        gate.verify_manifest(manifest, tmp_path)


def test_regression_failure_stops_remaining_checks(tmp_path):
    # Only the subprocess failure propagation is isolated here. The provision
    # gate itself requires real Torch and executes the source behavior checks.
    torch = pytest.importorskip("torch")
    with patch.object(torch, "__version__", "2.11.0"), patch.object(gate.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "regression")) as run, pytest.raises(subprocess.CalledProcessError):
        gate.run_checks(tmp_path, tmp_path)
    assert run.call_count == 1
    assert run.call_args.kwargs["check"] is True
    assert run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("failure", [None, "stale", "missing", "regression"])
@pytest.mark.parametrize("real_squashfs", [False, True])
def test_provision_gate_before_allocation(tmp_path, monkeypatch, failure, real_squashfs):
    import json
    import sys

    if real_squashfs and not shutil.which("mksquashfs"):
        pytest.skip("squashfs-tools required for image fixture")
    runtime, workspace = tmp_path / "runtime", tmp_path / "workspace"
    (runtime / "source").mkdir(parents=True)
    (runtime / "source/pp4-cpu-gate.json").write_text('{"result":"pass","stale":true}')
    staged = runtime / "overlay/sglang/python/sglang"
    staged.mkdir(parents=True)
    if failure != "missing":
        (staged / "contract.py").write_text("stale" if failure == "stale" else "patched")
    image = runtime / "sglang.sqsh"
    image.write_bytes(b"fixture image")
    monkeypatch.setattr(gate, "PINNED_IMAGE", gate.sha256(image))
    names = [f"patches/sglang/build-{i}.patch" for i in range(23)]
    for name in names + ["patches/sglang/prefill-pp4-ep1.patch"]:
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("diff --git a/python/sglang/contract.py b/python/sglang/contract.py\n--- a/python/sglang/contract.py\n+++ b/python/sglang/contract.py\n@@ -1 +1 @@\n-base\n+patched\n" if name.endswith("prefill-pp4-ep1.patch") else "fixture patch")
    config = runtime / "config.json"
    config.write_text(json.dumps({"images": {"sglang": {"sha256": gate.PINNED_IMAGE}}, "sglang_overlay_patches": ["patches/sglang/prefill-pp4-ep1.patch"]}))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(runtime), str(config), str(workspace)])
    manifest = {"schema_version": "oci-embedded-manifest/v1", "source_commits": [{"repository_id": "apac2026-sglang", "oid": gate.PINNED_SOURCE}], "applied_patches": [{"path": name, "sha256": gate.sha256(workspace / name)} for name in names]}

    def command(argv, **kwargs):
        if argv[0] == "unsquashfs":
            extracted = Path(argv[argv.index("-d") + 1])
            package = extracted / "opt/sglang/source/python/sglang"
            package.mkdir(parents=True)
            (package / "contract.py").write_text("patched")
            (extracted / "opt/apac").mkdir(parents=True)
            (extracted / "opt/apac/manifest.json").write_text(json.dumps(manifest))
        return subprocess.CompletedProcess(argv, 0)

    if real_squashfs:
        image_root = tmp_path / "image-root"
        package = image_root / "opt/sglang/source/python/sglang"
        package.mkdir(parents=True)
        (package / "contract.py").write_text("base\n")
        (image_root / "opt/apac").mkdir(parents=True)
        (image_root / "opt/apac/manifest.json").write_text(json.dumps(manifest))
        image.unlink()
        subprocess.run(["mksquashfs", str(image_root), str(image), "-noappend", "-processors", "1", "-no-progress"], check=True, capture_output=True)
        monkeypatch.setattr(gate, "PINNED_IMAGE", gate.sha256(image))
        config.write_text(json.dumps({"images": {"sglang": {"sha256": gate.PINNED_IMAGE}}, "sglang_overlay_patches": ["patches/sglang/prefill-pp4-ep1.patch"]}))
        if failure != "missing":
            (staged / "contract.py").write_text("stale" if failure == "stale" else "patched\n")
    else:
        monkeypatch.setattr(gate.subprocess, "run", command)
    calls = []

    def checks(*args):
        calls.append(args)
        if failure == "regression":
            raise subprocess.CalledProcessError(1, "real-check")
        return {"torch": "fixture"}

    monkeypatch.setattr(gate, "run_checks", checks)
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            gate.main()
        assert not (runtime / "source/pp4-cpu-gate.json").exists()
        assert len(calls) == int(failure == "regression")
    else:
        gate.main()
        assert json.loads((runtime / "source/pp4-cpu-gate.json").read_text())["result"] == "pass"
