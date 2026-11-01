"""Mandatory pre-allocation PP4 CPU gate against the actual staged image source.

Run with a CPU-capable Python containing torch==2.11.0 and numpy:
  python scripts/check-firmus-pp4-cpu.py RUNTIME CONFIG WORKSPACE
No SGLang import or CUDA initialization is needed. NCCL/GPU numerical parity,
actual NIXL transfer and full-model execution remain outside this CPU gate.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PINNED_SOURCE = "fc275674a97d78553b22597ee72c8fd48f5420f4"
PINNED_IMAGE = "149e0fc9d1f71733a25ddba2939f3373db8e33740c29575e4bd33d0fcc82fa3f"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_hashes(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            # The pinned image contains relative .clang-format links whose
            # targets lie outside the extracted Python package. Compare link
            # payloads verbatim without following them, including dangling links.
            result[str(path.relative_to(root))] = {"type": "symlink", "target": os.readlink(path)}
        elif path.is_file():
            result[str(path.relative_to(root))] = {"type": "file", "sha256": sha256(path)}
    if not result:
        raise ValueError(f"Empty source tree: {root}")
    return result


def compare_sources(expected, staged):
    expected_hashes, staged_hashes = tree_hashes(expected), tree_hashes(staged)
    if expected_hashes != staged_hashes:
        changed = sorted(k for k in expected_hashes.keys() | staged_hashes.keys() if expected_hashes.get(k) != staged_hashes.get(k))
        raise ValueError(f"Staged SGLang differs from exact ordered overlays: {changed}")
    return hashlib.sha256(json.dumps(expected_hashes, sort_keys=True).encode()).hexdigest()


def verify_manifest(manifest, workspace):
    if manifest.get("schema_version") != "oci-embedded-manifest/v1":
        raise ValueError("Unsupported image source manifest schema")
    sources = [s for s in manifest["source_commits"] if s["repository_id"] == "apac2026-sglang"]
    if len(sources) != 1 or sources[0]["oid"] != PINNED_SOURCE:
        raise ValueError("Unexpected image SGLang source revision")
    build_patches = []
    for patch in manifest["applied_patches"]:
        if patch["path"].startswith("patches/sglang/"):
            if sha256(workspace / patch["path"]) != patch["sha256"]:
                raise ValueError(f"Image build patch differs from workspace: {patch['path']}")
            build_patches.append(patch["path"])
    if len(build_patches) != 23:
        raise ValueError(f"Expected pinned image's 23 SGLang build overlays, got {len(build_patches)}")
    return build_patches


def run_checks(source, workspace):
    import numpy
    import torch

    if torch.__version__.split("+", 1)[0] != "2.11.0":
        raise ValueError(f"Expected image Torch 2.11.0 contract, got {torch.__version__}")
    # A real CPU operation prevents a fixture or import-only check passing here.
    torch.testing.assert_close(torch.arange(4, device="cpu") + 1, torch.tensor([1, 2, 3, 4]))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    for script, argument in (
        ("check-firmus-pp4-source.py", source),
        ("check-pp4-scheduler-sync.py", source / "python/sglang/srt/managers/scheduler_components/dp_attn.py"),
        ("check-pp4-output-relay.py", source),
        ("check-pp4-direct-loop.py", source),
    ):
        subprocess.run([sys.executable, str(workspace / "scripts" / script), str(argument)], check=True, env=env)
    return {"torch": torch.__version__, "numpy": numpy.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    runtime, workspace = args.runtime.resolve(), args.workspace.resolve()
    report_path = runtime / "source/pp4-cpu-gate.json"
    report_path.unlink(missing_ok=True)
    config = json.loads(args.config.read_text())
    image_sha = config["images"]["sglang"]["sha256"]
    if config.get("container_sha256", image_sha) != image_sha:
        raise ValueError("PP4 config pin differs from the provisioned image")
    if image_sha != PINNED_IMAGE:
        # A [require-build] run serves the image built from the commit being provisioned;
        # provision copied its sidecar and rewrote the config pin to it.
        sidecar = json.loads((runtime / "source/container-sidecar.json").read_text())
        if sidecar["container_artifact"]["sha256"] != image_sha or sidecar["source_commits"]["qwen"] != os.environ["GITHUB_SHA"]:
            raise ValueError("PP4 CPU gate requires the pinned successful native image or the image built from this commit")
    image = runtime / "sglang.sqsh"
    if sha256(image) != image_sha:
        raise ValueError("Actual image SHA-256 differs from PP4 config")
    patches = config["sglang_overlay_patches"]
    if "patches/sglang/prefill-pp4-ep1.patch" not in patches or len(patches) != len(set(patches)):
        raise ValueError("Missing PP4 overlay or duplicate runtime overlay")
    # Reconstruct from the actual immutable image, never from a convenient local
    # checkout. Config order is exactly stage-qwen-overlays.sh's application order.
    with tempfile.TemporaryDirectory(prefix="pp4-cpu-", dir=runtime) as temporary:
        extracted = Path(temporary) / "image"
        subprocess.run(["unsquashfs", "-no-progress", "-no-xattrs", "-processors", "1", "-d", str(extracted), str(image), "opt/sglang/source/python/sglang", "opt/apac/manifest.json"], check=True)
        manifest = json.loads((extracted / "opt/apac/manifest.json").read_text())
        build_patches = verify_manifest(manifest, workspace)
        source = Path(temporary) / "source"
        (source / "python").mkdir(parents=True)
        (extracted / "opt/sglang/source/python/sglang").rename(source / "python/sglang")
        patch_hashes = []
        for name in patches:
            patch = (workspace / name).resolve()
            if not patch.is_relative_to(workspace / "patches/sglang"):
                raise ValueError(f"Unexpected SGLang runtime overlay path: {name}")
            subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
            subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
            patch_hashes.append({"path": name, "sha256": sha256(patch)})
        source_hash = compare_sources(source / "python/sglang", runtime / "overlay/sglang/python/sglang")
        versions = run_checks(source, workspace)
        report = {"schema": "pp4-cpu-gate/v1", "image_sha256": image_sha, "sglang_commit": PINNED_SOURCE, "build_patches": build_patches, "runtime_patches": patch_hashes, "source_tree_sha256": source_hash, "versions": versions, "result": "pass", "limits": "CPU source contracts; no GPU, NCCL, full-model numerical parity or actual NIXL transfer guarantee"}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
