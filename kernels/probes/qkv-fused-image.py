"""Record the actual Genkai probe image's baked patches versus this workspace."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def compare(manifest, workspace):
    sources = manifest["source_commits"]
    qwen = next(s["oid"] for s in sources if s["repository_id"] == "apac2026-qwen")
    sglang = next(s["oid"] for s in sources if s["repository_id"] == "apac2026-sglang")
    assert qwen.startswith("6c6391f"), qwen
    assert sglang.startswith("fc275674"), sglang
    records = []
    for entry in manifest["applied_patches"]:
        path = workspace / entry["path"]
        current = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        records.append({"path": entry["path"], "image_sha256": entry["sha256"], "workspace_sha256": current, "same": current == entry["sha256"]})
    recipe = manifest["recipe"]
    recipes = []
    for entry in [{"path": recipe["path"], "blob_oid": recipe["blob_oid"]}, *recipe["build_steps"]]:
        oid = subprocess.check_output(["git", "hash-object", str(workspace / entry["path"])], text=True).strip()
        recipes.append({"path": entry["path"], "image_blob_oid": entry["blob_oid"], "workspace_blob_oid": oid, "same": oid == entry["blob_oid"]})
    series = (workspace / "patches/deep-gemm-host/series").read_text().splitlines()
    baked_hosts = [x["path"] for x in records if x["path"].startswith("patches/deep-gemm-host/")]
    added_hosts = ["patches/deep-gemm-host/" + x for x in series if "patches/deep-gemm-host/" + x not in baked_hosts]
    # All 23 baked SGLang patches must match: the rope writer/JIT base is unchanged.
    baked_sglang = [r for r in records if r["path"].startswith("patches/sglang/")]
    assert len(baked_sglang) == 23 and all(r["same"] for r in baked_sglang), baked_sglang
    return {"image_qwen": qwen, "image_sglang": sglang, "patches": records, "recipes": recipes, "workspace_only_host_patches": added_hosts}


if __name__ == "__main__":
    print(json.dumps(compare(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])), sort_keys=True), flush=True)
