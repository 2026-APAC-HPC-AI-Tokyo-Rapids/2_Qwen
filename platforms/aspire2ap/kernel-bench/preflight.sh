#!/usr/bin/env bash
# CPU login-node check of the actual image, package overlays and absolute mounts.
set -euo pipefail
root=$(realpath "${1:?runtime directory required}")
source "${HOME}/.local/.setenv"
export ENROOT_LIBRARY_PATH="${root}/enroot-library" ENROOT_SYSCONF_PATH="${root}/enroot-system" ENROOT_MOUNT_HOME=no NVIDIA_VISIBLE_DEVICES=void
scratch=$(mktemp -d "${TMPDIR:-/tmp}/megamoe-mount-preflight.XXXXXXXX")
trap 'rm -rf -- "${scratch:?}"' EXIT
export ENROOT_CACHE_PATH="${scratch}/cache" ENROOT_DATA_PATH="${scratch}/data" ENROOT_RUNTIME_PATH="${scratch}/runtime" ENROOT_TEMP_PATH="${scratch}/temp"
install -d -m 0700 "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}"
timeout 180 enroot start -e NVIDIA_VISIBLE_DEVICES=void -e HOME=/runtime-home -m "${root}/home:/runtime-home" -m "${root}:${root}:none:bind,ro,x-create=dir" "${root}/sglang.sqsh" /opt/sglang/venv/bin/python - "${root}" <<'PY'
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "void"
for name in ("torch", "triton", "tvm_ffi"):
    assert importlib.util.find_spec(name) is not None, name
config = json.loads((root / "results/variants.json").read_text())
host = json.loads((root / "image/opt/apac/deep-gemm-host-build.json").read_text())
for variant in config["deep_gemm"]:
    package_root = root / ("image/opt/sglang/venv/lib/python3.12/site-packages" if variant["name"] == "image" else "overlay/candidates/" + variant["name"])
    spec = importlib.machinery.PathFinder.find_spec("deep_gemm", [str(package_root)])
    assert spec is not None and spec.origin == str(package_root / "deep_gemm/__init__.py"), (variant["name"], spec)
    extension = package_root / "deep_gemm/_C.so"
    assert hashlib.sha256(extension.read_bytes()).hexdigest() == host["extension_sha256"]
    assert (package_root / "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh").is_file()
assert (root / "scripts/run-megamoe-kernel-bench.sh").is_file()
assert Path("/runtime-home").is_dir()
print("aspire_ep4_cpu_preflight=pass gpu_used=false", flush=True)
PY
