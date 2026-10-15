"""Validate the pinned serving packages and Enroot environment without a GPU."""

import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import torch
import vllm

expected = {"vllm": "0.30.0", "nixl": "1.4.1"}
actual = {package: version(package) for package in expected}
assert actual == expected, actual
assert vllm.__version__ == expected["vllm"], vllm.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
assert shutil.which("nvcc") == "/usr/local/cuda/bin/nvcc"
compiler = subprocess.check_output(["nvcc", "--version"], text=True)
assert "release 13.0" in compiler, compiler
environment = Path("/etc/environment").read_text().splitlines()
assert f'PATH="{os.environ["PATH"]}"' in environment
assert "NVIDIA_VISIBLE_DEVICES=all" in environment
assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in environment
manifest = json.loads(Path("/opt/apac/manifest.json").read_text())
assert manifest["schema_version"] == "oci-embedded-manifest/v1"
assert manifest["recipe"]["path"] == "containers/vllm.Dockerfile"
repair = json.loads(Path("/opt/apac/dense-draft-ep-repair.json").read_text())
assert (
    repair["before_sha256"]
    == "9c642a87264339c0dbc1d0b57330572149df0f9d5e16c6893b9895e50a619e19"
)
speculative_source = Path(vllm.__file__).parent / "config/speculative.py"
assert (
    hashlib.sha256(speculative_source.read_bytes()).hexdigest()
    == repair["after_sha256"]
)
runpy.run_path("/opt/apac/fix-vllm-dense-draft-ep.py")["validate"](
    speculative_source.read_bytes()
)
assert any(
    p["path"] == "containers/fix-vllm-dense-draft-ep.py"
    for p in manifest["applied_patches"]
)
print(
    json.dumps(
        {
            "packages": actual,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvcc": compiler,
        }
    )
)

# The actual native-library imports and three SM90 compile cases ran in the
# Docker build, so their report is embedded in the image rather than host state.
mega_gate = json.loads(Path("/opt/apac/sm90-megamoe-cpu-gate.json").read_text())
assert mega_gate["passed"] and mega_gate["cuda_initialized"] is False
assert mega_gate["ordinary_dense_provider"] == "vllm.third_party.deep_gemm"
assert mega_gate["private_provider"] == "apac_sm90.deep_gemm"
assert len(mega_gate["compiled_geometries"]) == 3
assert mega_gate["cpu_tensor_bridge"] is True
sys.path.insert(0, "/opt/apac/firmus-sm90-megamoe")
try:
    runpy.run_path("/opt/apac/firmus-sm90-megamoe/contract.py")["verify_runtime_package"](
        Path("/opt/apac/sm90-provider/deep_gemm")
    )
finally:
    sys.path.pop(0)
print(json.dumps(mega_gate))
