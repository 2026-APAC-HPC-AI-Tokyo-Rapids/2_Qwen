"""Validate the pinned serving packages and Enroot environment without a GPU."""

import hashlib
import json
import os
import runpy
import shutil
import subprocess
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
