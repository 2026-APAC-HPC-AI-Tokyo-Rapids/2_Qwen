"""Validate the pinned serving packages and Enroot environment without a GPU."""

import json
import os
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
print(json.dumps({"packages": actual, "torch": torch.__version__, "cuda": torch.version.cuda, "nvcc": compiler}))
