"""Validate both committed Dynamo backend variants without allocating GPUs."""
import json
import os
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch

config = json.loads(Path("/opt/apac/firmus-dynamo.json").read_text())
backend = config["framework"]
expected = {"ai-dynamo": config["dynamo"]["version"], "ai-dynamo-runtime": config["dynamo"]["version"], backend: config["dynamo"]["engine_version"], "nixl": config["dynamo"]["nixl_version"]}
actual = {package: version(package) for package in expected}
assert actual == expected, actual
assert torch.version.cuda == "13.0", torch.version.cuda
assert os.environ["DYNAMO_COMMIT_SHA"] == config["dynamo"]["commit"]
assert shutil.which("etcd"), "official runtime must supply etcd"
for module in ("dynamo.frontend", f"dynamo.{backend}"):
    subprocess.run(["python3", "-m", module, "--help"], check=True)
environment = Path("/etc/environment").read_text()
assert f'PATH="{os.environ["PATH"]}"' in environment
assert f'LD_LIBRARY_PATH="{os.environ["LD_LIBRARY_PATH"]}"' in environment
manifest = json.loads(Path("/opt/apac/manifest.json").read_text())
assert manifest["base_oci"]["reference"] == config["dynamo"]["image"]
print(json.dumps({"packages": actual, "torch": torch.__version__, "cuda": torch.version.cuda, "dynamo_commit": config["dynamo"]["commit"]}))
