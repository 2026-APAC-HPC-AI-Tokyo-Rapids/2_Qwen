"""CPU checks for the public Basic image; no inference modules are patched."""
import ast
import importlib.metadata
import importlib.util
import json
from pathlib import Path


def main():
    assert importlib.metadata.version("sglang") == "0.5.23"
    source = Path(importlib.util.find_spec("deep_gemm").origin).read_text()
    functions = {node.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)}
    assert {"fp8_mega_moe", "mega_moe_pre_dispatch_sm90"} <= functions
    manifest = json.loads(Path("/opt/apac/manifest.json").read_text())
    assert manifest["tier"] == "basic"
    assert manifest["applied_patches"] == []
    packages = {
        name: importlib.metadata.version(name)
        for name in ("sglang", "torch", "sgl-deep-gemm", "sglang-kernel", "nixl")
    }
    print(json.dumps(packages, sort_keys=True))


if __name__ == "__main__":
    main()
