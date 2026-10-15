"""Load a pinned provider without publishing top-level deep_gemm."""

import importlib.util
import sys
import types
from pathlib import Path

from contract import verify_runtime_package


def load_provider(root: Path):
    root = root.resolve()
    package = root / "deep_gemm"
    verify_runtime_package(package)
    if str(root) in sys.path or str(package) in sys.path:
        raise ValueError("Private provider root must not be on sys.path")
    name = "apac_sm90.deep_gemm"
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != package / "__init__.py":
            raise ValueError("Private namespace collision")
        return module
    if "apac_sm90" in sys.modules:
        raise ValueError("Unexpected existing apac_sm90 namespace")
    parent = types.ModuleType("apac_sm90")
    parent.__path__ = [str(root)]
    sys.modules["apac_sm90"] = parent
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
