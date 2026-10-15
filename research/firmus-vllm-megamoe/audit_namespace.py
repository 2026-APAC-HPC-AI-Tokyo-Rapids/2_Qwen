"""Hash-guarded CPU Python wiring audit; never loads native CUDA code.

The wheel stays outside sys.path. Real pinned Python modules are imported under
apac_sm90.deep_gemm with explicit Torch/TVM stubs. Native ABI, GPU math,
symmetric memory and performance remain separate build/GPU gates.
"""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import types
import zipfile
from pathlib import Path

WHEEL_SHA256 = "e5e9f92cdf0515fe878bd81131bc25d2984e00c005824d2932af01a5e0d75261"
VLLM_SHA256 = "5401bbf2556f072f210817888c3e13918773b6b91fdcb67870a235cf205fb73e"
ORIGINAL_HEADER_SHA256 = (
    "a53d83cad76109588b3d5053be02877a0c2a47c38b27472269dc1319df62676d"
)
SOURCES = {
    "__init__.py": "b33e89deacdce241f01f5070d321918f5e5480e3e6d3af569678d4192db4f2a7",
    "cuda_helpers.py": "6afbea22f11214c1ff2431877ccff1f1b79b115c78d64f423df9681686700c5c",
    "mega/__init__.py": "a3fa9f4ab614df8af29a0fcd8e1571e9e05cb1e9f2c02bbb77f4c730264d0339",
    "utils/__init__.py": "bf31f6e9fc30482ef83a73d63fdb6ed4d46deec847b35c0f9d39816d9b923bb4",
    "utils/dist.py": "acf0fc25aa44ca5a54f2f757a39d462f43cf4ea1585d49f3feb59deecffefee3",
    "utils/layout.py": "36dc5ef744c613944c5f355ddd5cfd52ca3af9daae020aa15daad308f211d295",
    "utils/math.py": "fccc9c4380f75aec31adeadf04f79625fe8c5ea87c79e0841a2063d91a3773c5",
    "testing/__init__.py": "860e35af472f2e293883f8d06b928058f05fd5c32a4f108e2d774958ee198529",
    "testing/bench.py": "478d694e5fe95cca78a2c6ee53f7436fc906dc4851cf51ab88a5a9b617142332",
    "testing/numeric.py": "de92b73f28acb3c5f560fbb942161c88c7e849e6c52bb38fd20c67b8e4d747c7",
    "testing/utils.py": "20e5e4450cec023dabd8e92e32d9d6b7500991b9264d1068e942999aa52afec0",
}
API = {
    "fp8_mega_moe": [
        "y",
        "l1_weights",
        "l2_weights",
        "sym_buffer",
        "cumulative_local_expert_recv_stats",
        "recipe",
        "activation",
        "activation_clamp",
        "fast_math",
    ],
    "mega_moe_pre_dispatch_sm90": [
        "x",
        "topk_idx",
        "topk_weights",
        "buf_x",
        "buf_x_sf",
        "buf_topk_idx",
        "buf_topk_weights",
        "num_tokens",
        "group_size",
        "routed_scaling_factor",
    ],
    "transform_weights_for_mega_moe_sm90": ["l1_weights", "l2_weights"],
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked(data: bytes, expected: str, label: str) -> bytes:
    actual = digest(data)
    if actual != expected:
        raise ValueError(f"source hash mismatch: {label}: {actual}")
    return data


def provider_function(source: str):
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_import_deep_gemm"
    )
    logger = types.SimpleNamespace(
        debug_once=lambda *_: None,
        info_once=lambda *_: None,
        warning_once=lambda *_: None,
    )
    namespace = {"functools": functools, "importlib": importlib, "logger": logger}
    code = compile(
        ast.Module(body=[function], type_ignores=[]),
        "pinned-vllm-utils-deep_gemm.py",
        "exec",
    )
    exec(code, namespace)  # noqa: S102 - exact hash-guarded native import function
    return namespace["_import_deep_gemm"]


class NativeStub:
    """Record wrapper arguments without importing or executing any native code."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return call


def fake_dependencies(native: NativeStub):
    torch = types.ModuleType("torch")
    torch.__path__ = []
    torch.Tensor = type("Tensor", (), {})
    torch.dtype = type("dtype", (), {})
    torch.float = "float32"
    torch.int = "int32"
    torch.bfloat16 = "bfloat16"
    torch.float8_e4m3fn = "e4m3fn"
    torch.distributed = types.ModuleType("torch.distributed")
    torch.distributed.__path__ = []
    torch.distributed.ProcessGroup = type("ProcessGroup", (), {})
    symm = types.ModuleType("torch.distributed._symmetric_memory")
    torch.distributed._symmetric_memory = symm
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    tvm = types.ModuleType("tvm_ffi")
    tvm.load_module = lambda _: native
    return {
        "torch": torch,
        "torch.distributed": torch.distributed,
        "torch.distributed._symmetric_memory": symm,
        "tvm_ffi": tvm,
    }


def audit(wheel: Path, vllm_source: Path, image_source: Path | None) -> dict:
    checked(wheel.read_bytes(), WHEEL_SHA256, wheel.name)
    vllm = checked(
        vllm_source.read_bytes(), VLLM_SHA256, "vllm/utils/deep_gemm.py"
    ).decode()
    with zipfile.ZipFile(wheel) as archive:
        selected = {
            name: checked(archive.read("deep_gemm/" + name), expected, name)
            for name, expected in SOURCES.items()
        }
        checked(
            archive.read("deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"),
            ORIGINAL_HEADER_SHA256,
            "original SM90 header",
        )
        metadata = archive.read("sgl_deep_gemm-0.1.5.dist-info/METADATA").decode()
        if "Requires-Dist: apache-tvm-ffi==0.1.11" not in metadata:
            raise ValueError("TVM dependency contract changed")
        version = archive.read("deep_gemm/VERSION")
    if image_source:
        for name, expected in SOURCES.items():
            checked((image_source / name).read_bytes(), expected, "image/" + name)
    relative = 0
    for name, data in selected.items():
        for node in ast.walk(ast.parse(data)):
            if isinstance(node, ast.ImportFrom):
                relative += int(node.level > 0)
                if node.level == 0 and (node.module or "").split(".")[0] == "deep_gemm":
                    raise ValueError("absolute package import: " + name)
            if isinstance(node, ast.Import) and any(
                alias.name.split(".")[0] == "deep_gemm" for alias in node.names
            ):
                raise ValueError("absolute package import: " + name)
    # A corrupted byte must be rejected before Python/native module execution.
    try:
        checked(
            selected["__init__.py"] + b"# corruption",
            SOURCES["__init__.py"],
            "negative control",
        )
    except ValueError:
        corruption_rejected = True
    else:
        raise AssertionError("hash guard failed")
    old_modules = dict(sys.modules)
    old_cuda = os.environ.get("CUDA_HOME")
    native = NativeStub()
    try:
        sys.modules.update(fake_dependencies(native))
        with tempfile.TemporaryDirectory(prefix="firmus-megamoe-isolation-") as temp:
            root = Path(temp)
            package = root / "apac_sm90" / "deep_gemm"
            for name, data in selected.items():
                destination = package / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            (package / "VERSION").write_bytes(version)
            # Force the real Python loader down its prebuilt-module branch.
            # TVM is stubbed: this placeholder is never dlopened.
            (package / "_C.so").write_bytes(b"CPU AUDIT PLACEHOLDER; NOT NATIVE CODE")
            os.environ["CUDA_HOME"] = str(root)
            parent = types.ModuleType("apac_sm90")
            parent.__path__ = [str(package.parent)]
            sys.modules["apac_sm90"] = parent
            spec = importlib.util.spec_from_file_location(
                "apac_sm90.deep_gemm",
                package / "__init__.py",
                submodule_search_locations=[str(package)],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            assert module.__version__ == "0.1.5"
            assert "deep_gemm" not in sys.modules
            for name, arguments in API.items():
                assert (
                    list(inspect.signature(getattr(module, name)).parameters)
                    == arguments
                )
            # Exercise the actual pinned Python wrappers, not a rewritten API.
            objects = [object() for _ in range(7)]
            module.mega_moe_pre_dispatch_sm90(
                *objects, num_tokens=192, group_size=128, routed_scaling_factor=1.0
            )
            assert native.calls[-1] == (
                "mega_moe_pre_dispatch_sm90",
                (*objects, 192, 128, 1.0),
                {},
            )
            buf = types.SimpleNamespace(
                buffer=object(),
                handle=types.SimpleNamespace(buffer_ptrs=object()),
                group=types.SimpleNamespace(rank=lambda: 2),
                num_max_tokens_per_rank=192,
                num_experts=128,
                num_topk=8,
            )
            output, w1, s1, w2, s2 = [object() for _ in range(5)]
            module.fp8_mega_moe(output, (w1, s1), (w2, s2), buf)
            assert native.calls[-1] == (
                "fp8_mega_moe",
                (
                    output,
                    w1,
                    s1,
                    w2,
                    s2,
                    None,
                    buf.buffer,
                    buf.handle.buffer_ptrs,
                    2,
                    192,
                    128,
                    8,
                    (128, 128, 128),
                    "swiglu",
                    None,
                    True,
                ),
                {},
            )
            vendored = types.ModuleType("vllm.third_party.deep_gemm")
            vendored.__version__ = "2.8.0"
            sys.modules[vendored.__name__] = vendored
            resolve = provider_function(vllm)
            assert resolve() is vendored
            # Demonstrate the real hazard, then restore private-only resolution.
            external = types.ModuleType("deep_gemm")
            sys.modules["deep_gemm"] = external
            resolve.cache_clear()
            assert resolve() is external
            del sys.modules["deep_gemm"]
            resolve.cache_clear()
            assert resolve() is vendored
            private_modules = sorted(
                name for name in sys.modules if name.startswith("apac_sm90.")
            )
            assert "apac_sm90.deep_gemm.mega" in private_modules
            assert not any(
                name == "deep_gemm" or name.startswith("deep_gemm.")
                for name in sys.modules
            )
    finally:
        for name in list(sys.modules):
            if name not in old_modules:
                del sys.modules[name]
        sys.modules.update(old_modules)
        if old_cuda is None:
            os.environ.pop("CUDA_HOME", None)
        else:
            os.environ["CUDA_HOME"] = old_cuda
    return {
        "schema": "vllm-megamoe-cpu-isolation/v1",
        "passed": True,
        "wheel_sha256": WHEEL_SHA256,
        "source_sha256": SOURCES,
        "vllm_import_source_sha256": VLLM_SHA256,
        "relative_import_statements": relative,
        "image_source_comparison": image_source is not None,
        "hash_corruption_rejected": corruption_rejected,
        "private_module_imports": private_modules,
        "native_wrapper_argument_checks": [
            "mega_moe_pre_dispatch_sm90",
            "fp8_mega_moe",
        ],
        "vendored_provider_before_and_after": "2.8.0 sentinel selected by exact native resolver",
        "top_level_install_negative_control": "external package overrides vendored provider",
        "native_library_loaded": False,
        "cuda_execution": False,
        "limitations": "Real pinned Python files with Torch and TVM stubs. Does not validate native ABI, kernel availability, numerical accuracy, CUDA capture/replay, symmetric memory, or performance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--image-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.wheel, args.vllm_source, args.image_source)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "PASS: pinned Python private-namespace import, API wrappers and native provider resolution; native ABI/GPU gates remain open"
    )


if __name__ == "__main__":
    main()
