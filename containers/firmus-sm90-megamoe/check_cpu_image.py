"""Actual native import and CPU CUDA-compilation gate; no GPU execution."""

import argparse
import inspect
import json
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from build_native import check_cpu_bridge
from contract import (
    HEADER,
    PATCHED_HEADER_SHA256,
    VLLM_IMPORT_SHA256,
    WHEEL_SHA256,
    verify,
)
from provider import load_provider


def check(root: Path) -> dict:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Expected pinned Linux x86_64 image")
    expected = {"vllm": "0.30.0", "torch": "2.13.0+cu130", "apache-tvm-ffi": "0.1.11"}
    actual = {name: version(name) for name in expected}
    if actual != expected:
        raise ValueError(f"Unsupported dependency versions: {actual}")
    import torch
    import vllm.utils.deep_gemm as dense

    if torch.cuda.is_initialized():
        raise ValueError("CPU image gate must not initialize CUDA")
    verify(Path(dense.__file__), VLLM_IMPORT_SHA256)
    dense._import_deep_gemm.cache_clear()
    before = dense._import_deep_gemm()
    if (
        before is None
        or before.__name__ != "vllm.third_party.deep_gemm"
        or before.__version__ != "2.8.0"
    ):
        raise ValueError("Expected unchanged vLLM-vendored DeepGEMM2.8.0")
    if "deep_gemm" in sys.modules:
        raise ValueError("Unexpected top-level deep_gemm provider")
    module = load_provider(root)
    check_cpu_bridge(module._C, torch)
    if module.__version__ != "0.1.5":
        raise ValueError("Unsupported private provider version")
    symbols = [
        "fp8_mega_moe",
        "mega_moe_pre_dispatch_sm90",
        "transform_weights_for_mega_moe_sm90",
        "get_symm_buffer_for_mega_moe",
    ]
    for name in symbols:
        if not callable(getattr(module, name, None)):
            raise TypeError("Missing Python API: " + name)
    native_symbols = [
        "fp8_mega_moe",
        "mega_moe_pre_dispatch_sm90",
        "get_symm_buffer_size_for_sm90_mega_moe",
    ]
    for name in native_symbols:
        if not callable(getattr(module._C, name, None)):
            raise TypeError("Missing native API: " + name)
    dense._import_deep_gemm.cache_clear()
    after = dense._import_deep_gemm()
    if after is not before or "deep_gemm" in sys.modules:
        raise ValueError("Private provider changed ordinary vLLM DeepGEMM resolution")
    signature = inspect.signature(module.fp8_mega_moe)
    if signature.parameters["recipe"].default != (128, 128, 128):
        raise ValueError("SM90 original block128 contract changed")
    package = root / "deep_gemm"
    verify(package / HEADER, PATCHED_HEADER_SHA256)
    # Identical finite SM90 template geometries used by Komori's CPU build gate.
    geometries = [(64, 128, 256, True), (128, 128, 256, False), (128, 256, 512, False)]
    with tempfile.TemporaryDirectory(prefix="firmus-sm90-cubin-") as temp:
        for block_m, block_n, threads, swap in geometries:
            source = Path(temp) / f"kernel-{block_m}-{block_n}.cu"
            source.write_text(f"""#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>
extern "C" void* instantiate() {{
    return reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mega_moe_impl<
        4096, 4096, 1536, 128, 8, 8,
        {block_m}, {block_n}, 128, 137472, 274944, 3,
        64, 64, {threads}, 132, 4,
        cute::numeric_limits<float>::infinity(), true, 0,
        false, false, true, false, {str(swap).lower()}>);
}}
""")
            output = source.with_suffix(".cubin")
            subprocess.run(
                [
                    "/usr/local/cuda/bin/nvcc",
                    "--cubin",
                    "--std=c++20",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-O3",
                    "-gencode",
                    "arch=compute_90a,code=sm_90a",
                    "-I" + str(package / "include"),
                    str(source),
                    "-o",
                    str(output),
                ],
                check=True,
            )
            if output.stat().st_size == 0:
                raise ValueError("Empty cubin")
    from check_adapter_cpu import check as check_adapter

    adapter_cpu = check_adapter()
    if torch.cuda.is_initialized():
        raise ValueError("CPU gate unexpectedly initialized CUDA")
    return {
        "schema": "vllm-megamoe-real-cpu-image-gate/v1",
        "passed": True,
        "versions": actual,
        "wheel_sha256": WHEEL_SHA256,
        "header_sha256": PATCHED_HEADER_SHA256,
        "private_provider": module.__name__,
        "ordinary_dense_provider": before.__name__,
        "python_symbols": symbols,
        "native_symbols": native_symbols,
        "compiled_geometries": geometries,
        "cpu_tensor_bridge": True,
        "vllm_adapter_cpu": adapter_cpu,
        "cuda_initialized": False,
        "limitations": "Native import, CPU tensor bridge and SM90 compilation only. CUDA buffer slicing, GPU launch, symmetric memory, math, changed-input graph replay, memory fit and throughput remain unverified.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check(args.provider_root.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
