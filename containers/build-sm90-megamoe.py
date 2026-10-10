"""Patch the pinned FP8 kernel and compile both epilogues without a GPU."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path


def main():
    assert version("sgl-deep-gemm") == "0.1.5"
    package = Path(importlib.util.find_spec("deep_gemm").origin).parent
    header = package / "include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"
    original = hashlib.sha256(header.read_bytes()).hexdigest()
    assert original == "a53d83cad76109588b3d5053be02877a0c2a47c38b27472269dc1319df62676d", original
    patch = Path("/opt/apac/sm90-late-topk-fp32.patch")
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=package.parent, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=package.parent, check=True)
    # Include the small-M SwapAB path and the split-M/split-N prefill paths.
    # This compiles CUDA code on the CPU; execution is checked in the one Slurm job.
    geometries = [(64, 128, 256, True), (128, 128, 256, False), (128, 256, 512, False)]
    with tempfile.TemporaryDirectory(prefix="sm90-megamoe-build-") as temp:
        # CUDA's stub library permits loading the wheel ABI on a GPU-free builder.
        Path(temp, "libcuda.so.1").symlink_to("/usr/local/cuda/lib64/stubs/libcuda.so")
        subprocess.run([sys.executable, "-c", "import deep_gemm; assert callable(deep_gemm.fp8_mega_moe); assert callable(deep_gemm.mega_moe_pre_dispatch_sm90); print('PASS: native DeepGEMM ABI import without GPU')"], env={**os.environ, "LD_LIBRARY_PATH": temp + ":" + os.environ.get("LD_LIBRARY_PATH", "")}, check=True)
        for block_m, block_n, epilogue_threads, swap_ab in geometries:
            source = Path(temp) / f"kernel-{block_m}-{block_n}.cu"
            source.write_text(f"""#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>
extern "C" void* instantiate() {{
    return reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mega_moe_impl<
        4096, 4096, 1536, 128, 8, 8,
        {block_m}, {block_n}, 128, 137472, 274944, 3,
        64, 64, {epilogue_threads}, 132, 4,
        cute::numeric_limits<float>::infinity(), true, 0,
        false, false, true, false, {str(swap_ab).lower()}>);
}}
""")
            subprocess.run(["/usr/local/cuda/bin/nvcc", "--cubin", "--std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3", "-gencode", "arch=compute_90a,code=sm_90a", "-I" + str(package / "include"), str(source), "-o", str(source.with_suffix(".cubin"))], check=True)
            assert source.with_suffix(".cubin").stat().st_size > 0
            print(f"PASS: CPU CUDA compilation M{block_m}/N{block_n}, SwapAB={swap_ab}", flush=True)
    result = {"package": "sgl-deep-gemm", "version": "0.1.5", "original_header_sha256": original, "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(), "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(), "weight_scale": "original FP32 block128", "fc2_activation_scale": "FP32 group64", "topk_weight": "after FC2", "combine_accumulator": "FP32", "compiled_geometries": geometries}
    Path("/opt/apac/sm90-megamoe-build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
