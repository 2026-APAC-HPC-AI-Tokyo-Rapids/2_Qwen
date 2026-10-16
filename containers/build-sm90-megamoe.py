"""Compile the observed Qwen Prefill and Decode specializations without a GPU."""
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
    pipeline_patch = Path("/opt/apac/sm90-fc2-pipeline.patch")
    subprocess.run(["git", "apply", "--check", str(pipeline_patch)], cwd=package.parent, check=True)
    subprocess.run(["git", "apply", str(pipeline_patch)], cwd=package.parent, check=True)
    prefill_patch = Path("/opt/apac/sm90-prefill-rescale.patch")
    subprocess.run(["git", "apply", "--check", str(prefill_patch)], cwd=package.parent, check=True)
    subprocess.run(["git", "apply", str(prefill_patch)], cwd=package.parent, check=True)
    decode_tiles_patch = Path("/opt/apac/sm90-decode-exact-tiles.patch")
    subprocess.run(["git", "apply", "--check", str(decode_tiles_patch)], cwd=package.parent, check=True)
    subprocess.run(["git", "apply", str(decode_tiles_patch)], cwd=package.parent, check=True)
    prefill_phase_patch = Path("/opt/apac/sm90-prefill-phase.patch")
    subprocess.run(["git", "apply", "--check", str(prefill_phase_patch)], cwd=package.parent, check=True)
    subprocess.run(["git", "apply", str(prefill_phase_patch)], cwd=package.parent, check=True)
    decode_register_quant_patch = Path("/opt/apac/sm90-decode-register-quant.patch")
    subprocess.run(["git", "apply", "--check", str(decode_register_quant_patch)], cwd=package.parent, check=True)
    subprocess.run(["git", "apply", str(decode_register_quant_patch)], cwd=package.parent, check=True)
    decode_scale_patch = Path("/opt/apac/sm90-decode-scale-cache.patch")
    subprocess.run(["git", "apply", "--check", str(decode_scale_patch)], cwd=package.parent, check=True)
    subprocess.run(["git", "apply", str(decode_scale_patch)], cwd=package.parent, check=True)
    # Match the saved 6344.96 baseline kernel symbols, including accumulator
    # reuse, arrival mode, stage count, EP wave and padded pool capacities.
    # Prefill's reuse path is distinct from Decode's pipelined SwapAB path.
    geometries = [
        {"role": "prefill", "block_m": 128, "block_n": 256, "epilogue_threads": 512, "stages": 3, "registers": 112, "reuse_accum": True, "arrival_counter": True, "full_sync": False, "swap_ab": False},
        {"role": "decode", "block_m": 64, "block_n": 128, "epilogue_threads": 256, "stages": 8, "registers": 0, "reuse_accum": False, "arrival_counter": False, "full_sync": True, "swap_ab": True},
    ]
    with tempfile.TemporaryDirectory(prefix="sm90-megamoe-build-") as temp:
        # CUDA's stub library permits loading the wheel ABI on a GPU-free builder.
        Path(temp, "libcuda.so.1").symlink_to("/usr/local/cuda/lib64/stubs/libcuda.so")
        subprocess.run([sys.executable, "-c", "import deep_gemm; assert callable(deep_gemm.fp8_mega_moe); assert callable(deep_gemm.mega_moe_pre_dispatch_sm90); print('PASS: native DeepGEMM ABI import without GPU')"], env={**os.environ, "LD_LIBRARY_PATH": temp + ":" + os.environ.get("LD_LIBRARY_PATH", "")}, check=True)
        for config in geometries:
            source = Path(temp) / f"kernel-{config['role']}.cu"
            source.write_text(f"""#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>
extern "C" void* instantiate() {{
    return reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mega_moe_impl<
        4224, 4096, 1536, 128, 8, 32,
        {config['block_m']}, {config['block_n']}, 128, 141312, 2260992, {config['stages']},
        64, 64, {config['epilogue_threads']}, 132, 4,
        cute::numeric_limits<float>::infinity(), true, {config['registers']},
        {str(config['reuse_accum']).lower()}, {str(config['arrival_counter']).lower()},
        {str(config['full_sync']).lower()}, false, {str(config['swap_ab']).lower()}>);
}}
""")
            subprocess.run(["/usr/local/cuda/bin/nvcc", "--cubin", "--std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3", "--ptxas-options=-v,-warn-spills", "-gencode", "arch=compute_90a,code=sm_90a", "-I" + str(package / "include"), str(source), "-o", str(source.with_suffix(".cubin"))], check=True)
            assert source.with_suffix(".cubin").stat().st_size > 0
            print("PASS: CPU CUDA compilation " + json.dumps(config), flush=True)
    result = {"package": "sgl-deep-gemm", "version": "0.1.5", "original_header_sha256": original, "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(), "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(), "weight_scale": "original FP32 block128", "fc2_activation_scale": "FP32 group64", "topk_weight": "after FC2", "combine_accumulator": "FP32", "compiled_geometries": geometries}
    result["fc2_pipeline_patch_sha256"] = hashlib.sha256(pipeline_patch.read_bytes()).hexdigest()
    result["prefill_rescale_patch_sha256"] = hashlib.sha256(prefill_patch.read_bytes()).hexdigest()
    result["decode_tiles_patch_sha256"] = hashlib.sha256(decode_tiles_patch.read_bytes()).hexdigest()
    result["decode_scale_patch_sha256"] = hashlib.sha256(decode_scale_patch.read_bytes()).hexdigest()
    result["decode_register_quant_patch_sha256"] = hashlib.sha256(decode_register_quant_patch.read_bytes()).hexdigest()
    result["prefill_phase_patch_sha256"] = hashlib.sha256(prefill_phase_patch.read_bytes()).hexdigest()
    Path("/opt/apac/sm90-megamoe-build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
