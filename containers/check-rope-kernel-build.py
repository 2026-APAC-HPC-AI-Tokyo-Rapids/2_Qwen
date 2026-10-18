"""Compile the FP8 Q RoPE path and existing APIs on a CPU-only SM90 builder."""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import tvm_ffi
from tvm_ffi.libinfo import find_include_path, find_libtvm_ffi

root = Path("/opt/sglang/source/python/sglang/kernels/jit")
rope_header = root / "csrc/elementwise/rope.cuh"
nvcc = shutil.which("nvcc") or str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc")
specializations = [
    "FusedRopeKernel<true,128,true,bf16_t>::run_norm_fp8_store",
    "FusedRopeKernel<true,128,false,bf16_t>::run_norm_fp8_store",
    "FusedRopeKernel<true,128,true,bf16_t>::run_norm_fp8",
    "FusedRopeKernel<true,128,false,bf16_t>::run_norm_fp8",
    "FusedRopeKernel<true,128,true,bf16_t>::run_fp8",
    "FusedRopeKernel<true,128,false,bf16_t>::run_fp8",
    "FusedRopeKernel<true,128,true,bf16_t>::run",
    "FusedRopeKernel<true,128,true,bf16_t>::run_fused",
    "FusedRopeKernel<false,128,true,bf16_t>::run",
    "FusedRopeKernel<true,128,true,bf16_t,fp16_t>::run",
]
for name, header, kernels, flags in (
    ("rope_fp8_q_kv", rope_header, specializations, []),
    ("topk_padding", root / "csrc/trtllm_lora_temp/topk_softmax_pack.cuh", ["topk_softmax_masked", "topk_softmax_pack"], ["--use_fast_math"]),
):
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / f"{name}.cu"
        source.write_text(f'#include "{header}"\n' + "\n".join(f"TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_{i}, ({kernel}));" for i, kernel in enumerate(kernels)))
        library = Path(temp) / f"{name}.so"
        ffi_library = Path(find_libtvm_ffi())
        # Match each JIT's flags: RoPE has no fast_math; AOT-equivalent Top-K does.
        subprocess.run([nvcc, "-std=c++20", "-O3", "--expt-relaxed-constexpr", "--ptxas-options=-v,-warn-spills", "-DSGL_CUDA_ARCH=900", "-gencode", "arch=compute_90a,code=sm_90a", f"-I{root / 'include'}", f"-I{find_include_path()}", "-shared", "-Xcompiler=-fPIC", *flags, str(source), str(ffi_library), f"-Xlinker=-rpath,{ffi_library.parent}", "-o", str(library)], check=True)
        module = tvm_ffi.load_module(str(library))
        for i in range(len(kernels)):
            assert callable(module[f"run_{i}"])
    print(json.dumps({"check": name, "target": "sm_90a", "specializations": len(kernels), "ffi_load": "PASS", "gpu_execution": False, "status": "PASS"}))
