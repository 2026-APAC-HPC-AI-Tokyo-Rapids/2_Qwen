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
header = root / "csrc/elementwise/rope.cuh"
nvcc = shutil.which("nvcc") or str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc")
specializations = [
    "FusedRopeKernel<true,128,true,bf16_t>::run_fp8",
    "FusedRopeKernel<true,128,false,bf16_t>::run_fp8",
    "FusedRopeKernel<true,128,true,bf16_t>::run",
    "FusedRopeKernel<true,128,true,bf16_t>::run_fused",
    "FusedRopeKernel<false,128,true,bf16_t>::run",
    "FusedRopeKernel<true,128,true,bf16_t,fp16_t>::run",
]
with tempfile.TemporaryDirectory() as temp:
    source = Path(temp) / "rope.cu"
    source.write_text(f'#include "{header}"\n' + "\n".join(f"TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_{i}, ({kernel}));" for i, kernel in enumerate(specializations)))
    library = Path(temp) / "rope.so"
    ffi_library = Path(find_libtvm_ffi())
    # Match the JIT flags: do not enable fast_math for RoPE.
    subprocess.run([nvcc, "-std=c++20", "-O3", "--expt-relaxed-constexpr", "--ptxas-options=-v,-warn-spills", "-DSGL_CUDA_ARCH=900", "-gencode", "arch=compute_90a,code=sm_90a", f"-I{root / 'include'}", f"-I{find_include_path()}", "-shared", "-Xcompiler=-fPIC", str(source), str(ffi_library), f"-Xlinker=-rpath,{ffi_library.parent}", "-o", str(library)], check=True)
    module = tvm_ffi.load_module(str(library))
    for i in range(len(specializations)):
        assert callable(module[f"run_{i}"])
print(json.dumps({"check": "rope_fp8_q", "target": "sm_90a", "specializations": len(specializations), "ffi_load": "PASS", "gpu_execution": False, "status": "PASS"}))
