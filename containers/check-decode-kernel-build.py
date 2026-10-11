"""Compile the new and existing group quantization specializations without a GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import tvm_ffi
from tvm_ffi.libinfo import find_include_path, find_libtvm_ffi


root = Path("/opt/sglang/source/python/sglang/kernels/jit")
header = root / "csrc/gemm/per_token_group_quant.cuh"
nvcc = shutil.which("nvcc") or str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc")
specializations = [
    "true,true,true",   # fused, PDL, FP32 SiLU*up with one BF16 rounding
    "true,false,true",  # same math without PDL
    "false,true",      # unchanged quant-only API
    "true,true",       # unchanged legacy fused rounding
]
with tempfile.TemporaryDirectory() as temp:
    source = Path(temp) / "decode-quant.cu"
    source.write_text(f'#include "{header}"\n' + "\n".join(f"TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_{i}, (PerTokenGroupQuantFlatKernel<__nv_bfloat16,__nv_fp8_e4m3,128,false,true,true,{flags}>::run));" for i, flags in enumerate(specializations)))
    library = Path(temp) / "decode-quant.so"
    ffi_library = Path(find_libtvm_ffi())
    subprocess.run([nvcc, "-std=c++20", "-O3", "--use_fast_math", "--expt-relaxed-constexpr", "-DSGL_CUDA_ARCH=900", "-gencode", "arch=compute_90a,code=sm_90a", f"-I{root / 'include'}", f"-I{find_include_path()}", "-shared", "-Xcompiler=-fPIC", str(source), str(ffi_library), f"-Xlinker=-rpath,{ffi_library.parent}", "-o", str(library)], check=True)
    module = tvm_ffi.load_module(str(library))
    for i in range(len(specializations)):
        assert callable(module[f"run_{i}"])
print(json.dumps({"check": "decode_silu_quant", "target": "sm_90a", "specializations": len(specializations), "ffi_load": "PASS", "gpu_execution": False, "status": "PASS"}))
