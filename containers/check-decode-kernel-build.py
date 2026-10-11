"""Compile the new and existing group quantization specializations without a GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from tvm_ffi.libinfo import find_include_path


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
    source.write_text(f'#include "{header}"\n' + "\n".join(f"template struct PerTokenGroupQuantFlatKernel<__nv_bfloat16,__nv_fp8_e4m3,128,false,true,true,{flags}>;" for flags in specializations))
    subprocess.run([nvcc, "-std=c++20", "-O3", "--use_fast_math", "--expt-relaxed-constexpr", "-DSGL_CUDA_ARCH=900", "-gencode", "arch=compute_90a,code=sm_90a", f"-I{root / 'include'}", f"-I{find_include_path()}", "-c", str(source), "-o", str(Path(temp) / "decode-quant.o")], check=True)
print(json.dumps({"check": "decode_silu_quant", "target": "sm_90a", "specializations": len(specializations), "gpu_execution": False, "status": "PASS"}))
