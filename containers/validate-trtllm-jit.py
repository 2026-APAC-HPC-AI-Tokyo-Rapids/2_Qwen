"""Compile the TRT-LLM DeepGEMM header preamble with NVRTC, without a GPU."""

import ctypes as c
import subprocess
from pathlib import Path

# Match the pinned C++ JIT resolver, including its dependency on the pip executable.
metadata = subprocess.check_output(["pip", "show", "tensorrt_llm"], text=True)
location = next(line.removeprefix("Location: ") for line in metadata.splitlines() if line.startswith("Location: "))
include = Path(location) / "tensorrt_llm/include"
assert (include / "deep_gemm/fp8_gemm_impl.cuh").is_file()
assert (include / "cuda_fp8.h").is_file()

nvrtc = c.CDLL("/usr/local/cuda/lib64/libnvrtc.so")
nvrtc.nvrtcCreateProgram.argtypes = [c.POINTER(c.c_void_p), c.c_char_p, c.c_char_p, c.c_int, c.c_void_p, c.c_void_p]
nvrtc.nvrtcCompileProgram.argtypes = [c.c_void_p, c.c_int, c.POINTER(c.c_char_p)]
nvrtc.nvrtcGetProgramLogSize.argtypes = [c.c_void_p, c.POINTER(c.c_size_t)]
nvrtc.nvrtcGetProgramLog.argtypes = [c.c_void_p, c.c_char_p]
nvrtc.nvrtcGetCUBINSize.argtypes = [c.c_void_p, c.POINTER(c.c_size_t)]
nvrtc.nvrtcDestroyProgram.argtypes = [c.POINTER(c.c_void_p)]
source = b"""
#define NVRTC_JIT_COMPILATION
#include <deep_gemm/nvrtc_std.cuh>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <deep_gemm/nvrtc_cutlass.cuh>
#include <deep_gemm/fp8_gemm_impl.cuh>
__global__ void header_smoke() {}
"""
flags = ["-std=c++17", "--gpu-architecture=sm_90a", f"-I{include}", "-D__FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__=1", "-D__FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__=1"]
options = (c.c_char_p * len(flags))(*(flag.encode() for flag in flags))
program = c.c_void_p()
assert nvrtc.nvrtcCreateProgram(c.byref(program), source, b"header_smoke.cu", 0, None, None) == 0
try:
    status = nvrtc.nvrtcCompileProgram(program, len(flags), options)
    log_size = c.c_size_t()
    assert nvrtc.nvrtcGetProgramLogSize(program, c.byref(log_size)) == 0
    log = c.create_string_buffer(log_size.value)
    assert nvrtc.nvrtcGetProgramLog(program, log) == 0
    assert status == 0, f"NVRTC status={status}: {log.value.decode()}"
    cubin_size = c.c_size_t()
    assert nvrtc.nvrtcGetCUBINSize(program, c.byref(cubin_size)) == 0
    assert cubin_size.value > 0
    print(f"TRT-LLM pip/header/NVRTC CPU check passed: {cubin_size.value} cubin bytes")
finally:
    assert nvrtc.nvrtcDestroyProgram(c.byref(program)) == 0
