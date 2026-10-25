#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${CUDA_HOME:=/usr/local/cuda}"
: "${DEEP_GEMM_INCLUDE:=/opt/sglang/venv/lib/python3.12/site-packages/deep_gemm/include}"
"${CUDA_HOME}/bin/nvcc" -std=c++20 -O3 --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_90a,code=sm_90a -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -Xptxas=-v,--warn-on-spills -Xcompiler=-fPIC --shared --cudart static -I"${DEEP_GEMM_INCLUDE}" "${root}/dense.cu" -o "${root}/dense-dg1.so"
