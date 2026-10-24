#!/usr/bin/env bash
set -euo pipefail
source "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
cp "${GITHUB_WORKSPACE}/kernels/probes/qkv-fusion.cu" "${GITHUB_WORKSPACE}/scripts/check-firmus-decode-kernels.py" .
python3 "${GITHUB_WORKSPACE}/kernels/probes/build-qkv-fusion.py" deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh qkv-fused.cuh
module load cuda/12.9.1
# Match the deployed DeepGEMM JIT register allocator setting.
nvcc --std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -O3 --ptxas-options=-v,-warn-spills,--register-usage-level=10 -ccbin /opt/rh/gcc-toolset-13/root/usr/bin/g++ -gencode arch=compute_90a,code=sm_90a -Ideep_gemm/include -L"$(dirname "$(command -v nvcc)")/../lib64/stubs" --shared --cudart static -Xcompiler -fPIC qkv-fusion.cu -lcuda -o qkv-fusion.so >"${RUNTIME_DIR}/logs/build.log" 2>&1 || { cat "${RUNTIME_DIR}/logs/build.log"; exit 1; }
cat "${RUNTIME_DIR}/logs/build.log"
if grep -Eq '[1-9][0-9]* bytes spill (stores|loads)' "${RUNTIME_DIR}/logs/build.log"; then echo 'Reject spilling kernels before allocating a GPU'; exit 1; fi
cp "${GITHUB_WORKSPACE}/kernels/probes/qkv-fusion.py" bench.py
jq -n --arg commit "${GITHUB_SHA}" --arg image "${image}" '{commit:$commit,image:$image,kernel:"QKV GEMM epilogue fusion",scope:"one GPU, QKV and postprocessing only, no serving"}' >provenance.json
