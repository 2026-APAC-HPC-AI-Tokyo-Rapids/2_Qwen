#!/usr/bin/env bash
# Run both actual output paths on one H200 within the same EP4 allocation.
# The outer Slurm step caps build plus six process trials at 180 seconds.
set -euo pipefail
root=$1
package=opt/sglang/venv/lib/python3.12/site-packages/deep_gemm
header=include/deep_gemm/impls/sm90_fp8_mega_moe.cuh
out="${root}/results/fc2-output"
install -d "${out}"
/usr/local/cuda/bin/nvcc --version
status=0
for variant in fc2-fragment-combine fc2-producer-transpose; do
  echo "native_variant=${variant}"
  /opt/sglang/venv/bin/python "${root}/scripts/build-fc2-output-probe.py" "${root}/image/${package}/${header}" "${root}/overlay/candidates/${variant}/deep_gemm/${header}" "${out}/${variant}.cu"
  if ! /usr/local/cuda/bin/nvcc --std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -O3 --ptxas-options=-v,-warn-spills -gencode arch=compute_90a,code=sm_90a -I"${root}/image/${package}/include" -L/usr/local/cuda/lib64/stubs "${out}/${variant}.cu" -lcuda -o "${root}/scratch/${variant}-output-probe" >"${out}/${variant}-build.log" 2>&1; then cat "${out}/${variant}-build.log"; status=1; continue; fi
  for trial in 1 2 3; do
    echo "native_variant=${variant} trial=${trial}"
    if ! timeout 30 "${root}/scratch/${variant}-output-probe" | tee "${out}/${variant}-${trial}.log"; then status=1; break; fi
  done
done
exit "${status}"
