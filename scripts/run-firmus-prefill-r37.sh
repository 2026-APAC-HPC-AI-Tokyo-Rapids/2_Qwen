#!/bin/bash
set -euo pipefail
root=$1
worker=$2
model=$3
run_id=$4
deadline=$5
set -a
source "${root}/source/prefill-kernels.env"
set +a
export TRITON_CACHE_DIR="/tmp/qwen-r37-${SLURM_JOB_ID}-${worker}/triton"
export DG_JIT_CACHE_DIR="/tmp/qwen-r37-${SLURM_JOB_ID}-${worker}/deepgemm"
export TVM_FFI_CACHE_DIR="/tmp/qwen-r37-${SLURM_JOB_ID}-${worker}/tvm"
exec /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${root}/scripts/select-firmus-prefill-r37.py" --output "${root}/results/prefill-r37/prefill-${worker}" --model-path "${model}" --run-id "${run_id}" --deadline "${deadline}" --image-sha256 "$6"
