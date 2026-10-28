#!/bin/bash
# Pre-serving Decode kernel selection on four H200s, inside the SGLang container
# with the provisioning overlays mounted. FA3 runs on the head node's spare
# four GPUs during the numeric gate; dense runs during Prefill model startup.
# Rejected or failed candidates leave the worker on its existing kernels.
set -uo pipefail
root=$1
part=$2
out="${root}/results/decode-selection"
mkdir -p "${out}"
torchrun=/opt/sglang/venv/bin/torchrun
: >"${out}/${part}.env"
if [[ "${part}" == dense ]]; then
  scratch="/tmp/qwen-decode-select-${SLURM_JOB_ID}-${part}"
  mkdir -p "${scratch}"
  export TRITON_CACHE_DIR="${scratch}/triton" DG_JIT_CACHE_DIR="${scratch}/deep-gemm" SGLANG_DG_CACHE_DIR="${scratch}/deep-gemm"
  timeout 420 "${torchrun}" --standalone --nproc-per-node=4 "${root}/scripts/select-firmus-decode-dense.py" --output "${out}/dense.json" --env-output "${out}/dense-selected.env" >"${out}/dense.log" 2>&1
  dense=$?
  if (( dense == 0 )); then cat "${out}/dense-selected.env" >>"${out}/dense.env"; fi
  printf 'dense=%s\n' "${dense}" >"${out}/dense-status.txt"
  timeout --kill-after=5 260 "${torchrun}" --standalone --nproc-per-node=4 "${root}/scripts/select-firmus-decode-prologue.py" --output "${out}/prologue.json" --env-output "${out}/prologue-selected.env" >"${out}/prologue.log" 2>&1
  prologue=$?
  if (( prologue == 0 )); then cat "${out}/prologue-selected.env" >>"${out}/dense.env"; fi
  printf 'prologue=%s\n' "${prologue}" >"${out}/prologue-status.txt"
elif [[ "${part}" == fa3 ]]; then
  timeout --kill-after=5 175 "${torchrun}" --standalone --nproc-per-node=4 "${root}/scripts/select-firmus-decode-fa3.py" --output "${out}/fa3.json" --env-output "${out}/fa3-selected.env" >"${out}/fa3.log" 2>&1
  fa3=$?
  if (( fa3 == 0 )); then cat "${out}/fa3-selected.env" >>"${out}/fa3.env"; fi
  printf 'fa3=%s\n' "${fa3}" >"${out}/fa3-status.txt"
else
  printf 'unknown selection part: %s\n' "${part}" >&2
  exit 2
fi
cat "${out}/${part}.env"
