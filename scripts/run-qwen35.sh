#!/usr/bin/env bash
#PJM -L rscgrp=b-batch
#PJM -L elapse=00:10:00
#PJM -L gpu=1

# Run the Qwen 3.5 benchmark on a Genkai compute node.
set -euo pipefail

: "${RUNTIME_DIR:?RUNTIME_DIR is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATASET_PATH:?DATASET_PATH is required}"

jit_cache="$(mktemp -d /dev/shm/sglang-jit.XXXXXXXXXX)"
jit_lock_fd=""
singularity_tmpdir="$(mktemp -d /dev/shm/sglang-run.XXXXXXXXXX)"

cleanup() {
  local status=$?
  trap - EXIT
  if [[ "${status}" -eq 0 && -n "${jit_lock_fd}" ]]; then
    echo "Merging local SGLang JIT cache back to ${HOME}/.cache/sglang-jit"
    if ! flock -x "${jit_lock_fd}" rsync -a -- "${jit_cache}/" "${HOME}/.cache/sglang-jit/"; then
      echo "Failed to merge the local SGLang JIT cache" >&2
      if [[ "${status}" -eq 0 ]]; then
        status=1
      fi
    fi
  fi
  rm -rf -- "${singularity_tmpdir:?}" "${jit_cache:?}"
  exit "${status}"
}
trap cleanup EXIT

cd "${RUNTIME_DIR}"
mkdir -p results "${HOME}/.cache/sglang-jit"

# The shared filesystem is reliable for storing the cache but occasionally
# fails while SGLang reads it.  Stage a consistent snapshot on the compute
# node instead.  Shared locks keep a writer from modifying that snapshot while
# it is copied; an exclusive lock serializes the later merge back to shared
# storage.
exec {jit_lock_fd}>>"${HOME}/.cache/sglang-jit.lock"
echo "Staging shared SGLang JIT cache in ${jit_cache}"
flock -s "${jit_lock_fd}" rsync -a -- "${HOME}/.cache/sglang-jit/" "${jit_cache}/"

module purge
module load cuda/13.2.0
module load singularity-ce/4.1.3
export SINGULARITY_TMPDIR="${singularity_tmpdir}"

{
  echo "job_id=${PJM_JOBID:-unknown}"
  echo "started_at=$(date -u +%FT%TZ)"
  echo "model_path=${MODEL_PATH}"
  echo "dataset_path=${DATASET_PATH}"
  singularity inspect sglang.sif
  module list 2>&1
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
  singularity --version
} >results/environment.log 2>&1

benchmark=(/opt/sglang/venv/bin/python -m sglang.benchmark.offline_throughput
  --model-path /model
  --load-format auto
  --tp-size 1
  --dataset-name sharegpt
  --dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json
  --num-prompts 32
  --sharegpt-output-len 256
  --sharegpt-context-len 262144
  --seed 42
  --mem-fraction-static 0.85
  --result-filename /output/results.jsonl)
command=(singularity exec --nv --cleanenv --containall
  --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_DATASETS_OFFLINE=1
  --env SGLANG_DG_CACHE_DIR=/jit-cache/deep-gemm,TORCHINDUCTOR_CACHE_DIR=/jit-cache/torchinductor,TRITON_CACHE_DIR=/jit-cache/triton
  --bind "${MODEL_PATH}:/model:ro,${DATASET_PATH}:/dataset/ShareGPT_V3_unfiltered_cleaned_split.json:ro,${jit_cache}:/jit-cache,${RUNTIME_DIR}/results:/output"
  sglang.sif "${benchmark[@]}")

{
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} >results/throughput.log

set +e
scripts/exec-profiler.sh "${command[@]}" 2>&1 | tee -a results/throughput.log
status=${PIPESTATUS[0]}
set -e
printf 'exit_code=%d\n' "${status}" | tee -a results/throughput.log
echo "finished_at=$(date -u +%FT%TZ)" >>results/environment.log

if [[ "${status}" -eq 0 ]]; then
  touch run-success
fi
exit "${status}"
