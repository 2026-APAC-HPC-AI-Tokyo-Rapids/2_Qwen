#!/usr/bin/env bash
# Run the Qwen 3.5 benchmark on a Genkai compute node.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 RUNTIME_DIR" >&2
  exit 2
fi
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATASET_PATH:?DATASET_PATH is required}"

runtime_dir="$1"
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
results_dir="${runtime_dir}/results"
sif_path="${runtime_dir}/sglang.sif"
singularity=/fast/pj26000080/sglang-tools/singularity-ce-4.1.3-no-xattrs/bin/singularity
jit_cache="${HOME}/.cache/sglang-jit"

mkdir -p "${results_dir}" "${jit_cache}/deep-gemm" "${jit_cache}/torchinductor" "${jit_cache}/triton"

module purge
module load cuda/12.8.1
singularity_tmpdir="$(mktemp -d /dev/shm/sglang-run.XXXXXXXXXX)"
cleanup() {
  rm -rf -- "${singularity_tmpdir:?}"
}
trap cleanup EXIT
export SINGULARITY_TMPDIR="${singularity_tmpdir}"

{
  echo "job_id=${PJM_JOBID:-unknown}"
  echo "started_at=$(date -u +%FT%TZ)"
  echo "model_path=${MODEL_PATH}"
  echo "dataset_path=${DATASET_PATH}"
  "${singularity}" inspect "${sif_path}"
  module list 2>&1
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
  "${singularity}" --version
} >"${results_dir}/environment.log" 2>&1

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
command=("${singularity}" exec --nv --cleanenv --containall
  --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_DATASETS_OFFLINE=1
  --env SGLANG_DG_CACHE_DIR=/jit-cache/deep-gemm,TORCHINDUCTOR_CACHE_DIR=/jit-cache/torchinductor,TRITON_CACHE_DIR=/jit-cache/triton
  --bind "${MODEL_PATH}:/model:ro,${DATASET_PATH}:/dataset/ShareGPT_V3_unfiltered_cleaned_split.json:ro,${jit_cache}:/jit-cache,${results_dir}:/output"
  "${sif_path}" "${benchmark[@]}")

{
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} >"${results_dir}/throughput.log"

set +e
"${script_dir}/exec-profiler.sh" "${command[@]}" 2>&1 | tee -a "${results_dir}/throughput.log"
status=${PIPESTATUS[0]}
set -e
printf 'exit_code=%d\n' "${status}" | tee -a "${results_dir}/throughput.log"
echo "finished_at=$(date -u +%FT%TZ)" >>"${results_dir}/environment.log"

if [[ "${status}" -eq 0 ]]; then
  touch "${runtime_dir}/run-success"
fi
exit "${status}"
