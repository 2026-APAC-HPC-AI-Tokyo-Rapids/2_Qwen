#!/usr/bin/env bash
# Run one Qwen 3.5 benchmark from a provisioned SGLang runtime.
set -euo pipefail

runtime_dir="${1:?usage: run-qwen35.sh RUNTIME_DIR RUN_DIR}"
run_dir="${2:?usage: run-qwen35.sh RUNTIME_DIR RUN_DIR}"
sif_path="${runtime_dir}/sglang.sif"
dataset_path="$(<"${runtime_dir}/dataset-path")"
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
mode=throughput

: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen 3.5 checkpoint directory}"

mkdir -p "${run_dir}"
exec > >(tee -a "${run_dir}/launcher.log") 2>&1
test -s "${sif_path}"
test -r "${dataset_path}"
jit_cache="${HOME}/.cache/sglang-jit"
singularity=/fast/pj26000080/sglang-tools/singularity-ce-4.1.3-no-xattrs/bin/singularity
mkdir -p "${jit_cache}/deep-gemm" "${jit_cache}/torchinductor" "${jit_cache}/triton"
module purge
module load cuda/12.8.1
test -x "${singularity}"
singularity_tmpdir="$(mktemp -d /var/tmp/sglang-run.XXXXXXXXXX)"
cleanup() {
  rm -rf -- "${singularity_tmpdir:?}"
}
trap cleanup EXIT
export SINGULARITY_TMPDIR="${singularity_tmpdir}"

{
  echo "job_id=${PJM_JOBID:-manual}"
  echo "mode=${mode}"
  echo "started_at=$(date -u +%FT%TZ)"
  echo "model_path=${MODEL_PATH}"
  "${singularity}" inspect "${sif_path}"
  module list 2>&1
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
  "${singularity}" --version
} >"${run_dir}/environment.log" 2>&1

command=(python -m sglang.benchmark.offline_throughput
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

printf 'command=' >"${run_dir}/command.txt"
printf '%q ' "${command[@]}" >>"${run_dir}/command.txt"
printf '\n' >>"${run_dir}/command.txt"

set +e
"${singularity}" exec --nv --cleanenv \
  --containall \
  --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_DATASETS_OFFLINE=1 \
  --env SGLANG_DG_CACHE_DIR=/jit-cache/deep-gemm,TORCHINDUCTOR_CACHE_DIR=/jit-cache/torchinductor,TRITON_CACHE_DIR=/jit-cache/triton \
  --bind "${MODEL_PATH}:/model:ro,${dataset_path}:/dataset/ShareGPT_V3_unfiltered_cleaned_split.json:ro,${jit_cache}:/jit-cache,${script_dir}:/workspace/scripts:ro,${run_dir}:/output" \
  "${sif_path}" /opt/sglang/venv/bin/python "${command[@]:1}" 2>&1 | tee "${run_dir}/${mode}.log"
status=${PIPESTATUS[0]}
set -e

printf '%s\n' "${status}" >"${run_dir}/exit-code.txt"
echo "finished_at=$(date -u +%FT%TZ)" >>"${run_dir}/environment.log"
exit "${status}"
