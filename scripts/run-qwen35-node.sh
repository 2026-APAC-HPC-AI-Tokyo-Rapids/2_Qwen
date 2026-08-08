#!/usr/bin/env bash
# Launch one SGLang process per allocated node. Each process controls the four
# GPUs on that Genkai Node Group B node, forming TP=8 across two nodes.
set -euo pipefail

: "${RUNTIME_DIR:?RUNTIME_DIR is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATASET_PATH:?DATASET_PATH is required}"
: "${OMPI_COMM_WORLD_RANK:?OMPI_COMM_WORLD_RANK is required}"

node_rank="${OMPI_COMM_WORLD_RANK}"
[[ "${OMPI_COMM_WORLD_SIZE}" -eq 2 ]]
[[ "${node_rank}" -eq 0 || "${node_rank}" -eq 1 ]]

cd "${RUNTIME_DIR}"
jit_cache="$(mktemp -d /tmp/sglang-jit.XXXXXXXXXX)"
singularity_tmpdir="$(mktemp -d /dev/shm/sglang-run.XXXXXXXXXX)"
export SINGULARITY_TMPDIR="${singularity_tmpdir}"
jit_lock_fd=""
engine_pid=""
gpu_monitor_pid=""
node_status=1

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${engine_pid}" ]]; then
    kill -- "-${engine_pid}" 2>/dev/null || true
    wait "${engine_pid}" 2>/dev/null || true
  fi
  if [[ -n "${gpu_monitor_pid}" ]]; then
    kill "${gpu_monitor_pid}" 2>/dev/null || true
    wait "${gpu_monitor_pid}" 2>/dev/null || true
  fi
  if [[ "${node_status}" -eq 0 && -n "${jit_lock_fd}" ]]; then
    echo "node_rank=${node_rank}: merging local SGLang JIT cache"
    flock -x "${jit_lock_fd}" rsync -a -- "${jit_cache}/" "${HOME}/.cache/sglang-jit/" || status=1
  fi
  rm -rf -- "${singularity_tmpdir:?}" "${jit_cache:?}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec {jit_lock_fd}>>"${HOME}/.cache/sglang-jit.lock"
echo "node_rank=${node_rank}: staging shared SGLang JIT cache in ${jit_cache}"
flock -s "${jit_lock_fd}" rsync -a -- "${HOME}/.cache/sglang-jit/" "${jit_cache}/"

master_host_file="${RUNTIME_DIR}/master-host.txt"
completion_file="${RUNTIME_DIR}/rank-zero-status"
if [[ "${node_rank}" -eq 0 ]]; then
  rm -f -- "${completion_file}"
  hostname > "${master_host_file}"
else
  while [[ ! -s "${master_host_file}" ]]; do
    sleep 0.1
  done
fi
master_host="$(<"${master_host_file}")"

gpu_metrics_file="${RUNTIME_DIR}/results/node-rank-${node_rank}-gpu-metrics.json"
gpu_ready_file="${RUNTIME_DIR}/metrics-socket/node-rank-${node_rank}-gpu.ready"
rm -f -- "${gpu_metrics_file}" "${gpu_ready_file}"
"${RUNTIME_DIR}/qwen-experiment-metrics/.venv/bin/qwen-experiment-metrics" \
  monitor-node-gpu \
  --node-rank "${node_rank}" \
  --output "${gpu_metrics_file}" \
  --ready "${gpu_ready_file}" &
gpu_monitor_pid=$!
while [[ ! -e "${gpu_ready_file}" ]]; do
  kill -0 "${gpu_monitor_pid}"
  sleep 0.1
done

benchmark=(/opt/sglang/venv/bin/python -m sglang.benchmark.offline_throughput
  --model-path /model
  --load-format auto
  --tp-size 8
  --nnodes 2
  --node-rank "${node_rank}"
  --dist-init-addr "${master_host}:25000"
  --dataset-name sharegpt
  --dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json
  --num-prompts 32
  --sharegpt-output-len 256
  --sharegpt-context-len 262144
  --seed 42
  --mem-fraction-static 0.85
  --result-filename /output/results.jsonl)
command=(singularity exec --userns --nv --cleanenv --containall
  --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_DATASETS_OFFLINE=1
  --env SGLANG_DG_CACHE_DIR=/jit-cache/deep-gemm,TORCHINDUCTOR_CACHE_DIR=/jit-cache/torchinductor,TRITON_CACHE_DIR=/jit-cache/triton
  --env QWEN_EXPERIMENT_METRICS_SOCKET=/metrics/events.sock
  --bind "${MODEL_PATH}:/model:ro,${DATASET_PATH}:/dataset/ShareGPT_V3_unfiltered_cleaned_split.json:ro,${jit_cache}:/jit-cache,${jit_cache}:/root/.cache,${RUNTIME_DIR}/results:/output,${RUNTIME_DIR}/metrics-socket:/metrics"
  sglang.sif "${benchmark[@]}")

if [[ "${node_rank}" -eq 0 ]]; then
  set +e
  "${command[@]}"
  node_status=$?
  set -e
  printf '%s\n' "${node_status}" > "${completion_file}"
  exit "${node_status}"
fi

setsid "${command[@]}" >results/node-rank-1.log 2>&1 &
engine_pid=$!
while kill -0 "${engine_pid}" 2>/dev/null; do
  if [[ -s "${completion_file}" ]]; then
    node_status="$(<"${completion_file}")"
    exit "${node_status}"
  fi
  sleep 1
done

engine_status=0
wait "${engine_pid}" || engine_status=$?
engine_pid=""

# The rank-zero benchmark tears down all distributed workers before its
# Singularity process returns. Rank one can therefore observe SIGKILL (137)
# shortly before rank zero publishes the authoritative benchmark status.
for _ in {1..300}; do
  if [[ -s "${completion_file}" ]]; then
    node_status="$(<"${completion_file}")"
    exit "${node_status}"
  fi
  sleep 0.1
done

node_status="${engine_status}"
echo "node_rank=1 exited before node_rank=0 completed (status=${node_status})" >&2
exit "${node_status}"
