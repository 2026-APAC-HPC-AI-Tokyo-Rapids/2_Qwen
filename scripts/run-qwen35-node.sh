#!/usr/bin/env bash
# Launch one SGLang process per node. Observability remains node-local until
# completed fragments and source files are merged by the Actions runner.
set -euo pipefail

: "${RUNTIME_DIR:?RUNTIME_DIR is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATASET_PATH:?DATASET_PATH is required}"
: "${OMPI_COMM_WORLD_RANK:?OMPI_COMM_WORLD_RANK is required}"

node_rank="${OMPI_COMM_WORLD_RANK}"
[[ "${OMPI_COMM_WORLD_SIZE}" -eq 2 ]]
[[ "${node_rank}" -eq 0 || "${node_rank}" -eq 1 ]]

cd "${RUNTIME_DIR}"
metrics_cli="${RUNTIME_DIR}/qwen-experiment-metrics/.venv/bin/qwen-experiment-metrics"
manifest="${RUNTIME_DIR}/control/run-manifest.json"
run_id="$(jq -er '.run_id' "${manifest}")"
run_purpose="$(jq -er '.run_purpose' "${manifest}")"
source_blob_oid="$(singularity exec sglang.sif cat /opt/sglang/offline-throughput-source-blob-oid.txt | tr -d '[:space:]')"
jit_cache="$(mktemp -d /tmp/sglang-jit.XXXXXXXXXX)"
singularity_tmpdir="$(mktemp -d /dev/shm/sglang-run.XXXXXXXXXX)"
node_metrics_dir="$(mktemp -d /dev/shm/qwen-metrics.XXXXXXXXXX)"
export SINGULARITY_TMPDIR="${singularity_tmpdir}"
jit_lock_fd=""
engine_pid=""
event_server_pid=""
gpu_sampler_pid=""
node_status=1

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${engine_pid}" ]]; then
    kill -- "-${engine_pid}" 2>/dev/null || true
    wait "${engine_pid}" 2>/dev/null || true
  fi
  if [[ -n "${gpu_sampler_pid}" ]]; then
    kill -TERM "${gpu_sampler_pid}" 2>/dev/null || true
    wait "${gpu_sampler_pid}" 2>/dev/null || true
  fi
  if [[ -n "${event_server_pid}" ]]; then
    kill -TERM "${event_server_pid}" 2>/dev/null || true
    wait "${event_server_pid}" 2>/dev/null || true
  fi
  if [[ "${node_status}" -eq 0 && -n "${jit_lock_fd}" ]]; then
    echo "node_rank=${node_rank}: merging local SGLang JIT cache"
    flock -x "${jit_lock_fd}" rsync -a -- "${jit_cache}/" "${HOME}/.cache/sglang-jit/" || status=1
  fi
  rm -rf -- "${singularity_tmpdir:?}" "${jit_cache:?}" "${node_metrics_dir:?}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec {jit_lock_fd}>>"${HOME}/.cache/sglang-jit.lock"
echo "node_rank=${node_rank}: staging shared SGLang JIT cache in ${jit_cache}"
flock -s "${jit_lock_fd}" rsync -a -- "${HOME}/.cache/sglang-jit/" "${jit_cache}/"

master_host_file="${RUNTIME_DIR}/control/master-host.txt"
completion_file="${RUNTIME_DIR}/control/rank-zero-status"
if [[ "${node_rank}" -eq 0 ]]; then
  rm -f -- "${completion_file}"
  hostname > "${master_host_file}"
else
  while [[ ! -s "${master_host_file}" ]]; do
    sleep 0.1
  done
fi
master_host="$(<"${master_host_file}")"

topology_path="${RUNTIME_DIR}/source/topology/node-${node_rank}.json"
topology_tmp="${topology_path}.tmp-${BASHPID}"
mkdir -p "${RUNTIME_DIR}/source/topology"
gpu_rows="$(nvidia-smi \
  --query-gpu=index,uuid,pci.bus_id,name,memory.total \
  --format=csv,noheader,nounits | \
  jq -R -s '
    split("\n") | map(select(length > 0)) |
    map(split(",") | map(gsub("^[[:space:]]+|[[:space:]]+$"; "")) |
      {logical_index:(.[0]|tonumber),uuid:.[1],pci_bus_id:.[2],model:.[3],memory_total_mib:(.[4]|tonumber)})')"
jq -n \
  --arg run_id "${run_id}" \
  --argjson node_rank "${node_rank}" \
  --arg hostname "$(hostname)" \
  --argjson gpus "${gpu_rows}" \
  '{schema_version:"topology/v1",run_id:$run_id,node_rank:$node_rank,hostname:$hostname,gpus:$gpus}' \
  >"${topology_tmp}"
mv -- "${topology_tmp}" "${topology_path}"

event_socket="${node_metrics_dir}/events.sock"
event_ready="${node_metrics_dir}/events.ready"
"${metrics_cli}" serve-events \
  --manifest "${manifest}" \
  --runtime-dir "${RUNTIME_DIR}" \
  --socket "${event_socket}" \
  --ready-file "${event_ready}" &
event_server_pid=$!
while [[ ! -e "${event_ready}" ]]; do
  kill -0 "${event_server_pid}"
  sleep 0.1
done

if [[ "${run_purpose}" == profile ]]; then
  gpu_output="${RUNTIME_DIR}/profile/raw/gpu/node-${node_rank}.jsonl"
  gpu_completion="${RUNTIME_DIR}/control/gpu/node-${node_rank}.complete"
  gpu_ready="${node_metrics_dir}/gpu.ready"
  clock_id="$(tr -d '[:space:]' </proc/sys/kernel/random/boot_id):CLOCK_MONOTONIC"
  gpu_args=()
  while IFS=, read -r uuid index; do
    uuid="${uuid//[[:space:]]/}"
    index="${index//[[:space:]]/}"
    gpu_args+=(--expected-gpu "${uuid}=${index}")
  done < <(nvidia-smi --query-gpu=uuid,index --format=csv,noheader,nounits)
  [[ "${#gpu_args[@]}" -eq 4 ]]
  "${metrics_cli}" sample-gpu \
    --manifest "${manifest}" \
    --runtime-dir "${RUNTIME_DIR}" \
    --node-rank "${node_rank}" \
    --clock-id "${clock_id}" \
    --output "${gpu_output}" \
    --ready-file "${gpu_ready}" \
    --completion-file "${gpu_completion}" \
    --interval-sec 1 \
    "${gpu_args[@]}" &
  gpu_sampler_pid=$!
  while [[ ! -e "${gpu_ready}" ]]; do
    kill -0 "${gpu_sampler_pid}"
    sleep 0.1
  done
fi

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
  --mem-fraction-static 0.85)
if [[ "${run_purpose}" == profile ]]; then
  mkdir -p "${RUNTIME_DIR}/profile/raw/torch"
  benchmark+=(--profile --profile-activities CPU GPU --profile-steps 5)
fi

profile_env_args=()
profile_bind_args=()
if [[ "${run_purpose}" == profile ]]; then
  profile_env_args=(--env SGLANG_TORCH_PROFILER_DIR=/profile)
  profile_bind_args=(--bind "${RUNTIME_DIR}/profile/raw/torch:/profile")
fi

command=(singularity exec --userns --nv --cleanenv --containall
  --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_DATASETS_OFFLINE=1
  --env SGLANG_DG_CACHE_DIR=/jit-cache/deep-gemm,TORCHINDUCTOR_CACHE_DIR=/jit-cache/torchinductor,TRITON_CACHE_DIR=/jit-cache/triton
  --env QWEN_EXPERIMENT_METRICS_SOCKET=/metrics/events.sock
  --env QWEN_EXPERIMENT_METRICS_RUN_ID="${run_id}",QWEN_EXPERIMENT_METRICS_NODE_RANK="${node_rank}"
  --env QWEN_EXPERIMENT_METRICS_SOURCE_BLOB_OID="${source_blob_oid}"
  --env QWEN_EXPERIMENT_METRICS_SOURCE_DIR=/source-output
  "${profile_env_args[@]}"
  "${profile_bind_args[@]}"
  --bind "${MODEL_PATH}:/model:ro,${DATASET_PATH}:/dataset/ShareGPT_V3_unfiltered_cleaned_split.json:ro,${jit_cache}:/jit-cache,${jit_cache}:/root/.cache,${node_metrics_dir}:/metrics,${RUNTIME_DIR}/source:/source-output"
  sglang.sif "${benchmark[@]}")

if [[ "${node_rank}" -eq 0 ]]; then
  set +e
  "${command[@]}"
  node_status=$?
  set -e
  printf '%s\n' "${node_status}" > "${completion_file}"
  exit "${node_status}"
fi

setsid "${command[@]}" >"${RUNTIME_DIR}/logs/node-rank-1.log" 2>&1 &
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
