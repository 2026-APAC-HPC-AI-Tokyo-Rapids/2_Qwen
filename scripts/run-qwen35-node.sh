#!/usr/bin/env bash
# Launch one SGLang process per node. Observability remains node-local until
# completed fragments and source files are merged by the Actions runner.
set -euo pipefail

: "${RUNTIME_DIR:?RUNTIME_DIR is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATASET_PATH:?DATASET_PATH is required}"
: "${OMPI_COMM_WORLD_RANK:?OMPI_COMM_WORLD_RANK is required}"
: "${SINGULARITY_LAUNCHER:?SINGULARITY_LAUNCHER is required}"

cd "${RUNTIME_DIR}"
metrics_cli="${RUNTIME_DIR}/qwen-experiment-metrics/.venv/bin/qwen-experiment-metrics"
manifest="${RUNTIME_DIR}/control/run-manifest.json"
execution_settings="${RUNTIME_DIR}/control/execution-settings.json"
run_id="$(jq -er '.run_id' "${manifest}")"
run_purpose="$(jq -er '.run_purpose' "${manifest}")"
nodes="$(jq -er '.nodes' "${execution_settings}")"
mpi_processes="$(jq -er '.mpi_processes' "${execution_settings}")"
gpus_per_node="$(jq -er '.gpus_per_node' "${execution_settings}")"
tensor_parallel_size="$(jq -er '.tensor_parallel_size' "${execution_settings}")"
sglang_port="$(jq -er '.sglang_port' "${execution_settings}")"
rendezvous_timeout_sec="$(jq -er '.rendezvous_timeout_sec' "${execution_settings}")"
ready_timeout_sec="$(jq -er '.ready_timeout_sec' "${execution_settings}")"
node_rank="${OMPI_COMM_WORLD_RANK}"
[[ "${OMPI_COMM_WORLD_SIZE}" -eq "${mpi_processes}" ]]
[[ "${node_rank}" -ge 0 && "${node_rank}" -lt "${nodes}" ]]
bench_args_blob_oid="$("${SINGULARITY_LAUNCHER}" exec sglang.sif cat /opt/sglang/offline-throughput-source-blob-oid.txt | tr -d '[:space:]')"
server_args_blob_oid="$("${SINGULARITY_LAUNCHER}" exec sglang.sif cat /opt/sglang/server-args-source-blob-oid.txt | tr -d '[:space:]')"
dataset_manifest_sha256="$(sha256sum "${RUNTIME_DIR}/source/dataset-manifest.json" | awk '{ print $1 }')"
jit_cache="$(mktemp -d /tmp/sglang-jit.XXXXXXXXXX)"
singularity_tmpdir="$(mktemp -d /dev/shm/sglang-run.XXXXXXXXXX)"
node_metrics_dir="$(mktemp -d /dev/shm/qwen-metrics.XXXXXXXXXX)"
export SINGULARITY_TMPDIR="${singularity_tmpdir}"
jit_lock_fd=""
event_server_pid=""
gpu_sampler_pid=""
node_status=1

cleanup() {
  local status=$?
  trap - EXIT INT TERM
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
"${metrics_cli}" record-cache \
  --manifest "${manifest}" \
  --runtime-dir "${RUNTIME_DIR}" \
  --node-rank "${node_rank}" \
  --cache-root "${jit_cache}"
"${metrics_cli}" record-environment \
  --manifest "${manifest}" \
  --runtime-dir "${RUNTIME_DIR}" \
  --scope host-node \
  --index "${node_rank}"
"${metrics_cli}" capture-hardware \
  --manifest "${manifest}" \
  --runtime-dir "${RUNTIME_DIR}" \
  --node-rank "${node_rank}"

master_host_file="${RUNTIME_DIR}/control/master-host.txt"
if [[ "${node_rank}" -eq 0 ]]; then
  hostname > "${master_host_file}"
else
  wait_started="${SECONDS}"
  while [[ ! -s "${master_host_file}" ]]; do
    (( SECONDS - wait_started < rendezvous_timeout_sec ))
    sleep 0.1
  done
fi
master_host="$(<"${master_host_file}")"

event_socket="${node_metrics_dir}/events.sock"
event_ready="${node_metrics_dir}/events.ready"
"${metrics_cli}" serve-events \
  --manifest "${manifest}" \
  --runtime-dir "${RUNTIME_DIR}" \
  --node-rank "${node_rank}" \
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
  [[ "$(( ${#gpu_args[@]} / 2 ))" -eq "${gpus_per_node}" ]]
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
  --tp-size "${tensor_parallel_size}"
  --nnodes "${nodes}"
  --node-rank "${node_rank}"
  --dist-init-addr "${master_host}:${sglang_port}"
  --dataset-name sharegpt
  --dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json
  --num-prompts 32
  --sharegpt-output-len 256
  --sharegpt-context-len 262144
  --seed 42
  --mem-fraction-static 0.85)
sglang_command=("${benchmark[@]}")

profile_env_args=()
profile_bind_args=()
if [[ "${run_purpose}" == profile ]]; then
  mkdir -p "${RUNTIME_DIR}/profile/raw/torch"
  profile_env_args=(
    --env SGLANG_TORCH_PROFILER_DIR=/profile
    --env SGLANG_PROFILE_ID=qwen-profile
    --env QWEN_EXPERIMENT_METRICS_PHASE_SOCKET=/metrics/phase.sock
    --env SGLANG_PROFILE_WITH_STACK=0,SGLANG_PROFILE_RECORD_SHAPES=0)
  profile_bind_args=(
    --bind "${RUNTIME_DIR}/profile/raw/torch:/profile"
    --bind "${RUNTIME_DIR}/scripts/qwen-profile-wrapper.py:/qwen-profile-wrapper.py:ro")
  sglang_command=(/opt/sglang/venv/bin/python /qwen-profile-wrapper.py
    --output-dir /profile -- "${benchmark[@]}")
fi

singularity_runtime_args=(--nv --cleanenv --containall
  --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_DATASETS_OFFLINE=1
  --env SGLANG_DG_CACHE_DIR=/jit-cache/deep-gemm,TORCHINDUCTOR_CACHE_DIR=/jit-cache/torchinductor,TRITON_CACHE_DIR=/jit-cache/triton
  --env QWEN_EXPERIMENT_METRICS_SOCKET=/metrics/events.sock
  --env QWEN_EXPERIMENT_METRICS_RUN_ID="${run_id}",QWEN_EXPERIMENT_METRICS_NODE_RANK="${node_rank}"
  --env QWEN_EXPERIMENT_METRICS_BENCH_ARGS_BLOB_OID="${bench_args_blob_oid}"
  --env QWEN_EXPERIMENT_METRICS_SERVER_ARGS_BLOB_OID="${server_args_blob_oid}"
  --env QWEN_EXPERIMENT_METRICS_DATASET_MANIFEST_SHA256="${dataset_manifest_sha256}"
  --env QWEN_EXPERIMENT_METRICS_SOURCE_DIR=/source-output
  "${profile_env_args[@]}"
  "${profile_bind_args[@]}"
  --bind "${MODEL_PATH}:/model:ro,${DATASET_PATH}:/dataset/ShareGPT_V3_unfiltered_cleaned_split.json:ro,${jit_cache}:/jit-cache,${jit_cache}:/root/.cache,${node_metrics_dir}:/metrics,${RUNTIME_DIR}/source:/source-output")
singularity_namespace_args=()
if ! "${SINGULARITY_LAUNCHER}" exec "${singularity_runtime_args[@]}" sglang.sif true; then
  # Genkai has alternated between requiring the user-namespace fallback and
  # rejecting user namespace creation.  Select a working mode before the
  # benchmark with the exact namespace-affecting options and binds so a
  # namespace-policy change cannot consume the measured run.
  singularity_namespace_args=(--userns)
  "${SINGULARITY_LAUNCHER}" exec "${singularity_namespace_args[@]}" "${singularity_runtime_args[@]}" sglang.sif true
fi

command=("${SINGULARITY_LAUNCHER}" exec "${singularity_namespace_args[@]}" "${singularity_runtime_args[@]}"
  sglang.sif /opt/sglang/venv/bin/python -c
  'import os,sys; p=sys.argv[1]; os.makedirs(os.path.dirname(p),exist_ok=True); f=open(p,"xb"); f.write(b"".join(k+b"\0"+v+b"\0" for k,v in sorted(os.environb.items()))); f.close(); os.execvpe(sys.argv[2],sys.argv[2:],os.environ)'
  "/source-output/environment/sglang-${node_rank}.snapshot" "${sglang_command[@]}")

argv_dir="${node_metrics_dir}/argv"
mkdir -p "${argv_dir}"
jq -n --args '$ARGS.positional' -- "$0" "$@" >"${argv_dir}/node-wrapper.json"
jq -n --args '$ARGS.positional' -- "${command[@]}" >"${argv_dir}/singularity.json"
jq -n --args '$ARGS.positional' -- "${sglang_command[@]}" >"${argv_dir}/sglang.json"
"${metrics_cli}" capture-topology \
  --manifest "${manifest}" \
  --runtime-dir "${RUNTIME_DIR}" \
  --node-rank "${node_rank}" \
  --global-rank "${OMPI_COMM_WORLD_RANK}" \
  --local-rank "${OMPI_COMM_WORLD_LOCAL_RANK:-0}" \
  --size "${OMPI_COMM_WORLD_SIZE}" \
  --node-wrapper-argv "${argv_dir}/node-wrapper.json" \
  --singularity-argv "${argv_dir}/singularity.json" \
  --sglang-argv "${argv_dir}/sglang.json" \
  --singularity-launcher "${SINGULARITY_LAUNCHER}" \
  --sif "${RUNTIME_DIR}/sglang.sif"

phase_agent_args=()
if [[ "${run_purpose}" == profile ]]; then
  phase_agent_args=(--phase-socket "${node_metrics_dir}/phase.sock")
fi
node_agent=("${RUNTIME_DIR}/qwen-experiment-metrics/.venv/bin/python"
  -m qwen_experiment_metrics.node_agent
  --manifest "${manifest}"
  --runtime-dir "${RUNTIME_DIR}"
  --node-rank "${node_rank}"
  --master-host "${master_host}"
  "${phase_agent_args[@]}"
  -- "${command[@]}")
set +e
if [[ "${node_rank}" -eq 0 ]]; then
  "${node_agent[@]}"
  node_status=$?
else
  "${node_agent[@]}" >"${RUNTIME_DIR}/logs/node-rank-${node_rank}.log" 2>&1
  node_status=$?
fi
set -e

if [[ "${run_purpose}" == profile ]]; then
  kill -TERM "${gpu_sampler_pid}"
  gpu_sampler_status=0
  wait "${gpu_sampler_pid}" || gpu_sampler_status=$?
  gpu_sampler_pid=""
  [[ "${gpu_sampler_status}" -eq 0 ]]
  "${metrics_cli}" capture-profile-hardware \
    --manifest "${manifest}" \
    --runtime-dir "${RUNTIME_DIR}" \
    --node-rank "${node_rank}"
fi

sglang_snapshot="${RUNTIME_DIR}/source/environment/sglang-${node_rank}.snapshot"
if [[ -s "${sglang_snapshot}" ]]; then
  "${metrics_cli}" record-environment \
    --manifest "${manifest}" \
    --runtime-dir "${RUNTIME_DIR}" \
    --scope sglang-node \
    --index "${node_rank}" \
    --source "${sglang_snapshot}"
fi
for ((local_rank=0; local_rank<gpus_per_node; local_rank++)); do
  global_rank=$(( node_rank * gpus_per_node + local_rank ))
  worker_snapshot="${RUNTIME_DIR}/source/environment/worker-${global_rank}.snapshot"
  if [[ -s "${worker_snapshot}" ]]; then
    "${metrics_cli}" record-environment \
      --manifest "${manifest}" \
      --runtime-dir "${RUNTIME_DIR}" \
      --scope worker-rank \
      --index "${global_rank}" \
      --source "${worker_snapshot}"
  fi
done
if [[ "${node_rank}" -eq 0 && -s "${RUNTIME_DIR}/source/workload-warmup.json" ]]; then
  "${metrics_cli}" record-workload \
    --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" \
    --kind warmup --source "${RUNTIME_DIR}/source/workload-warmup.json"
  "${metrics_cli}" record-workload \
    --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" \
    --kind measured --source "${RUNTIME_DIR}/source/workload-measured.json"
fi
exit "${node_status}"
