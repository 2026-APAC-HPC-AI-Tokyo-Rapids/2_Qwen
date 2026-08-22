#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_IMAGE:?Enroot image is required}"
: "${GPUS_PER_NODE:?GPU count per node is required}"
: "${MODEL_PATH:?model metadata directory is required}"
: "${OMPI_COMM_WORLD_RANK:?MPI rank is required}"
: "${OMPI_COMM_WORLD_SIZE:?MPI size is required}"
: "${RUNTIME_DIR:?runtime directory is required}"

cd "${RUNTIME_DIR}"
metrics_cli="${RUNTIME_DIR}/qwen-experiment-metrics/.venv/bin/qwen-experiment-metrics"
manifest="${RUNTIME_DIR}/control/run-manifest.json"
execution_settings="${RUNTIME_DIR}/control/execution-settings.json"
run_id="$(jq -er '.run_id' "${manifest}")"
run_purpose="$(jq -er '.run_purpose' "${manifest}")"
dataset_path="$(jq -er '.source.path' "${RUNTIME_DIR}/source/dataset-manifest.json")"
nodes="$(jq -er '.nodes' "${execution_settings}")"
tensor_parallel_size="$(jq -er '.tensor_parallel_size' "${execution_settings}")"
sglang_port="$(jq -er '.sglang_port' "${execution_settings}")"
rendezvous_timeout_sec="$(jq -er '.rendezvous_timeout_sec' "${execution_settings}")"
node_rank="${OMPI_COMM_WORLD_RANK}"
[[ "${OMPI_COMM_WORLD_SIZE}" -eq "${nodes}" ]]
[[ "${node_rank}" -ge 0 && "${node_rank}" -lt "${nodes}" ]]

node_temporary="$(mktemp -d /dev/shm/qwen-enroot-node.XXXXXXXXXX)"
jit_temporary_root="${TMPDIR:-${TEMPDIR:-${TEMP:-/tmp}}}"
[[ "${jit_temporary_root}" == /* && -d "${jit_temporary_root}" && -w "${jit_temporary_root}" ]]
jit_temporary="$(mktemp -d "${jit_temporary_root%/}/qwen-jit.XXXXXXXXXX")"
container_home="${jit_temporary}/home"
jit_cache="${jit_temporary}/cache"
enroot_data="${jit_temporary}/enroot-data"
enroot_runtime="${jit_temporary}/enroot-runtime"
enroot_cache="${jit_temporary}/enroot-cache"
enroot_temp="${jit_temporary}/enroot-temp"
node_metrics_dir="${node_temporary}/metrics"
mkdir -p "${container_home}/.cache" "${jit_cache}" "${enroot_data}" "${enroot_runtime}" "${enroot_cache}" "${enroot_temp}" "${node_metrics_dir}"
export ENROOT_DATA_PATH="${enroot_data}"
export ENROOT_RUNTIME_PATH="${enroot_runtime}"
export ENROOT_CACHE_PATH="${enroot_cache}"
export ENROOT_TEMP_PATH="${enroot_temp}"
container_name="qwen-node-${node_rank}"
container_created=0
event_server_pid=""
gpu_sampler_pid=""
node_status=1

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${gpu_sampler_pid}" ]]; then kill -TERM "${gpu_sampler_pid}" 2>/dev/null || true; wait "${gpu_sampler_pid}" 2>/dev/null || true; fi
  if [[ -n "${event_server_pid}" ]]; then kill -TERM "${event_server_pid}" 2>/dev/null || true; wait "${event_server_pid}" 2>/dev/null || true; fi
  if [[ "${container_created}" -eq 1 ]]; then enroot remove -f "${container_name}" 2>/dev/null || true; fi
  rm -rf -- "${node_temporary:?}"
  rm -rf -- "${jit_temporary:?}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

enroot create --name "${container_name}" "${CONTAINER_IMAGE}"
container_created=1

"${metrics_cli}" capture-hardware --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" --node-rank "${node_rank}"

master_host_file="${RUNTIME_DIR}/control/master-host.txt"
if [[ "${node_rank}" -eq 0 ]]; then hostname >"${master_host_file}"; else wait_started="${SECONDS}"; while [[ ! -s "${master_host_file}" ]]; do (( SECONDS - wait_started < rendezvous_timeout_sec )); sleep 0.1; done; fi
master_host="$(<"${master_host_file}")"

event_socket="${node_metrics_dir}/events.sock"
event_ready="${node_metrics_dir}/events.ready"
"${metrics_cli}" serve-events --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" --node-rank "${node_rank}" --socket "${event_socket}" --ready-file "${event_ready}" &
event_server_pid=$!
while [[ ! -e "${event_ready}" ]]; do kill -0 "${event_server_pid}"; sleep 0.1; done

if [[ "${run_purpose}" == profile ]]; then
  mkdir -p "${RUNTIME_DIR}/profile/raw/gpu" "${RUNTIME_DIR}/profile/raw/torch"
  gpu_output="${RUNTIME_DIR}/profile/raw/gpu/node-${node_rank}.jsonl"
  gpu_completion="${RUNTIME_DIR}/control/gpu/node-${node_rank}.complete"
  gpu_ready="${node_metrics_dir}/gpu.ready"
  clock_id="$(tr -d '[:space:]' </proc/sys/kernel/random/boot_id):CLOCK_MONOTONIC"
  gpu_args=()
  while IFS=, read -r uuid index; do uuid="${uuid//[[:space:]]/}"; index="${index//[[:space:]]/}"; gpu_args+=(--expected-gpu "${uuid}=${index}"); done < <(nvidia-smi --query-gpu=uuid,index --format=csv,noheader,nounits)
  [[ "$(( ${#gpu_args[@]} / 2 ))" -eq "${GPUS_PER_NODE}" ]]
  "${metrics_cli}" sample-gpu --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" --node-rank "${node_rank}" --clock-id "${clock_id}" --output "${gpu_output}" --ready-file "${gpu_ready}" --completion-file "${gpu_completion}" --interval-sec 1 "${gpu_args[@]}" &
  gpu_sampler_pid=$!
  while [[ ! -e "${gpu_ready}" ]]; do kill -0 "${gpu_sampler_pid}"; sleep 0.1; done
fi

# Qwen3.5 397B FP8のdummy初期化にもweight相当のVRAMが必要なため、玄界は2ノード8 GPUで実行する。
benchmark=(/opt/sglang/venv/bin/python -m sglang.benchmark.offline_throughput --model-path /model --load-format dummy --tp-size "${tensor_parallel_size}" --nnodes "${nodes}" --node-rank "${node_rank}" --dist-init-addr "${master_host}:${sglang_port}" --dataset-name random --dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json --random-input-len 1024 --random-output-len 128 --random-range-ratio 1 --num-prompts 8 --seed 42 --mem-fraction-static 0.80)
sglang_command=("${benchmark[@]}")
profile_mounts=()
profile_environ=()
if [[ "${run_purpose}" == profile ]]; then profile_mounts=(-m "${RUNTIME_DIR}/profile/raw/torch:/profile" -m "${RUNTIME_DIR}/scripts/qwen-profile-wrapper.py:/qwen-profile-wrapper.py"); profile_environ=(-e SGLANG_TORCH_PROFILER_DIR=/profile -e SGLANG_PROFILE_ID=qwen-profile -e QWEN_EXPERIMENT_METRICS_PHASE_SOCKET=/metrics/phase.sock -e SGLANG_PROFILE_WITH_STACK=0 -e SGLANG_PROFILE_RECORD_SHAPES=0); sglang_command=(/opt/sglang/venv/bin/python /qwen-profile-wrapper.py --output-dir /profile -- "${benchmark[@]}"); fi

runtime_source_oids="${RUNTIME_DIR}/source/runtime-source-oids.json"
bench_args_blob_oid="$(jq -er 'select(.schema_version == "qwen-runtime-source-oids/v1") | .bench_args_blob_oid' "${runtime_source_oids}")"
server_args_blob_oid="$(jq -er '.server_args_blob_oid' "${runtime_source_oids}")"
container_environment=(-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_DISABLE_REQUIRE=1 -e MELLANOX_VISIBLE_DEVICES=all -e MELLANOX_MOUNT_DRIVER=yes -e HOME=/runtime-home -e XDG_CACHE_HOME=/runtime-home/.cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e FLASHINFER_WORKSPACE_BASE=/jit-cache -e SGLANG_DG_CACHE_DIR=/jit-cache/deep-gemm -e TORCHINDUCTOR_CACHE_DIR=/jit-cache/torchinductor -e TRITON_CACHE_DIR=/jit-cache/triton -e QWEN_EXPERIMENT_METRICS_SOCKET=/metrics/events.sock -e "QWEN_EXPERIMENT_METRICS_RUN_ID=${run_id}" -e "QWEN_EXPERIMENT_METRICS_NODE_RANK=${node_rank}" -e "QWEN_EXPERIMENT_METRICS_BENCH_ARGS_BLOB_OID=${bench_args_blob_oid}" -e "QWEN_EXPERIMENT_METRICS_SERVER_ARGS_BLOB_OID=${server_args_blob_oid}" -e QWEN_EXPERIMENT_METRICS_SOURCE_DIR=/source-output)
container_mounts=(-m "${MODEL_PATH}:/model" -m "${dataset_path}:/dataset/ShareGPT_V3_unfiltered_cleaned_split.json:none:x-create=file,bind,ro,nosuid,nodev" -m "${container_home}:/runtime-home" -m "${jit_cache}:/jit-cache" -m "${node_metrics_dir}:/metrics" -m "${RUNTIME_DIR}/source:/source-output")
command=(enroot start "${container_environment[@]}" "${profile_environ[@]}" "${container_mounts[@]}" "${profile_mounts[@]}" "${container_name}" "${sglang_command[@]}")

argv_dir="${node_metrics_dir}/argv"
mkdir -p "${argv_dir}"
jq -n --args '$ARGS.positional' -- "$0" "$@" >"${argv_dir}/node-wrapper.json"
jq -n --args '$ARGS.positional' -- "${command[@]}" >"${argv_dir}/enroot.json"
jq -n --args '$ARGS.positional' -- "${sglang_command[@]}" >"${argv_dir}/sglang.json"
"${metrics_cli}" capture-topology --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" --node-rank "${node_rank}" --global-rank "${OMPI_COMM_WORLD_RANK}" --local-rank "${OMPI_COMM_WORLD_LOCAL_RANK:-0}" --size "${OMPI_COMM_WORLD_SIZE}" --node-wrapper-argv "${argv_dir}/node-wrapper.json" --container-argv "${argv_dir}/enroot.json" --sglang-argv "${argv_dir}/sglang.json" --container-launcher "$(command -v enroot)" --container-target "${container_name}"

phase_agent_args=()
if [[ "${run_purpose}" == profile ]]; then phase_agent_args=(--phase-socket "${node_metrics_dir}/phase.sock"); fi
node_agent=("${RUNTIME_DIR}/qwen-experiment-metrics/.venv/bin/python" -m qwen_experiment_metrics.node_agent --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" --node-rank "${node_rank}" --master-host "${master_host}" "${phase_agent_args[@]}" -- "${command[@]}")
set +e
if [[ "${node_rank}" -eq 0 ]]; then "${node_agent[@]}"; node_status=$?; else "${node_agent[@]}" >"${RUNTIME_DIR}/logs/node-rank-${node_rank}.log" 2>&1; node_status=$?; fi
set -e

if [[ "${run_purpose}" == profile ]]; then kill -TERM "${gpu_sampler_pid}"; gpu_sampler_status=0; wait "${gpu_sampler_pid}" || gpu_sampler_status=$?; gpu_sampler_pid=""; [[ "${gpu_sampler_status}" -eq 0 ]]; "${metrics_cli}" capture-profile-hardware --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" --node-rank "${node_rank}"; fi
if [[ "${node_rank}" -eq 0 && -s "${RUNTIME_DIR}/source/workload-warmup.json" ]]; then "${metrics_cli}" record-workload --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" --kind warmup --source "${RUNTIME_DIR}/source/workload-warmup.json"; fi
if [[ "${node_rank}" -eq 0 && -s "${RUNTIME_DIR}/source/workload-measured.json" ]]; then "${metrics_cli}" record-workload --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" --kind measured --source "${RUNTIME_DIR}/source/workload-measured.json"; fi
exit "${node_status}"
