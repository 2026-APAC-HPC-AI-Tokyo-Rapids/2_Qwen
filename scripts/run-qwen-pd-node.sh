#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_IMAGE:?container image is required}"
: "${MODEL_PATH:?model path is required}"
: "${PROFILE:?profile selection is required}"
: "${RUNTIME_DIR:?runtime directory is required}"
: "${SLURM_PROCID:?Slurm process rank is required}"

[[ "${SLURM_PROCID}" =~ ^[01]$ ]]
node_rank="${SLURM_PROCID}"
hostname_short="$(hostname -s)"
node_ip="$(getent ahostsv4 "${hostname_short}" | awk '$2 == "STREAM" { print $1; exit }')"
[[ "${node_ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]
node_tmp="$(mktemp -d "${TMPDIR:-/tmp}/qwen-pd-${node_rank}.XXXXXXXXXX")"
pids=()
sampler_pid=
system_sampler_pid=

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${sampler_pid}" ]]; then kill "${sampler_pid}" 2>/dev/null || true; wait "${sampler_pid}" 2>/dev/null || true; fi
  if [[ -n "${system_sampler_pid}" ]]; then kill "${system_sampler_pid}" 2>/dev/null || true; wait "${system_sampler_pid}" 2>/dev/null || true; fi
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  rm -rf -- "${node_tmp:?}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ib_devices=()
for state_file in /sys/class/infiniband/*/ports/1/state; do if grep -q 'ACTIVE' "${state_file}"; then device="${state_file#/sys/class/infiniband/}"; ib_devices+=("${device%%/*}:1"); fi; done
[[ "${#ib_devices[@]}" -gt 0 ]]
UCX_NET_DEVICES="$(IFS=,; printf '%s' "${ib_devices[*]}")"
printf 'hostname=%s\nnode_ip=%s\nnode_rank=%s\nucx_net_devices=%s\n' "${hostname_short}" "${node_ip}" "${node_rank}" "${UCX_NET_DEVICES}" >"${RUNTIME_DIR}/logs/node-${node_rank}-environment.log"
nvidia-smi -L >>"${RUNTIME_DIR}/logs/node-${node_rank}-environment.log"

if [[ "${PROFILE}" == true ]]; then
  OUTPUT="${RUNTIME_DIR}/profile/gpu-node-${node_rank}.csv" "${RUNTIME_DIR}/scripts/sample-gpu.sh" & sampler_pid=$!
  NODE_RANK="${node_rank}" OUTPUT_DIR="${RUNTIME_DIR}/profile" "${RUNTIME_DIR}/scripts/sample-system.sh" & system_sampler_pid=$!
fi

start_worker() {
  local role="$1" worker="$2" tp="$3" port="$4" bootstrap="$5" gpus="$6"
  local worker_tmp="${node_tmp}/${role}-${worker}"
  install -d -m 0700 "${worker_tmp}/cache" "${worker_tmp}/data" "${worker_tmp}/home" "${worker_tmp}/runtime" "${worker_tmp}/temp"
  local args=(/opt/sglang/venv/bin/python -m sglang.launch_server --model-path /model --served-model-name qwen --trust-remote-code --context-length 8192 --tp-size "${tp}" --mem-fraction-static 0.90 --max-total-tokens 131072 --disaggregation-mode "${role}" --disaggregation-transfer-backend nixl --host 0.0.0.0 --port "${port}")
  if [[ "${role}" == prefill ]]; then args+=(--disaggregation-bootstrap-port "${bootstrap}"); else args+=(--ep-size 4 --max-running-requests 256); fi
  (export ENROOT_CACHE_PATH="${worker_tmp}/cache" ENROOT_DATA_PATH="${worker_tmp}/data" ENROOT_RUNTIME_PATH="${worker_tmp}/runtime" ENROOT_TEMP_PATH="${worker_tmp}/temp"; exec enroot start -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_DISABLE_REQUIRE=1 -e MELLANOX_VISIBLE_DEVICES=all -e MELLANOX_MOUNT_DRIVER=yes -e "CUDA_VISIBLE_DEVICES=${gpus}" -e "UCX_NET_DEVICES=${UCX_NET_DEVICES}" -e "SGLANG_HOST_IP=${node_ip}" -e HOME=/runtime-home -e XDG_CACHE_HOME=/runtime-home/.cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e FLASHINFER_WORKSPACE_BASE=/runtime-home/.cache/flashinfer -e SGLANG_DG_CACHE_DIR=/runtime-home/.cache/deep-gemm -e TORCHINDUCTOR_CACHE_DIR=/runtime-home/.cache/torchinductor -e TRITON_CACHE_DIR=/runtime-home/.cache/triton -m "${MODEL_PATH}:/model" -m "${worker_tmp}/home:/runtime-home" "${CONTAINER_IMAGE}" "${args[@]}") >"${RUNTIME_DIR}/logs/${role}-${worker}.log" 2>&1 &
  pids+=("$!")
}

if [[ "${node_rank}" -eq 0 ]]; then
  start_worker prefill 0 2 30100 9000 0,1
  start_worker prefill 1 2 30101 9001 2,3
  start_worker prefill 2 2 30102 9002 4,5
  start_worker prefill 3 2 30103 9003 6,7
else
  start_worker decode 0 4 30200 unused 0,1,2,3
  start_worker decode 1 4 30201 unused 4,5,6,7
fi

while true; do
  for pid in "${pids[@]}"; do if ! kill -0 "${pid}" 2>/dev/null; then wait "${pid}"; fi; done
  sleep 2
done
