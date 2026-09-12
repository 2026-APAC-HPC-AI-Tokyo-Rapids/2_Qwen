#!/usr/bin/env bash
#SBATCH --partition=hpcai
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=00:17:00

set -euo pipefail
source "${HOME}/.local/.setenv"
module purge
module load nvhpc-hpcx-cuda13/26.5

script_path="$(realpath "${BASH_SOURCE[0]}")"
submission_dir="$(cd "$(dirname "${script_path}")/.." && pwd)"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-${submission_dir}/build/sglang.sqsh}"
export MODEL_PATH="${MODEL_PATH:-/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
export OUTPUT_DIR="${OUTPUT_DIR:-${submission_dir}/results/logs/${SLURM_JOB_ID}}"
export PROFILE="${PROFILE:-false}"
export CONCURRENCY="${CONCURRENCY:-16}"
mode="${1:-orchestrate}"

sample_system() {
  local rank="$1" profile_dir="$2" timestamp user nice system idle iowait irq softirq steal interface rx_bytes rx_packets rx_errors tx_bytes tx_packets tx_errors counters device
  local cpu_output="${profile_dir}/cpu-node-${rank}.csv"
  local network_output="${profile_dir}/network-node-${rank}.csv"
  local infiniband_output="${profile_dir}/infiniband-node-${rank}.csv"
  printf 'timestamp,user,nice,system,idle,iowait,irq,softirq,steal\n' >"${cpu_output}"
  printf 'timestamp,interface,rx_bytes,rx_packets,rx_errors,tx_bytes,tx_packets,tx_errors\n' >"${network_output}"
  printf 'timestamp,device,port_xmit_data,port_rcv_data,port_xmit_packets,port_rcv_packets\n' >"${infiniband_output}"
  while true; do
    timestamp="$(date -u +%FT%T.%NZ)"
    read -r _ user nice system idle iowait irq softirq steal _ </proc/stat
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "${timestamp}" "${user}" "${nice}" "${system}" "${idle}" "${iowait}" "${irq}" "${softirq}" "${steal}" >>"${cpu_output}"
    while IFS=': ' read -r interface rx_bytes rx_packets rx_errors _ _ _ _ _ tx_bytes tx_packets tx_errors _; do
      [[ -n "${interface}" && "${rx_bytes}" =~ ^[0-9]+$ ]] || continue
      printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "${timestamp}" "${interface}" "${rx_bytes}" "${rx_packets}" "${rx_errors}" "${tx_bytes}" "${tx_packets}" "${tx_errors}" >>"${network_output}"
    done </proc/net/dev
    for counters in /sys/class/infiniband/*/ports/1/counters; do
      [[ -d "${counters}" ]] || continue
      [[ -r "${counters}/port_xmit_data" && -r "${counters}/port_rcv_data" && -r "${counters}/port_xmit_packets" && -r "${counters}/port_rcv_packets" ]] || continue
      device="${counters#/sys/class/infiniband/}"
      device="${device%%/*}"
      printf '%s,%s,%s,%s,%s,%s\n' "${timestamp}" "${device}" "$(<"${counters}/port_xmit_data")" "$(<"${counters}/port_rcv_data")" "$(<"${counters}/port_xmit_packets")" "$(<"${counters}/port_rcv_packets")" >>"${infiniband_output}"
    done
    sleep 1
  done
}

run_node() {
  node_rank="${SLURM_PROCID:?Slurm process rank is required}"
  hostname_short="$(hostname -s)"
  node_ip="$(getent ahostsv4 "${hostname_short}" | awk '$2 == "STREAM" { print $1; exit }')"
  [[ "${node_ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]
  node_tmp="$(mktemp -d "${TMPDIR:-/tmp}/qwen-submission-node-${node_rank}.XXXXXXXXXX")"
  pids=()
  sampler_pid=
  system_sampler_pid=
  cleanup_node() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "${sampler_pid}" ]]; then kill "${sampler_pid}" 2>/dev/null || true; wait "${sampler_pid}" 2>/dev/null || true; fi
    if [[ -n "${system_sampler_pid}" ]]; then kill "${system_sampler_pid}" 2>/dev/null || true; wait "${system_sampler_pid}" 2>/dev/null || true; fi
    for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
    rm -rf -- "${node_tmp:?}"
    exit "${status}"
  }
  trap cleanup_node EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  ib_devices=()
  for state_file in /sys/class/infiniband/*/ports/1/state; do device="${state_file#/sys/class/infiniband/}"; device="${device%%/*}"; if grep -q ACTIVE "${state_file}" && grep -qx InfiniBand "/sys/class/infiniband/${device}/ports/1/link_layer"; then ib_devices+=("${device}:1"); fi; done
  [[ "${#ib_devices[@]}" -gt 0 ]]
  ucx_devices="$(IFS=,; printf '%s' "${ib_devices[*]}")"
  printf 'hostname=%s\nnode_ip=%s\nnode_rank=%s\nucx_net_devices=%s\n' "${hostname_short}" "${node_ip}" "${node_rank}" "${ucx_devices}" >"${OUTPUT_DIR}/node-${node_rank}-environment.log"
  nvidia-smi -L >>"${OUTPUT_DIR}/node-${node_rank}-environment.log"
  if [[ "${PROFILE}" == true ]]; then
    install -d -m 0755 "${OUTPUT_DIR}/profile"
    nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm --format=csv -l 1 >"${OUTPUT_DIR}/profile/gpu-node-${node_rank}.csv" & sampler_pid=$!
    sample_system "${node_rank}" "${OUTPUT_DIR}/profile" & system_sampler_pid=$!
  fi

  start_worker() {
    role="$1"
    worker="$2"
    tp="$3"
    port="$4"
    bootstrap="$5"
    gpus="$6"
    worker_tmp="${node_tmp}/${role}-${worker}"
    install -d -m 0700 "${worker_tmp}/cache" "${worker_tmp}/data" "${worker_tmp}/home" "${worker_tmp}/runtime" "${worker_tmp}/temp"
    args=(/opt/sglang/venv/bin/python -m sglang.launch_server --model-path /model --served-model-name qwen --trust-remote-code --context-length 8192 --tp-size "${tp}" --mem-fraction-static 0.90 --disaggregation-mode "${role}" --disaggregation-transfer-backend nixl --host 0.0.0.0 --port "${port}")
    if [[ "${role}" == prefill ]]; then args+=(--disaggregation-bootstrap-port "${bootstrap}"); else args+=(--ep-size 4 --max-running-requests 256); fi
    (export ENROOT_CACHE_PATH="${worker_tmp}/cache" ENROOT_DATA_PATH="${worker_tmp}/data" ENROOT_RUNTIME_PATH="${worker_tmp}/runtime" ENROOT_TEMP_PATH="${worker_tmp}/temp"; exec enroot start -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_DISABLE_REQUIRE=1 -e MELLANOX_VISIBLE_DEVICES=all -e MELLANOX_MOUNT_DRIVER=yes -e "CUDA_VISIBLE_DEVICES=${gpus}" -e "UCX_NET_DEVICES=${ucx_devices}" -e "SGLANG_HOST_IP=${node_ip}" -e HOME=/runtime-home -e XDG_CACHE_HOME=/runtime-home/.cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e FLASHINFER_WORKSPACE_BASE=/runtime-home/.cache/flashinfer -e SGLANG_DG_CACHE_DIR=/runtime-home/.cache/deep-gemm -e TORCHINDUCTOR_CACHE_DIR=/runtime-home/.cache/torchinductor -e TRITON_CACHE_DIR=/runtime-home/.cache/triton -m "${MODEL_PATH}:/model" -m "${worker_tmp}/home:/runtime-home" "${CONTAINER_IMAGE}" "${args[@]}") >"${OUTPUT_DIR}/${role}-${worker}.log" 2>&1 &
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
  while true; do for pid in "${pids[@]}"; do if ! kill -0 "${pid}" 2>/dev/null; then wait "${pid}"; fi; done; sleep 2; done
}

run_router() {
  router_tmp="$(mktemp -d "${TMPDIR:-/tmp}/qwen-submission-router.XXXXXXXXXX")"
  child_pid=
  cleanup_router() { status=$?; trap - EXIT INT TERM; if [[ -n "${child_pid}" ]]; then kill "${child_pid}" 2>/dev/null || true; wait "${child_pid}" 2>/dev/null || true; fi; rm -rf -- "${router_tmp:?}"; exit "${status}"; }
  trap cleanup_router EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  export ENROOT_CACHE_PATH="${router_tmp}/cache" ENROOT_DATA_PATH="${router_tmp}/data" ENROOT_RUNTIME_PATH="${router_tmp}/runtime" ENROOT_TEMP_PATH="${router_tmp}/temp"
  install -d -m 0700 "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}"
  enroot start -e NVIDIA_VISIBLE_DEVICES=none -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -m "${MODEL_PATH}:/model" "${CONTAINER_IMAGE}" /opt/sglang/venv/bin/python -m sglang_router.launch_router --pd-disaggregation --prefill "http://${HEAD_IP}:30100" 9000 --prefill "http://${HEAD_IP}:30101" 9001 --prefill "http://${HEAD_IP}:30102" 9002 --prefill "http://${HEAD_IP}:30103" 9003 --decode "http://${TAIL_IP}:30200" --decode "http://${TAIL_IP}:30201" --host 0.0.0.0 --port 8000 &
  child_pid=$!
  wait "${child_pid}"
  child_pid=
}

run_aiperf() {
  aiperf_tmp="$(mktemp -d "${TMPDIR:-/tmp}/qwen-submission-aiperf.XXXXXXXXXX")"
  child_pid=
  cleanup_aiperf() { status=$?; trap - EXIT INT TERM; if [[ -n "${child_pid}" ]]; then kill "${child_pid}" 2>/dev/null || true; wait "${child_pid}" 2>/dev/null || true; fi; rm -rf -- "${aiperf_tmp:?}"; exit "${status}"; }
  trap cleanup_aiperf EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  export ENROOT_CACHE_PATH="${aiperf_tmp}/cache" ENROOT_DATA_PATH="${aiperf_tmp}/data" ENROOT_RUNTIME_PATH="${aiperf_tmp}/runtime" ENROOT_TEMP_PATH="${aiperf_tmp}/temp"
  install -d -m 0700 "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}"
  enroot start -e NVIDIA_VISIBLE_DEVICES=none -e HOME=/tmp -e HF_HOME=/hf-cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -m "${OUTPUT_DIR}/results:/results" -m "${OUTPUT_DIR}/tokenizer-cache:/hf-cache" "${CONTAINER_IMAGE}" /bin/bash -c 'cd /results && exec "$@"' _ /opt/sglang/venv/bin/aiperf profile --url http://127.0.0.1:8000 --endpoint-type chat --endpoint /v1/chat/completions --model qwen --tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507 --streaming -H 'Accept: text/event-stream' --concurrency "${CONCURRENCY}" --workers-max "${CONCURRENCY}" --api-key NOT_USED --warmup-duration 60 --benchmark-duration 120 --benchmark-grace-period 30 --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 --output-tokens-mean 200 --output-tokens-stddev 0 --extra-inputs min_tokens:200 --extra-inputs max_tokens:200 --extra-inputs ignore_eos:true --extra-inputs temperature:0.0 --extra-inputs repetition_penalty:1.0 --num-dataset-entries 12800 --random-seed 2026 --cache-bust first_turn_prefix --goodput "time_to_first_token:3000 inter_token_latency:20" --ui simple &
  child_pid=$!
  wait "${child_pid}"
  child_pid=
  export_file="$(find "${OUTPUT_DIR}/results/artifacts" -type f -name profile_export_aiperf.json -print -quit)"
  [[ -n "${export_file}" && -s "${export_file}" ]]
  cp "${export_file}" "${OUTPUT_DIR}/results/profile_export_aiperf.json"
  python3 - "${OUTPUT_DIR}/results/profile_export_aiperf.json" "${OUTPUT_DIR}/results/score.json" "${CONCURRENCY}" <<'PY'
import json
import math
import sys

source, destination, concurrency = sys.argv[1:]
metrics = json.load(open(source, encoding="utf-8"))

def average(name, default=None):
    value = metrics.get(name)
    if value is None and default is not None:
        return default
    if not isinstance(value, dict) or not isinstance(value.get("avg"), (int, float)):
        raise SystemExit(f"missing numeric {name}.avg")
    return float(value["avg"])

goodput = average("goodput")
fraction = average("good_request_fraction")
eligible = fraction >= 0.90
result = {
    "schema_version": "qwen3-pd-score/v1",
    "concurrency": int(concurrency),
    "goodput_requests_per_second": goodput,
    "good_request_fraction": fraction,
    "eligible": eligible,
    "score_output_tokens_per_second": goodput * 200 if eligible else 0.0,
    "request_count": average("request_count"),
    "error_request_count": average("error_request_count", 0.0),
    "observed_input_sequence_length_avg": average("input_sequence_length"),
    "observed_output_sequence_length_avg": average("output_sequence_length"),
    "sla": {"time_to_first_token_ms": 3000, "inter_token_latency_ms": 20},
}
with open(destination, "w", encoding="utf-8") as output:
    json.dump(result, output, indent=2, sort_keys=True)
    output.write("\n")
print(json.dumps(result, sort_keys=True))
if not eligible:
    raise SystemExit("official run did not meet good_request_fraction >= 0.90")
if not math.isclose(result["observed_input_sequence_length_avg"], 4000.0):
    raise SystemExit("official run did not produce average input length 4000")
if not math.isclose(result["observed_output_sequence_length_avg"], 200.0):
    raise SystemExit("official run did not produce average output length 200")
PY
}

if [[ "${mode}" == node ]]; then run_node; exit; fi
if [[ "${mode}" == router ]]; then run_router; exit; fi
if [[ "${mode}" == aiperf ]]; then run_aiperf; exit; fi
[[ "${mode}" == orchestrate ]]
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]
[[ "${PROFILE}" == true || "${PROFILE}" == false ]]
[[ -s "${CONTAINER_IMAGE}" ]]
[[ -s "${MODEL_PATH}/config.json" && -s "${MODEL_PATH}/model.safetensors.index.json" ]]
install -d -m 0755 "${OUTPUT_DIR}/results" "${OUTPUT_DIR}/tokenizer-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/refs" "${OUTPUT_DIR}/tokenizer-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e"
tokenizer_snapshot="${OUTPUT_DIR}/tokenizer-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e"
for tokenizer_file in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do test -s "${MODEL_PATH}/${tokenizer_file}"; ln -s "${MODEL_PATH}/${tokenizer_file}" "${tokenizer_snapshot}/${tokenizer_file}"; done
printf '%s' ac9c66cc9b46af7306746a9250f23d47083d689e >"${OUTPUT_DIR}/tokenizer-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/refs/main"
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ "${#nodes[@]}" -eq 2 ]]
head_host="${nodes[0]}"
tail_host="${nodes[1]}"
head_ip="$(getent ahostsv4 "${head_host}" | awk '$2 == "STREAM" { print $1; exit }')"
tail_ip="$(getent ahostsv4 "${tail_host}" | awk '$2 == "STREAM" { print $1; exit }')"
[[ "${head_ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]
[[ "${tail_ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]
server_pid=
router_pid=
cleanup_all() { status=$?; trap - EXIT INT TERM; if [[ -n "${router_pid}" ]]; then kill "${router_pid}" 2>/dev/null || true; wait "${router_pid}" 2>/dev/null || true; fi; if [[ -n "${server_pid}" ]]; then kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true; fi; exit "${status}"; }
trap cleanup_all EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 --distribution=block:block "${script_path}" node >"${OUTPUT_DIR}/server-step.log" 2>&1 &
server_pid=$!
worker_urls=("http://${head_ip}:30100/health" "http://${head_ip}:30101/health" "http://${head_ip}:30102/health" "http://${head_ip}:30103/health" "http://${tail_ip}:30200/health" "http://${tail_ip}:30201/health")
deadline="$((SECONDS + 900))"
while true; do kill -0 "${server_pid}"; ready=0; for url in "${worker_urls[@]}"; do if curl --fail --silent --max-time 2 "${url}" >/dev/null; then ready="$((ready + 1))"; fi; done; if [[ "${ready}" -eq 6 ]]; then break; fi; (( SECONDS < deadline )); sleep 5; done
srun --overlap --nodes=1 --ntasks=1 --nodelist="${head_host}" env HEAD_IP="${head_ip}" TAIL_IP="${tail_ip}" "${script_path}" router >"${OUTPUT_DIR}/router.log" 2>&1 &
router_pid=$!
deadline="$((SECONDS + 300))"
models_url="http://${head_ip}:8000/v1/models"
models_pending="${OUTPUT_DIR}/models.pending.json"
while true; do
  kill -0 "${router_pid}"
  if curl --fail --silent --max-time 2 "${models_url}" >"${models_pending}" && jq -e '.data | map(.id) | index("qwen") != null' "${models_pending}" >/dev/null; then mv "${models_pending}" "${OUTPUT_DIR}/models.json"; break; fi
  (( SECONDS < deadline ))
  sleep 2
done
srun --overlap --nodes=1 --ntasks=1 --nodelist="${head_host}" "${script_path}" aiperf >"${OUTPUT_DIR}/aiperf.log" 2>&1
touch "${OUTPUT_DIR}/run-success"
