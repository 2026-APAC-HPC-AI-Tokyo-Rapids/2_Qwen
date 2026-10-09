#!/bin/bash
#SBATCH -J sglang-native-pep8-dp2-c96
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:8
#SBATCH --propagate=NONE
#SBATCH --no-requeue
#SBATCH -o runs/sglang-%j.out
#SBATCH -e runs/sglang-%j.err
# Compare shared Prefill experts and DP attention; retain f084943 Decode and c96.
# One full benchmark allocation, with the official 60/120/30 client.
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR}
SGLANG=${ROOT}/sglang.sqsh
AIPERF=${ROOT}/sglang.sqsh
MODEL=/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8
R_PORT=8000
READY_SECS=600  # ASPIRE full stack took 7m09s; leave time for the 60/120/30 client.
N=$(jq -er .concurrency "${ROOT}/source/firmus-framework.json")
# Parse configuration on the host; the pinned serving image does not include jq.
speculation=(--speculative-algorithm EAGLE3 --speculative-draft-model-path /draft --speculative-num-steps "$(jq -er '.speculation.steps' "${ROOT}/source/firmus-framework.json")" --speculative-eagle-topk 1 --speculative-num-draft-tokens "$(jq -er '.speculation.draft_tokens' "${ROOT}/source/firmus-framework.json")")
decode_graph=(--cuda-graph-bs-decode)
while read -r size; do decode_graph+=("${size}"); done < <(jq -er '.decode_cuda_graph_bs[]' "${ROOT}/source/firmus-framework.json")
ART=${ROOT}/logs
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
export MELLANOX_VISIBLE_DEVICES=all MELLANOX_MOUNT_DRIVER=yes
[[ "${PROFILE:-false}" == false ]]
jq -e '.framework == "sglang" and .workload == "official"' "${ROOT}/source/firmus-framework.json" >/dev/null

NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD=${NODES[0]}
TAIL=${NODES[1]}
ip_of() { getent ahostsv4 "$1" | awk '{print $1; exit}'; }
HEAD_IP=$(ip_of "${HEAD}")
TAIL_IP=$(ip_of "${TAIL}")

mounts() {
  mkdir -p "${ROOT}/home/$1"
  echo "/scratch:/scratch,${ROOT}/home/$1:${HOME},${ROOT}:${ROOT},${ROOT}/draft:/draft,/dev/infiniband:/dev/infiniband"
}

wait_http() {
  local deadline=$((SECONDS + READY_SECS)) pid status
  while true; do
    for pid in "${worker_pids[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        status=0
        wait "${pid}" || status=$?
        printf 'server_exited pid=%s status=%s waiting_for=%s\n' "${pid}" "${status}" "$1" >&2
        if (( status == 0 )); then status=1; fi
        return "${status}"
      fi
    done
    if curl --fail --silent --max-time 2 -o /dev/null "$1"; then return 0; fi
    if (( SECONDS >= deadline )); then printf 'server_readiness_timeout url=%s\n' "$1" >&2; return 124; fi
    sleep 2
  done
}

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} user=$(whoami)"
echo "head=${HEAD}(${HEAD_IP}) tail=${TAIL}(${TAIL_IP}) model=${MODEL} isl=4000 osl=200 N=${N} topology=P1world8-EP8-DP2-attnTP4-D2TP4-EP1 kv=fp8_e4m3 speculation=EAGLE3 kernels=p-native-triton"
test -r "${SGLANG}" && test -r "${AIPERF}" && test -f "${MODEL}/config.json" && test -s "${ROOT}/draft/model.safetensors"
mkdir -p "${ART}"
worker_pids=()
cleanup_all() {
  status=$?
  trap - EXIT INT TERM
  for pid in "${worker_pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${worker_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup_all EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'prefill-0\t%s\t30100\t8\n' "${HEAD_IP}" >"${ROOT}/source/worker-endpoints.tsv"
for worker in 0 1; do printf 'decode-%s\t%s\t%s\t4\n' "${worker}" "${TAIL_IP}" "$((30200 + worker))"; done >>"${ROOT}/source/worker-endpoints.tsv"

cat >"${ROOT}/source/worker.sh" <<'EOF'
set -euo pipefail
name=$1
port=$2
model=$3
tp=$4
export SGLANG_HOST_IP=$5
shift 5
role=${name%%-*}
max_total_tokens=131072
if [[ "$role" == decode ]]; then max_total_tokens=262144; fi
ep=1
chunk=8192
if [[ "$role" == prefill ]]; then ep=8; chunk=16384; export SGLANG_CUTLASS_MOE=0; fi
UCX_IB=$(for d in /sys/class/infiniband/*; do
  [ "$(cat "$d/ports/1/state" 2>/dev/null)" = "4: ACTIVE" ] || continue
  echo "$(basename "$d"):1"
done | paste -sd,)
[ -n "$UCX_IB" ] || { echo "FATAL: no active HCA on $(hostname)"; exit 1; }
export UCX_NET_DEVICES=$UCX_IB
export UCX_POSIX_USE_PROC_LINK=n
export TRITON_CACHE_DIR=/tmp/sglang-$name/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-$name/inductor
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
echo "$name host=$(hostname) ip=$SGLANG_HOST_IP hca=$UCX_IB gpu=$CUDA_VISIBLE_DEVICES tp=$tp shuffle=${SGLANG_PREFILL_FUSED_FP8_SHUFFLE:-0} routing=${SGLANG_PREFILL_TILED_ROUTING:-0}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit --format=csv
exec /opt/sglang/venv/bin/python -m sglang.launch_server --model-path "$model" --served-model-name qwen --trust-remote-code --context-length 8192 --tp-size "$tp" --ep-size "$ep" --mem-fraction-static 0.90 --max-total-tokens "$max_total_tokens" --chunked-prefill-size "$chunk" --kv-cache-dtype fp8_e4m3 --attention-backend fa3 --nccl-port "$((port + 1000))" --disaggregation-mode "$role" --disaggregation-transfer-backend nixl "$@" --host 0.0.0.0 --port "$port"
EOF

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:8 --cpus-per-task=64 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts prefill-0)" bash "${ROOT}/source/worker.sh" prefill-0 30100 "${MODEL}" 8 "${HEAD_IP}" "${speculation[@]}" --disaggregation-bootstrap-port 9000 --max-running-requests "${N}" --dp-size 2 --enable-dp-attention --max-prefill-tokens 16384 --moe-a2a-backend none --moe-runner-backend triton --speculative-moe-a2a-backend none --speculative-moe-runner-backend triton >"${ART}/prefill-0.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts decode-0)" bash "${ROOT}/source/worker.sh" decode-0 30200 "${MODEL}" 4 "${TAIL_IP}" "${speculation[@]}" "${decode_graph[@]}" --max-running-requests 256 >"${ART}/decode-0.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts decode-1)" bash "${ROOT}/source/worker.sh" decode-1 30201 "${MODEL}" 4 "${TAIL_IP}" "${speculation[@]}" "${decode_graph[@]}" --max-running-requests 256 >"${ART}/decode-1.log" 2>&1 &
worker_pids+=("$!")

wait_http "http://${HEAD_IP}:30100/health"; echo "prefill-0 ready"
wait_http "http://${TAIL_IP}:30200/health"; echo "decode-0 ready"
wait_http "http://${TAIL_IP}:30201/health"; echo "decode-1 ready"

srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts router)" /opt/sglang/venv/bin/python -m sglang_router.launch_router --pd-disaggregation --prefill "http://${HEAD_IP}:30100" 9000 --decode "http://${TAIL_IP}:30200" --decode "http://${TAIL_IP}:30201" --prefill-policy round_robin --host 0.0.0.0 --port "${R_PORT}" >"${ART}/router.log" 2>&1 &
worker_pids+=("$!")

wait_http "http://${HEAD_IP}:${R_PORT}/health"
echo "READY after ${SECONDS}s"
# The fixed source divides only chunked_prefill_size by DP; max_prefill_tokens stays local.
grep -F "chunked prefill size is adjusted from 16384 to 8192" "${ART}/prefill-0.log"
grep -F "max_prefill_tokens=16384" "${ART}/prefill-0.log"

while IFS=$'\t' read -r worker host port _tp; do
  curl --fail --silent --show-error --max-time 15 "http://${host}:${port}/get_server_info" >"${ROOT}/results/server-info-${worker}.json"
  jq -e --arg role "${worker%%-*}" --slurpfile config "${ROOT}/source/firmus-framework.json" '.tp_size == (if $role == "prefill" then 8 else 4 end) and .ep_size == (if $role == "prefill" then 8 else 1 end) and .kv_cache_dtype == "fp8_e4m3" and .context_length == 8192 and .max_total_tokens == (if $role == "decode" then 262144 else 131072 end) and .chunked_prefill_size == 8192 and .attention_backend == "fa3" and .speculative_algorithm == "EAGLE3" and .speculative_draft_model_path == "/draft" and .speculative_num_steps == $config[0].speculation.steps and .speculative_num_draft_tokens == $config[0].speculation.draft_tokens and .speculative_eagle_topk == 1 and .speculative_draft_model_quantization == null and .max_running_requests == (if $role == "prefill" then $config[0].concurrency else 256 end) and (if $role == "prefill" then .enable_symm_mem == false and .dp_size == 2 and .enable_dp_attention == true and .max_prefill_tokens == 16384 and .moe_a2a_backend == "none" and .moe_runner_backend == "triton" and .speculative_moe_a2a_backend == "none" and .speculative_moe_runner_backend == "triton" else .enable_symm_mem == false and .cuda_graph_bs_decode == $config[0].decode_cuda_graph_bs end)' "${ROOT}/results/server-info-${worker}.json" >/dev/null
done <"${ROOT}/source/worker-endpoints.tsv"
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${ART}/aiperf.log" 2>&1
echo "AIPERF ok"
touch "${ROOT}/run-success"
