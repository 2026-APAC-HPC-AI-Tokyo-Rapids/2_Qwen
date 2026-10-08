#!/bin/bash
#SBATCH -J sglang-aspire-conditions
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:8
#SBATCH --propagate=NONE
#SBATCH -o runs/sglang-%j.out
#SBATCH -e runs/sglang-%j.err
# Match ASPIRE run 37714793539: P2TP4/D2TP4 EP1, FP8 KV and standard prefill CUTLASS.
# EAGLE and custom kernel patches are excluded; use the official 60/120/30 client.
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR}
SGLANG=${ROOT}/sglang.sqsh
AIPERF=${ROOT}/sglang.sqsh
MODEL=/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8
R_PORT=8000
READY_SECS=600  # /health cap; typically 2-4 min, fail if still down at 10 min
N=$(jq -er .concurrency "${ROOT}/source/firmus-framework.json")
ART=${ROOT}/logs
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
  echo "/scratch:/scratch,${ROOT}/home/$1:${HOME},${ROOT}:${ROOT},/dev/infiniband:/dev/infiniband"
}

wait_http() {
  timeout "${READY_SECS}" curl -sf --retry-connrefused --retry-all-errors --retry-delay 2 --retry 300 --max-time 2 -o /dev/null "$1"
}

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} user=$(whoami)"
echo "head=${HEAD}(${HEAD_IP}) tail=${TAIL}(${TAIL_IP}) model=${MODEL} isl=4000 osl=200 N=${N} topology=P2TP4-D2TP4-EP1 kv=fp8_e4m3 speculation=none"
test -r "${SGLANG}" && test -r "${AIPERF}" && test -f "${MODEL}/config.json"
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
for worker in 0 1; do printf 'prefill-%s\t%s\t%s\t4\n' "${worker}" "${HEAD_IP}" "$((30100 + worker))"; done >"${ROOT}/source/worker-endpoints.tsv"
for worker in 0 1; do printf 'decode-%s\t%s\t%s\t4\n' "${worker}" "${TAIL_IP}" "$((30200 + worker))"; done >>"${ROOT}/source/worker-endpoints.tsv"

cat >"${ROOT}/source/worker.sh" <<'EOF'
set -euo pipefail
name=$1
port=$2
model=$3
tp=$4
shift 4
role=${name%%-*}
UCX_IB=$(for d in /sys/class/infiniband/*; do
  [ "$(cat "$d/ports/1/link_layer" 2>/dev/null)" = InfiniBand ] || continue
  [ "$(cat "$d/ports/1/state" 2>/dev/null)" = "4: ACTIVE" ] || continue
  echo "$(basename "$d"):1"
done | paste -sd,)
[ -n "$UCX_IB" ] || { echo "FATAL: no active IB on $(hostname)"; exit 1; }
export UCX_NET_DEVICES=$UCX_IB
export UCX_POSIX_USE_PROC_LINK=n
export TRITON_CACHE_DIR=/tmp/sglang-$name/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-$name/inductor
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
echo "$name host=$(hostname) ib=$UCX_IB gpu=$CUDA_VISIBLE_DEVICES tp=$tp"
exec /opt/sglang/venv/bin/python -m sglang.launch_server --model-path "$model" --served-model-name qwen --trust-remote-code --context-length 8192 --tp-size "$tp" --ep-size 1 --mem-fraction-static 0.90 --max-total-tokens 131072 --chunked-prefill-size 8192 --kv-cache-dtype fp8_e4m3 --attention-backend fa3 --nccl-port "$((port + 1000))" --disaggregation-mode "$role" --disaggregation-transfer-backend nixl "$@" --host 0.0.0.0 --port "$port"
EOF

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-image="${SGLANG}" --container-mounts="$(mounts prefill-0)" bash "${ROOT}/source/worker.sh" prefill-0 30100 "${MODEL}" 4 --disaggregation-bootstrap-port 9000 --max-running-requests "${N}" --moe-runner-backend cutlass >"${ART}/prefill-0.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-image="${SGLANG}" --container-mounts="$(mounts prefill-1)" bash "${ROOT}/source/worker.sh" prefill-1 30101 "${MODEL}" 4 --disaggregation-bootstrap-port 9001 --max-running-requests "${N}" --moe-runner-backend cutlass >"${ART}/prefill-1.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-image="${SGLANG}" --container-mounts="$(mounts decode-0)" bash "${ROOT}/source/worker.sh" decode-0 30200 "${MODEL}" 4 --max-running-requests 256 >"${ART}/decode-0.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-image="${SGLANG}" --container-mounts="$(mounts decode-1)" bash "${ROOT}/source/worker.sh" decode-1 30201 "${MODEL}" 4 --max-running-requests 256 >"${ART}/decode-1.log" 2>&1 &
worker_pids+=("$!")

wait_http "http://${HEAD_IP}:30100/health"; echo "prefill-0 ready"
wait_http "http://${HEAD_IP}:30101/health"; echo "prefill-1 ready"
wait_http "http://${TAIL_IP}:30200/health"; echo "decode-0 ready"
wait_http "http://${TAIL_IP}:30201/health"; echo "decode-1 ready"

srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 --container-image="${SGLANG}" --container-mounts="$(mounts router)" /opt/sglang/venv/bin/python -m sglang_router.launch_router --pd-disaggregation --prefill "http://${HEAD_IP}:30100" 9000 --prefill "http://${HEAD_IP}:30101" 9001 --decode "http://${TAIL_IP}:30200" --decode "http://${TAIL_IP}:30201" --prefill-policy round_robin --host 0.0.0.0 --port "${R_PORT}" >"${ART}/router.log" 2>&1 &
worker_pids+=("$!")

wait_http "http://${HEAD_IP}:${R_PORT}/health"
echo "READY after ${SECONDS}s"

while IFS=$'\t' read -r worker host port _tp; do
  curl --fail --silent --show-error --max-time 15 "http://${host}:${port}/get_server_info" >"${ROOT}/results/server-info-${worker}.json"
  jq -e --arg role "${worker%%-*}" --argjson concurrency "${N}" '.tp_size == 4 and .ep_size == 1 and .kv_cache_dtype == "fp8_e4m3" and .context_length == 8192 and .max_total_tokens == 131072 and .chunked_prefill_size == 8192 and .speculative_algorithm == null and .max_running_requests == (if $role == "prefill" then $concurrency else 256 end)' "${ROOT}/results/server-info-${worker}.json" >/dev/null
done <"${ROOT}/source/worker-endpoints.tsv"
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${ART}/aiperf.log" 2>&1
echo "AIPERF ok"
touch "${ROOT}/run-success"
