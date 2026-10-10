#!/bin/bash
#SBATCH -J sglang-firmus-dp4-ep4
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:8
#SBATCH --propagate=NONE
#SBATCH --no-requeue
#SBATCH -o runs/sglang-%j.out
#SBATCH -e runs/sglang-%j.err
# H200: three 4-GPU attention-DP4/MoE-EP4 prefill groups and one TP4 decode group.
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
prefill_args=(--ep-size 4 --dp-size 4 --enable-dp-attention --chunked-prefill-size 16384 --max-total-tokens 131072 --max-running-requests "${N}" --moe-a2a-backend "$(jq -er .prefill.moe_a2a_backend "${ROOT}/source/firmus-framework.json")" --moe-runner-backend "$(jq -er .prefill.moe_runner_backend "${ROOT}/source/firmus-framework.json")" --deepep-mode normal --disable-cuda-graph)
ART=${ROOT}/logs
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
export MELLANOX_VISIBLE_DEVICES=all MELLANOX_MOUNT_DRIVER=yes
PROFILE=${PROFILE:-false}
[[ "${PROFILE}" == true || "${PROFILE}" == false ]]
jq -e '.framework == "sglang" and .workload == "official"' "${ROOT}/source/firmus-framework.json" >/dev/null

NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD=${NODES[0]}
TAIL=${NODES[1]}
ip_of() { getent ahostsv4 "$1" | awk '{print $1; exit}'; }
HEAD_IP=$(ip_of "${HEAD}")
TAIL_IP=$(ip_of "${TAIL}")

mounts() {
  local profiler_mount=
  if [[ "${PROFILE}" == true ]]; then profiler_mount=",${ROOT}/nsys:/opt/nsys:ro"; fi
  mkdir -p "${ROOT}/home/$1"
  echo "/scratch:/scratch,${ROOT}/home/$1:${HOME},${ROOT}:${ROOT},${ROOT}/draft:/draft,/dev/infiniband:/dev/infiniband${profiler_mount}"
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
echo "head=${HEAD}(${HEAD_IP}) tail=${TAIL}(${TAIL_IP}) model=${MODEL} isl=4000 osl=200 N=${N} topology=P3-TP4-DP4-EP4-D1TP4-EP1 kv=fp8_e4m3 speculation=EAGLE3 kernels=aspire-full"
test -r "${SGLANG}" && test -r "${AIPERF}" && test -f "${MODEL}/config.json" && test -s "${ROOT}/draft/model.safetensors"
mkdir -p "${ART}"
worker_pids=()
profile_capture_pid=
cleanup_all() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${profile_capture_pid}" ]]; then kill "${profile_capture_pid}" 2>/dev/null || true; wait "${profile_capture_pid}" 2>/dev/null || true; fi
  for pid in "${worker_pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${worker_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup_all EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "${PROFILE}" == true ]]; then
  "${ROOT}/nsys/bin/nsys" --version >"${ROOT}/profile/nsys-version.txt"
  printf 'Separate normal benchmark (60/120/30), then the same serving configuration with a profile-only client (20/30/90). Capture starts five seconds into the profiling phase and lasts five seconds after all start acknowledgements. CUDA graph nodes and Torch CPU stage spans cover every TP rank. Exclude start/stop/export tails; profile/client is not an adoption score. The normal pass has tracing disabled, but uses the Nsight launcher.\n' >"${ROOT}/profile/interpretation.txt"
fi
for worker in 0 1; do printf 'prefill-%s\t%s\t%s\t4\n' "${worker}" "${HEAD_IP}" "$((30100 + worker))"; done >"${ROOT}/source/worker-endpoints.tsv"
for worker in 2; do printf 'prefill-%s\t%s\t%s\t4\n' "${worker}" "${TAIL_IP}" "$((30100 + worker))"; done >>"${ROOT}/source/worker-endpoints.tsv"
printf 'decode-0\t%s\t30200\t4\n' "${TAIL_IP}" >>"${ROOT}/source/worker-endpoints.tsv"

cat >"${ROOT}/source/worker.sh" <<'EOF'
set -euo pipefail
name=$1
port=$2
model=$3
tp=$4
export SGLANG_HOST_IP=$5
shift 5
role=${name%%-*}
UCX_IB=$(for d in /sys/class/infiniband/*; do
  [ "$(cat "$d/ports/1/state" 2>/dev/null)" = "4: ACTIVE" ] || continue
  echo "$(basename "$d"):1"
done | paste -sd,)
[ -n "$UCX_IB" ] || { echo "FATAL: no active HCA on $(hostname)"; exit 1; }
export UCX_NET_DEVICES=$UCX_IB
export UCX_POSIX_USE_PROC_LINK=n
export SGLANG_DISAGG_STAGING_BUFFER=1 SGLANG_DISAGG_STAGING_POOL_SIZE_MB=1024
export SGLANG_DG_CACHE_DIR="${SLURM_SUBMIT_DIR}/home/deep-gemm"
if [[ "$role" == prefill ]]; then
  export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096 SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1
fi
export TRITON_CACHE_DIR=/tmp/sglang-$name/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-$name/inductor
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
echo "$name host=$(hostname) ip=$SGLANG_HOST_IP hca=$UCX_IB gpu=$CUDA_VISIBLE_DEVICES tp=$tp shuffle=${SGLANG_PREFILL_FUSED_FP8_SHUFFLE:-0} routing=${SGLANG_PREFILL_TILED_ROUTING:-0}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit --format=csv
command=(/opt/sglang/venv/bin/python -m sglang.launch_server --model-path "$model" --served-model-name qwen --trust-remote-code --context-length 8192 --tp-size "$tp" --mem-fraction-static 0.90 --page-size 64 --kv-cache-dtype fp8_e4m3 --attention-backend fa3 --nccl-port "$((port + 1000))" --disaggregation-mode "$role" --disaggregation-transfer-backend nixl "$@" --host 0.0.0.0 --port "$port")
if [[ "$role" == prefill ]]; then
  # DP attention derives six TCP endpoints from this address. Adjacent HTTP
  # ports would otherwise create overlapping endpoint ranges on the same node.
  command+=(--dist-init-addr "127.0.0.1:$((32000 + ${name##*-} * 100))")
fi
if [[ -x /opt/nsys/bin/nsys ]]; then
  runtime_dir=$(dirname "$(dirname "$0")")
  export QWEN_PROFILER_NSYS_BIN=/opt/nsys/bin/nsys
  export QWEN_PROFILER_INVOCATION_PATH="${runtime_dir}/profile/nsys-${name}.argv"
  command=(bash "${runtime_dir}/scripts/exec-profiler.sh" "${command[@]}")
fi
exec "${command[@]}"
EOF

# Part of this allocation: a finite numeric gate, then the full benchmark.
# All later workers reuse its native compilation cache. No extra sbatch or retry.
srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts megamoe-check)" timeout --signal=TERM --kill-after=15 240 env DG_JIT_CACHE_DIR="${ROOT}/home/deep-gemm" /opt/sglang/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 "${ROOT}/scripts/check-firmus-megamoe.py" >"${ART}/megamoe-numeric-check.log" 2>&1
echo "SM90 MegaMoE numeric gate passed after ${SECONDS}s"

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts prefill-0)" bash "${ROOT}/source/worker.sh" prefill-0 30100 "${MODEL}" 4 "${HEAD_IP}" "${speculation[@]}" "${prefill_args[@]}" --disaggregation-bootstrap-port 9000 >"${ART}/prefill-0.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts prefill-1)" bash "${ROOT}/source/worker.sh" prefill-1 30101 "${MODEL}" 4 "${HEAD_IP}" "${speculation[@]}" "${prefill_args[@]}" --disaggregation-bootstrap-port 9001 >"${ART}/prefill-1.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts prefill-2)" bash "${ROOT}/source/worker.sh" prefill-2 30102 "${MODEL}" 4 "${TAIL_IP}" "${speculation[@]}" "${prefill_args[@]}" --disaggregation-bootstrap-port 9002 >"${ART}/prefill-2.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts decode-0)" bash "${ROOT}/source/worker.sh" decode-0 30200 "${MODEL}" 4 "${TAIL_IP}" "${speculation[@]}" "${decode_graph[@]}" --ep-size 1 --chunked-prefill-size 8192 --max-running-requests 256 --max-total-tokens "$(jq -er .decode_max_total_tokens "${ROOT}/source/firmus-framework.json")" >"${ART}/decode-0.log" 2>&1 &
worker_pids+=("$!")

wait_http "http://${HEAD_IP}:30100/health"; echo "prefill-0 ready"
wait_http "http://${HEAD_IP}:30101/health"; echo "prefill-1 ready"
wait_http "http://${TAIL_IP}:30102/health"; echo "prefill-2 ready"
wait_http "http://${TAIL_IP}:30200/health"; echo "decode-0 ready"

srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts router)" /opt/sglang/venv/bin/python -m sglang_router.launch_router --pd-disaggregation --prefill "http://${HEAD_IP}:30100" 9000 --prefill "http://${HEAD_IP}:30101" 9001 --prefill "http://${TAIL_IP}:30102" 9002 --decode "http://${TAIL_IP}:30200" --prefill-policy round_robin --host 0.0.0.0 --port "${R_PORT}" >"${ART}/router.log" 2>&1 &
worker_pids+=("$!")

wait_http "http://${HEAD_IP}:${R_PORT}/health"
echo "READY after ${SECONDS}s"

while IFS=$'\t' read -r worker host port _tp; do
  curl --fail --silent --show-error --max-time 15 "http://${host}:${port}/get_server_info" >"${ROOT}/results/server-info-${worker}.json"
  python3 "${ROOT}/scripts/check-firmus-server-info.py" "${ROOT}/source/firmus-framework.json" "${ROOT}/results/server-info-${worker}.json" "${worker%%-*}"
done <"${ROOT}/source/worker-endpoints.tsv"
benchmark_status=0
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${ART}/aiperf.log" 2>&1 || benchmark_status=$?
test -s "${ROOT}/results/score.json"
if [[ "${PROFILE}" == true ]]; then
  "${ROOT}/scripts/capture-firmus-profile.sh" "${ROOT}" >"${ART}/profile-capture.log" 2>&1 &
  profile_capture_pid=$!
  srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" profile >"${ART}/profile-aiperf.log" 2>&1
  wait "${profile_capture_pid}"
  profile_capture_pid=
  expected_reports=$(wc -l <"${ROOT}/source/worker-endpoints.tsv")
  deadline=$((SECONDS + 180))
  while [[ "$(find "${ROOT}/profile" -maxdepth 1 -name '*.nsys-rep' -size +0c | wc -l)" -ne "${expected_reports}" ]]; do (( SECONDS < deadline )); sleep 2; done
  srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" /opt/sglang/venv/bin/python "${ROOT}/scripts/validate-firmus-profile.py" "${ROOT}" >"${ART}/profile-validation.log" 2>&1
fi
if (( benchmark_status != 0 )); then exit "${benchmark_status}"; fi
echo "AIPERF ok"
touch "${ROOT}/run-success"
