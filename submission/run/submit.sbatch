#!/bin/bash
#SBATCH -J sglang-firmus-dp4-ep4
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:8
#SBATCH --propagate=NONE
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%j.log
#SBATCH -e logs/slurm-%j.stderr.log
# H200: MegaMoE Prefill on 3xDP4/EP4 and MegaMoE Decode on DP4/EP4.
# One full benchmark allocation, with the official 60/120/30 client.
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR}
SGLANG=${ROOT}/sglang.sqsh
AIPERF=${ROOT}/sglang.sqsh
MODEL=$(jq -er .model.path "${ROOT}/source/model.json")
R_PORT=8000
READY_SECS=600  # ASPIRE full stack took 7m09s; leave time for the 60/120/30 client.
N=$(jq -er .concurrency "${ROOT}/source/firmus-framework.json")
# Parse configuration on the host; the pinned serving image does not include jq.
speculation=(--speculative-algorithm EAGLE3 --speculative-draft-model-path /draft --speculative-num-steps "$(jq -er '.speculation.steps' "${ROOT}/source/firmus-framework.json")" --speculative-eagle-topk 1 --speculative-num-draft-tokens "$(jq -er '.speculation.draft_tokens' "${ROOT}/source/firmus-framework.json")")
if [[ "$(jq -r '.speculation.draft_quantization // empty' "${ROOT}/source/firmus-framework.json")" == fp8 ]]; then speculation+=(--speculative-draft-model-quantization fp8); fi
decode_graph=(--cuda-graph-bs-decode)
while read -r size; do decode_graph+=("${size}"); done < <(jq -er '.decode_cuda_graph_bs[]' "${ROOT}/source/firmus-framework.json")
# Prefill maps 128 logical experts to physical slots, including hot replicas; Decode keeps trivial placement.
prefill_location=$(jq -er .prefill.expert_location "${ROOT}/source/firmus-framework.json")
if [[ "${prefill_location}" != trivial ]]; then prefill_location=${ROOT}/source/prefill-expert-location.json; fi
prefill_args=(--ep-size 4 --dp-size 4 --enable-dp-attention --chunked-prefill-size "$(jq -er .prefill.chunked_prefill_size "${ROOT}/source/firmus-framework.json")" --max-total-tokens 131072 --max-running-requests "${N}" --moe-a2a-backend "$(jq -er .prefill.moe_a2a_backend "${ROOT}/source/firmus-framework.json")" --moe-runner-backend "$(jq -er .prefill.moe_runner_backend "${ROOT}/source/firmus-framework.json")" --deepep-mode normal --disable-cuda-graph --init-expert-location "${prefill_location}" --ep-num-redundant-experts "$(jq -er .prefill.ep_num_redundant_experts "${ROOT}/source/firmus-framework.json")" --optimistic-prefill-attempts "$(jq -er .prefill.optimistic_prefill_attempts "${ROOT}/source/firmus-framework.json")")
ART=${ROOT}/logs
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
export MELLANOX_VISIBLE_DEVICES=all MELLANOX_MOUNT_DRIVER=yes
# Tier 2 switch: idle Prefill uses 256 rows, or 280 with physical-140 replicas.
# The replica overlay also pads tiny live Prefill ranks to 280 rows.
export SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY=1
jq -e '.framework == "sglang" and .workload == "official"' "${ROOT}/source/firmus-framework.json" >/dev/null

NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD=${NODES[0]}
TAIL=${NODES[1]}
ip_of() { getent ahostsv4 "$1" | awk '{print $1; exit}'; }
HEAD_IP=$(ip_of "${HEAD}")
TAIL_IP=$(ip_of "${TAIL}")

mounts() {
  local overlay_mount=
  # Patched packages staged by submission/run/prepare.sh.
  if [[ -d "${ROOT}/overlay/sglang/python/sglang" ]]; then overlay_mount=",${ROOT}/overlay/sglang/python/sglang:/opt/sglang/source/python/sglang"; fi
  if [[ -d "${ROOT}/overlay/deep-gemm/deep_gemm" ]]; then overlay_mount+=",${ROOT}/overlay/deep-gemm/deep_gemm:/opt/sglang/venv/lib/python3.12/site-packages/deep_gemm"; fi
  mkdir -p "${ROOT}/home/$1"
  echo "/scratch:/scratch,${ROOT}/home/$1:${HOME},${ROOT}:${ROOT},${ROOT}/draft:/draft,/dev/infiniband:/dev/infiniband${overlay_mount}"
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
echo "head=${HEAD}(${HEAD_IP}) tail=${TAIL}(${TAIL_IP}) model=${MODEL} isl=4000 osl=200 N=${N} topology=P3-TP4-DP4-EP4-D1-TP4-DP4-EP4 kv=fp8_e4m3 speculation=EAGLE3 prefill=megamoe decode=megamoe router=live-prefill-min-load"
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
# Prefill: a 3994-token prompt must not straddle two 4096-token batches (prefill-request-aligned-chunks.patch; images without it ignore this).
# Tier 2 switches: equal-load DP ties rotate (dp-total-requests-rotate-ties.patch);
# Decode admission without host waits on the WAR-fenced stream (decode-async-admission.patch, with decode_polling_interval 1).
# Prefill sets the same switch: it only reaches the min_new_tokens penalizer there, and prefill-async-admission.patch removes the other host waits that stalled each Prefill launch until the in-flight batch ended (~5 ms GPU idle per iteration).
# DG_SM90_PREFILL_2WG: the rebuilt DeepGEMM host launches the Qwen Prefill MegaMoE tile with two math warpgroups (sm90-prefill-2wg-host-epi.patch carries the header path).
# DG_SM90_DECODE_INDEPENDENT_WG: the rebuilt host gives the Decode SwapAB MegaMoE two independent math-WG task queues (sm90-decode-independent-wg host + header patches; kernel bench 39945097647: -2.3% at 128 tokens/rank).
if [[ "$role" == prefill ]]; then export SGLANG_PREFILL_REQUEST_ALIGNED_CHUNKS=1 SGLANG_DP_TOTAL_REQUESTS_ROTATE_TIES=1 SGLANG_DECODE_ASYNC_ADMISSION=1 DG_SM90_PREFILL_2WG=1; else export SGLANG_DECODE_ASYNC_ADMISSION=1 DG_SM90_DECODE_INDEPENDENT_WG=1; fi
# Decode kernel choices measured before serving by select-firmus-decode-kernels.sh (empty file = current kernels).
if [[ "$role" == decode ]]; then set -a; source "$(dirname "$0")/decode-kernels.env"; set +a; fi
# To opt in for a separate diagnostic run, uncomment the next line.
# if [[ "$role" == decode ]]; then export SGLANG_DECODE_ROUTING_DIR="${SLURM_SUBMIT_DIR}/results/decode-routing"; fi
# Optional graph-safe routing diagnostics: set SGLANG_DECODE_ROUTING_DIR to a run-dir path and create its active marker during measurement.
if [[ "$role" != decode ]]; then unset SGLANG_DECODE_ROUTING_DIR; fi
# The MegaMoE symmetric buffer must hold one Prefill batch per DP rank (chunked prefill size / DP size); Decode batches stay far below 4096 tokens per rank.
megamoe_tokens=4096
if [[ "$role" == prefill ]]; then args=" $* "; chunk=${args##* --chunked-prefill-size }; dp=${args##* --dp-size }; megamoe_tokens=$(( ${chunk%% *} / ${dp%% *} )); fi
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=${megamoe_tokens} SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1
UCX_IB=$(for d in /sys/class/infiniband/*; do
  [ "$(cat "$d/ports/1/state" 2>/dev/null)" = "4: ACTIVE" ] || continue
  echo "$(basename "$d"):1"
done | paste -sd,)
[ -n "$UCX_IB" ] || { echo "FATAL: no active HCA on $(hostname)"; exit 1; }
export UCX_NET_DEVICES=$UCX_IB
export UCX_POSIX_USE_PROC_LINK=n
# Both endpoints now own all KV heads (attention TP1). Use direct NIXL
# page transfer; heterogeneous-TP staging would only allocate unused pools.
export SGLANG_DISAGG_STAGING_BUFFER=0
export SGLANG_DG_CACHE_DIR="${SLURM_SUBMIT_DIR}/home/deep-gemm"
export TRITON_CACHE_DIR=/tmp/sglang-$name/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-$name/inductor
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
# Only Prefill sources the common result, screened on all twelve Prefill GPUs.
if [[ "$role" == prefill ]]; then set -a; source "$(dirname "$0")/prefill-kernels.env"; set +a; fi
echo "$name host=$(hostname) ip=$SGLANG_HOST_IP hca=$UCX_IB gpu=$CUDA_VISIBLE_DEVICES tp=$tp shuffle=${SGLANG_PREFILL_FUSED_FP8_SHUFFLE:-0} routing=${SGLANG_PREFILL_TILED_ROUTING:-0} aligned_chunks=${SGLANG_PREFILL_REQUEST_ALIGNED_CHUNKS:-0} rotate_ties=${SGLANG_DP_TOTAL_REQUESTS_ROTATE_TIES:-0} idle_prefill_geometry=${SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY:-0} async_admission=${SGLANG_DECODE_ASYNC_ADMISSION:-0} megamoe_tokens=${SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK} prefill_2wg=${DG_SM90_PREFILL_2WG:-0} dense_config=${DG_SM90_1D2D_DECODE_CONFIG:-stock} dense_qkv=${SGLANG_DECODE_DENSE_GEMM_QKV:-deepgemm} dense_o=${SGLANG_DECODE_DENSE_GEMM_O:-deepgemm} fa3_verify=${SGLANG_DECODE_FA3_VERIFY:-stock} prefill_fa3=${SGLANG_PREFILL_FA3_VERIFY:-control} independent_wg=${DG_SM90_DECODE_INDEPENDENT_WG:-0}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit --format=csv
# Queue-aware DP routing replaces bootstrap-room hashing on P and round-robin
# on D. Publish every iteration so completed requests release routing budget.
# Native P registration and D batched lookup preserve the actual KV source rank.
command=(/opt/sglang/venv/bin/python -m sglang.launch_server --model-path "$model" --served-model-name qwen --trust-remote-code --context-length 8192 --tp-size "$tp" --mem-fraction-static 0.90 --page-size 64 --kv-cache-dtype fp8_e4m3 --attention-backend fa3 --nccl-port "$((port + 1000))" --disaggregation-mode "$role" --disaggregation-transfer-backend nixl --load-balance-method total_requests --load-snapshot-publish-interval 1 "$@" --host 0.0.0.0 --port "$port")
# DP attention derives six TCP endpoints from this address.
# Adjacent HTTP ports need separate endpoint ranges on the same node.
# Keep all six endpoints below the Firmus ephemeral range (32768-60999).
if [[ "$role" == prefill ]]; then
  command+=(--dist-init-addr "127.0.0.1:$((32000 + ${name##*-} * 100))")
else
  command+=(--dist-init-addr "127.0.0.1:29000")
fi
exec "${command[@]}"
EOF

# Kernel choices measured in Actions run 40225743220; workers source their role's file.
printf '%s\n' 'SGLANG_PREFILL_FA3_VERIFY=n192' >"${ROOT}/source/prefill-kernels.env"
printf '%s\n' 'SGLANG_DECODE_FA3_VERIFY=p128split2-s3-fp8' 'DG_SM90_1D2D_DECODE_CONFIG=QKV:128,144,5,1,0' 'SGLANG_DECODE_DENSE_GEMM_QKV=' 'SGLANG_DECODE_DENSE_GEMM_O=' >"${ROOT}/source/decode-kernels.env"

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts prefill-0)" bash "${ROOT}/source/worker.sh" prefill-0 30100 "${MODEL}" 4 "${HEAD_IP}" "${speculation[@]}" "${prefill_args[@]}" --disaggregation-bootstrap-port 9000 >"${ART}/prefill-0.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts prefill-1)" bash "${ROOT}/source/worker.sh" prefill-1 30101 "${MODEL}" 4 "${HEAD_IP}" "${speculation[@]}" "${prefill_args[@]}" --disaggregation-bootstrap-port 9001 >"${ART}/prefill-1.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts prefill-2)" bash "${ROOT}/source/worker.sh" prefill-2 30102 "${MODEL}" 4 "${TAIL_IP}" "${speculation[@]}" "${prefill_args[@]}" --disaggregation-bootstrap-port 9002 >"${ART}/prefill-2.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts decode-0)" bash "${ROOT}/source/worker.sh" decode-0 30200 "${MODEL}" 4 "${TAIL_IP}" "${speculation[@]}" "${decode_graph[@]}" --ep-size 4 --dp-size "$(jq -er .decode_dp_size "${ROOT}/source/firmus-framework.json")" --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1 --moe-a2a-backend megamoe --moe-runner-backend deep_gemm --chunked-prefill-size 8192 --max-running-requests "$(jq -er .decode_max_running_requests "${ROOT}/source/firmus-framework.json")" --disaggregation-decode-extra-slots "$(jq -er .decode_extra_slots "${ROOT}/source/firmus-framework.json")" --disaggregation-decode-polling-interval "$(jq -er .decode_polling_interval "${ROOT}/source/firmus-framework.json")" --max-total-tokens "$(jq -er .decode_max_total_tokens "${ROOT}/source/firmus-framework.json")" >"${ART}/decode-0.log" 2>&1 &
worker_pids+=("$!")

wait_http "http://${HEAD_IP}:30100/health"; echo "prefill-0 ready"
wait_http "http://${HEAD_IP}:30101/health"; echo "prefill-1 ready"
wait_http "http://${TAIL_IP}:30102/health"; echo "prefill-2 ready"
wait_http "http://${TAIL_IP}:30200/health"; echo "decode-0 ready"

srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts router)" /opt/apac/bin/sglang-router launch --pd-disaggregation --prefill "http://${HEAD_IP}:30100" 9000 --prefill "http://${HEAD_IP}:30101" 9001 --prefill "http://${TAIL_IP}:30102" 9002 --decode "http://${TAIL_IP}:30200" --prefill-policy manual --assignment-mode min_load --host 0.0.0.0 --port "${R_PORT}" >"${ART}/router.log" 2>&1 &
worker_pids+=("$!")

# /health is liveness only; /readiness waits until a Prefill and a Decode worker are registered (39074570813 got 503 1 s after /health).
wait_http "http://${HEAD_IP}:${R_PORT}/readiness"
echo "READY after ${SECONDS}s"

while IFS=$'\t' read -r worker host port _tp; do
  curl --fail --silent --show-error --max-time 15 "http://${host}:${port}/get_server_info" >"${ROOT}/results/server-info-${worker}.json"
  if [[ "$(jq -r '.speculation.draft_quantization // empty' "${ROOT}/source/firmus-framework.json")" == fp8 ]]; then jq -e '.speculative_draft_model_quantization == "fp8"' "${ROOT}/results/server-info-${worker}.json" >/dev/null; fi
done <"${ROOT}/source/worker-endpoints.tsv"
# Placement must not change outputs; keep one greedy answer through a Prefill worker for inspection.
curl --fail --silent --show-error --max-time 120 -H 'Content-Type: application/json' -d '{"model":"qwen","messages":[{"role":"user","content":"Name the capital of France and explain in two sentences why it is historically important."}],"max_tokens":64,"temperature":0}' "http://${HEAD_IP}:${R_PORT}/v1/chat/completions" >"${ROOT}/results/sanity-chat.json"
benchmark_status=0
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${ART}/aiperf.log" 2>&1 || benchmark_status=$?
test -s "${ROOT}/results/score.json"
if (( benchmark_status == 0 )); then
  while IFS=$'\t' read -r worker host port _; do curl --fail --silent --show-error --max-time 15 "http://${host}:${port}/get_server_info" >"${ROOT}/results/server-info-${worker}.json"; done <"${ROOT}/source/worker-endpoints.tsv"
fi
if (( benchmark_status != 0 )); then exit "${benchmark_status}"; fi
echo "AIPERF ok"
touch "${ROOT}/run-success"
