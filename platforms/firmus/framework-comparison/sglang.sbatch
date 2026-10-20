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
# H200: MegaMoE Prefill on 3xDP4/EP4 and MegaMoE Decode on DP4/EP4.
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
# A placement file is a static permutation of the 128 experts over the four Prefill EP ranks; Decode keeps the trivial placement.
prefill_location=$(jq -er .prefill.expert_location "${ROOT}/source/firmus-framework.json")
if [[ "${prefill_location}" != trivial ]]; then prefill_location=${ROOT}/source/prefill-expert-location.json; fi
prefill_args=(--ep-size 4 --dp-size 4 --enable-dp-attention --chunked-prefill-size "$(jq -er .prefill.chunked_prefill_size "${ROOT}/source/firmus-framework.json")" --max-total-tokens 131072 --max-running-requests "${N}" --moe-a2a-backend "$(jq -er .prefill.moe_a2a_backend "${ROOT}/source/firmus-framework.json")" --moe-runner-backend "$(jq -er .prefill.moe_runner_backend "${ROOT}/source/firmus-framework.json")" --deepep-mode normal --disable-cuda-graph --init-expert-location "${prefill_location}" --optimistic-prefill-attempts "$(jq -er .prefill.optimistic_prefill_attempts "${ROOT}/source/firmus-framework.json")")
ART=${ROOT}/logs
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
export MELLANOX_VISIBLE_DEVICES=all MELLANOX_MOUNT_DRIVER=yes
# Tier 2 switch (megamoe-idle-prefill-geometry.patch): 256 masked rows on idle Prefill ranks.
# The code acts only in Prefill mode; with 1 the MegaMoE GPU gate also checks and times that case.
export SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY=1
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
  local profiler_mount= overlay_mount=
  if [[ "${PROFILE}" == true ]]; then profiler_mount=",${ROOT}/nsys:/opt/nsys:ro"; fi
  # Patched packages staged by provision-firmus-framework.sh (sglang_overlay_patches, deep_gemm_overlay_patches).
  if [[ -d "${ROOT}/overlay/sglang/python/sglang" ]]; then overlay_mount=",${ROOT}/overlay/sglang/python/sglang:/opt/sglang/source/python/sglang"; fi
  if [[ -d "${ROOT}/overlay/deep-gemm/deep_gemm" ]]; then overlay_mount+=",${ROOT}/overlay/deep-gemm/deep_gemm:/opt/sglang/venv/lib/python3.12/site-packages/deep_gemm"; fi
  mkdir -p "${ROOT}/home/$1"
  echo "/scratch:/scratch,${ROOT}/home/$1:${HOME},${ROOT}:${ROOT},${ROOT}/draft:/draft,/dev/infiniband:/dev/infiniband${profiler_mount}${overlay_mount}"
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
# Prefill: a 3994-token prompt must not straddle two 4096-token batches (prefill-request-aligned-chunks.patch; images without it ignore this).
# Tier 2 switches: equal-load DP ties rotate (dp-total-requests-rotate-ties.patch);
# Decode admission without host waits on the WAR-fenced stream (decode-async-admission.patch, with decode_polling_interval 1).
if [[ "$role" == prefill ]]; then export SGLANG_PREFILL_REQUEST_ALIGNED_CHUNKS=1 SGLANG_DP_TOTAL_REQUESTS_ROTATE_TIES=1; else export SGLANG_DECODE_ASYNC_ADMISSION=1; fi
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
echo "$name host=$(hostname) ip=$SGLANG_HOST_IP hca=$UCX_IB gpu=$CUDA_VISIBLE_DEVICES tp=$tp shuffle=${SGLANG_PREFILL_FUSED_FP8_SHUFFLE:-0} routing=${SGLANG_PREFILL_TILED_ROUTING:-0} aligned_chunks=${SGLANG_PREFILL_REQUEST_ALIGNED_CHUNKS:-0} rotate_ties=${SGLANG_DP_TOTAL_REQUESTS_ROTATE_TIES:-0} idle_prefill_geometry=${SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY:-0} async_admission=${SGLANG_DECODE_ASYNC_ADMISSION:-0} megamoe_tokens=${SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK}"
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
if [[ -x /opt/nsys/bin/nsys ]]; then
  runtime_dir=$(dirname "$(dirname "$0")")
  export QWEN_PROFILER_NSYS_BIN=/opt/nsys/bin/nsys
  export QWEN_PROFILER_INVOCATION_PATH="${runtime_dir}/profile/nsys-${name}.argv"
  command=(bash "${runtime_dir}/scripts/exec-profiler.sh" "${command[@]}")
fi
exec "${command[@]}"
EOF

# Provisioning verifies the image's source patches and router before sbatch.
# Validate the new FC2 kernel before model startup in this same allocation.
srun -N1 -n1 --exact -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts numeric-gate)" /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${ROOT}/scripts/check-firmus-megamoe.py" >"${ART}/sm90-megamoe-gate.log" 2>&1

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
done <"${ROOT}/source/worker-endpoints.tsv"
# Placement must not change outputs; keep one greedy answer through a Prefill worker for inspection.
curl --fail --silent --show-error --max-time 120 -H 'Content-Type: application/json' -d '{"model":"qwen","messages":[{"role":"user","content":"Name the capital of France and explain in two sentences why it is historically important."}],"max_tokens":64,"temperature":0}' "http://${HEAD_IP}:${R_PORT}/v1/chat/completions" >"${ROOT}/results/sanity-chat.json"
benchmark_status=0
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${ART}/aiperf.log" 2>&1 || benchmark_status=$?
test -s "${ROOT}/results/score.json"
if (( benchmark_status == 0 )); then
  while IFS=$'\t' read -r worker host port _; do curl --fail --silent --show-error --max-time 15 "http://${host}:${port}/get_server_info" >"${ROOT}/results/server-info-${worker}.json"; done <"${ROOT}/source/worker-endpoints.tsv"
  /usr/bin/python3 "${ROOT}/scripts/collect-eagle-acceptance.py" --cluster firmus --config "${ROOT}/source/firmus-framework.json" --endpoints "${ROOT}/source/worker-endpoints.tsv" --server-info-dir "${ROOT}/results" --log-dir "${ROOT}/logs" --output "${ROOT}/results/eagle-acceptance.json"
fi
if [[ "${PROFILE}" == true ]]; then
  "${ROOT}/scripts/capture-firmus-profile.sh" "${ROOT}" >"${ART}/profile-capture.log" 2>&1 &
  profile_capture_pid=$!
  # A failed profile client must not cancel capture or report flushing; keep both statuses.
  profile_client_status=0
  srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" profile >"${ART}/profile-aiperf.log" 2>&1 || profile_client_status=$?
  profile_capture_status=0
  wait "${profile_capture_pid}" || profile_capture_status=$?
  profile_capture_pid=
  printf 'profile_client_status=%s\nprofile_capture_status=%s\n' "${profile_client_status}" "${profile_capture_status}" >"${ART}/profile-status.txt"
  (( profile_capture_status == 0 )) || exit "${profile_capture_status}"
  expected_reports=$(wc -l <"${ROOT}/source/worker-endpoints.tsv")
  deadline=$((SECONDS + 180))
  while [[ "$(find "${ROOT}/profile" -maxdepth 1 -name '*.nsys-rep' -size +0c | wc -l)" -ne "${expected_reports}" ]]; do (( SECONDS < deadline )); sleep 2; done
  srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" /opt/sglang/venv/bin/python "${ROOT}/scripts/validate-firmus-profile.py" "${ROOT}" >"${ART}/profile-validation.log" 2>&1
  (( profile_client_status == 0 )) || exit "${profile_client_status}"
fi
if (( benchmark_status != 0 )); then exit "${benchmark_status}"; fi
echo "AIPERF ok"
touch "${ROOT}/run-success"
