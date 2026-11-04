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
if [[ "$(jq -r '.speculation.draft_quantization // empty' "${ROOT}/source/firmus-framework.json")" == fp8 ]]; then speculation+=(--speculative-draft-model-quantization fp8); fi
tokenizer_backend="$(jq -r '.tokenizer_backend // "huggingface"' "${ROOT}/source/firmus-framework.json")"
tokenizer_args=()
if [[ "${tokenizer_backend}" != huggingface ]]; then tokenizer_args+=(--tokenizer-backend "${tokenizer_backend}"); fi
decode_graph=(--cuda-graph-bs-decode)
while read -r size; do decode_graph+=("${size}"); done < <(jq -er '.decode_cuda_graph_bs[]' "${ROOT}/source/firmus-framework.json")
# Prefill maps 128 logical experts to physical slots, including hot replicas; Decode keeps trivial placement.
prefill_location=$(jq -er .prefill.expert_location "${ROOT}/source/firmus-framework.json")
if [[ "${prefill_location}" != trivial ]]; then prefill_location=${ROOT}/source/prefill-expert-location.json; fi
prefill_args=(--ep-size 4 --dp-size 4 --enable-dp-attention --chunked-prefill-size "$(jq -er .prefill.chunked_prefill_size "${ROOT}/source/firmus-framework.json")" --max-total-tokens 131072 --max-running-requests "${N}" --moe-a2a-backend "$(jq -er .prefill.moe_a2a_backend "${ROOT}/source/firmus-framework.json")" --moe-runner-backend "$(jq -er .prefill.moe_runner_backend "${ROOT}/source/firmus-framework.json")" --deepep-mode normal --disable-cuda-graph --init-expert-location "${prefill_location}" --ep-num-redundant-experts "$(jq -er .prefill.ep_num_redundant_experts "${ROOT}/source/firmus-framework.json")" --optimistic-prefill-attempts "$(jq -er .prefill.optimistic_prefill_attempts "${ROOT}/source/firmus-framework.json")")
prefill_tp=$(jq -er '.prefill.tp' "${ROOT}/source/firmus-framework.json")
prefill_pp=$(jq -er '.prefill.pp // 1' "${ROOT}/source/firmus-framework.json")
if [[ "${prefill_pp}" == 4 ]]; then
  [[ "${prefill_tp}" == 1 ]]
  prefill_args=(--pp-size 4 --pp-max-micro-batch-size 2 --pp-async-batch-depth "$(jq -er .prefill.pp_async_batch_depth "${ROOT}/source/firmus-framework.json")" --ep-size 1 --dp-size 1 --chunked-prefill-size 8192 --max-total-tokens 131072 --max-running-requests "${N}" --moe-a2a-backend megamoe --moe-runner-backend deep_gemm --disable-cuda-graph --disable-overlap-schedule --init-expert-location trivial --ep-num-redundant-experts 0 --optimistic-prefill-attempts 0)
fi
ART=${ROOT}/logs
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
export MELLANOX_VISIBLE_DEVICES=all MELLANOX_MOUNT_DRIVER=yes
# Tier 2 switch: idle Prefill uses 256 rows, or 280 with physical-140 replicas.
# The replica overlay also pads tiny live Prefill ranks to 280 rows.
# The code acts only in Prefill mode; with 1 the MegaMoE GPU gate also checks and times that case.
export SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY=1
PROFILE=${PROFILE:-false}
[[ "${PROFILE}" == true || "${PROFILE}" == false ]]
profile_only=$(jq -r '.profile_only // false' "${ROOT}/source/firmus-framework.json")
if [[ "${profile_only}" == true ]]; then [[ "${PROFILE}" == true ]]; fi
routing_capture_samples=$(jq -er '.prefill.routing_capture_samples // 0' "${ROOT}/source/firmus-framework.json")
[[ "${routing_capture_samples}" =~ ^[0-8]$ ]]
if (( routing_capture_samples > 0 )); then [[ "${PROFILE}" == true ]]; fi
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
echo "head=${HEAD}(${HEAD_IP}) tail=${TAIL}(${TAIL_IP}) model=${MODEL} isl=4000 osl=200 N=${N} topology=P3-TP${prefill_tp}-PP${prefill_pp}-D1-TP4-DP4-EP4 kv=fp8_e4m3 speculation=EAGLE3 prefill=megamoe decode=megamoe router=live-prefill-min-load"
test -r "${SGLANG}" && test -r "${AIPERF}" && test -f "${MODEL}/config.json" && test -s "${ROOT}/draft/model.safetensors"
mkdir -p "${ART}"
worker_pids=()
profile_capture_pid=
telemetry_pids=()
cleanup_all() {
  status=$?
  trap - EXIT INT TERM
  for worker in prefill-0 prefill-1 prefill-2; do rm -f "${ROOT}/results/prefill-routing/${worker}/active"; done
  if [[ -n "${profile_capture_pid}" ]]; then kill "${profile_capture_pid}" 2>/dev/null || true; wait "${profile_capture_pid}" 2>/dev/null || true; fi
  for pid in "${telemetry_pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${telemetry_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
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
  if [[ "${profile_only}" == true && "${prefill_pp}" == 4 ]]; then printf 'Profile-only PP4: ordinary benchmark is omitted; no new official score. Client 20/30/90; CUDA_PROFILER only. Start 5 s after profiling phase begins, capture 8 s after all ACKs; analyze the inner 7 s. NVML SM clock/power at target 10 Hz with UTC timestamps, Nsight GPU counters at 1000 Hz if permitted.\n' >"${ROOT}/profile/interpretation.txt"; elif [[ "${profile_only}" == true ]]; then printf 'Profile-only recovery: ordinary benchmark omitted; no new score. Client 20/30/90.\n' >"${ROOT}/profile/interpretation.txt"; fi
  if (( routing_capture_samples > 0 )); then printf 'Prefill routing capture copies physical top-8 to the host during the separate profile pass. These Nsight timings contain routing copy overhead and must not be used for performance adoption.\n' >>"${ROOT}/profile/interpretation.txt"; fi
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
runtime_dir=$(dirname "$(dirname "$0")")
# The Decode peer also needs explicit global target/draft KV region IDs.
if /opt/sglang/venv/bin/python -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("prefill", {}).get("pp", 1) != 4)' "${runtime_dir}/source/firmus-framework.json"; then
  export SGLANG_QWEN_PP4_EP1=1
  # Millisecond scheduler log times (PP4 only): the start-burst first tokens of 41020974704 stall from 0.8 to
  # 1.5 s, and second-resolution batch lines cannot place the Prefill admissions and releases in that window.
  export SGLANG_LOG_MS=1
  if [[ "$role" == prefill ]]; then
    # The last stage also runs lm_head and the EAGLE draft extend (about two layers of time).
    partition=$(/opt/sglang/venv/bin/python -c 'import json,sys; print(",".join(map(str, json.load(open(sys.argv[1]))["prefill"]["pp_layer_partition"])))' "${runtime_dir}/source/firmus-framework.json")
    export DG_SM90_PREFILL_LOCAL_EP1=1 SGLANG_PP_LAYER_PARTITION="${partition}" SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=0
    # Stage s runs on visible device s. The lightest (last) stage moves to this worker's slowest GPU at the power cap (order-firmus-pp4-gpus.py).
    if [[ -s "${runtime_dir}/source/firmus-gpu-sm-clock.json" ]]; then
      gpu_order=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | /opt/sglang/venv/bin/python "${runtime_dir}/scripts/order-firmus-pp4-gpus.py" "${runtime_dir}/source/firmus-gpu-sm-clock.json" "${CUDA_VISIBLE_DEVICES}")
      export CUDA_VISIBLE_DEVICES="${gpu_order}"
    fi
  fi
fi
# Keep inherited diagnostic variables out of Decode and ordinary benchmark workers.
unset SGLANG_PREFILL_ROUTING_DIR SGLANG_PREFILL_ROUTING_SAMPLES
routing_capture_samples=$(/opt/sglang/venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("prefill", {}).get("routing_capture_samples", 0))' "${runtime_dir}/source/firmus-framework.json")
if [[ "$role" == prefill ]] && (( routing_capture_samples > 0 )); then
  export SGLANG_PREFILL_ROUTING_DIR="${runtime_dir}/results/prefill-routing/${name}" SGLANG_PREFILL_ROUTING_SAMPLES="${routing_capture_samples}"
  mkdir -p "${SGLANG_PREFILL_ROUTING_DIR}"
fi
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
if [[ -x /opt/nsys/bin/nsys ]]; then
  # Record logical PP order after the power-cap GPU reorder; NVML UUIDs are not Nsight export UUIDs.
  for device in ${CUDA_VISIBLE_DEVICES//,/ }; do nvidia-smi --id="${device}" --query-gpu=uuid,pci.bus_id --format=csv,noheader; done >"${runtime_dir}/profile/gpu-map-${name}.csv"
  export QWEN_PROFILER_NSYS_BIN=/opt/nsys/bin/nsys
  export QWEN_PROFILER_INVOCATION_PATH="${runtime_dir}/profile/nsys-${name}.argv"
  command=(bash "${runtime_dir}/scripts/exec-profiler.sh" "${command[@]}")
fi
exec "${command[@]}"
EOF

# One low-frequency NVML query process per node, outside the serving workers.
# Explicit GPU UUIDs identify all eight GPUs on each node without rank duplication.
telemetry_interval=(--loop=1)
if [[ "${PROFILE}" == true ]]; then telemetry_interval=(--loop-ms=100); fi
for host in "${HEAD}" "${TAIL}"; do
  srun --overlap --exact -N1 -n1 -w "${host}" --gres=gpu:8 --cpus-per-task=1 env TZ=UTC nvidia-smi --query-gpu=timestamp,index,uuid,power.draw,clocks.sm,clocks.mem,clocks_throttle_reasons.active,temperature.gpu --format=csv "${telemetry_interval[@]}" >"${ART}/gpu-telemetry-${host}.csv" 2>"${ART}/gpu-telemetry-${host}.stderr.log" &
  telemetry_pids+=("$!")
done

# Provisioning verifies the image's source patches and router before sbatch.
# Validate the new FC2 kernel before model startup in this same allocation.
# BeeGFS shared-cache replacement can transiently fail dlopen (39727354451;
# reproduced without GPUs). All four ranks share a node-local cache instead.
# Decode kernel selection (select-firmus-decode-kernels.sh, results/decode-selection/). The 17-minute site
# limit leaves no serial time: "fa3" runs on the head node's other four GPUs during the numeric gate, and
# "dense" on the Decode GPUs while the Prefill workers start. Only the Decode worker sources the result.
if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then
  # Every selection gate has settled: 41005872470, 41020974704 and 41040348302; the fixed FA3 pipe
  # passed its 1200-call gate and served in 41040348302. That run spent 7.3 min selecting, then hit TIMEOUT.
  test -s "${ROOT}/source/prefill-kernels.env" && test -s "${ROOT}/source/decode-kernels.env"
  printf 'PP4 serving: prior kernel settings plus the pinned PP4 selections (prepare-firmus-pp4-settings.py); no repeated kernel sweep; optional pair gate follows.\n'
elif [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]; then
  # Provision has verified the prior image, ordered overlays, expert map and
  # normalized serving config. Draft precision is the sole changed model setting.
  test -s "${ROOT}/source/prefill-kernels.env" && test -s "${ROOT}/source/decode-kernels.env"
  printf 'Reusing verified target kernel selections from run %s; no repeated target kernel sweep or numeric gate.\n' "$(jq -r .source_run "${ROOT}/source/reused-kernel-selection.json")"
else
srun -N1 -n1 --exact -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts decode-selection-fa3)" bash "${ROOT}/scripts/select-firmus-decode-kernels.sh" "${ROOT}" fa3 >"${ART}/decode-selection-fa3.log" 2>&1 &
fa3_selection=$!
srun -N1 -n1 --exact -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts decode-selection-dense)" bash "${ROOT}/scripts/select-firmus-decode-kernels.sh" "${ROOT}" dense >"${ART}/decode-selection-dense.log" 2>&1 &
dense_selection=$!
# --dense-selected: the dense candidates were checked by the selection; the gate keeps the configured path.
srun -N1 -n1 --exact -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts numeric-gate)" env TRITON_CACHE_DIR="/tmp/qwen-numeric-gate-${SLURM_JOB_ID}" DG_SM90_DECODE_INDEPENDENT_WG=1 /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${ROOT}/scripts/check-firmus-megamoe.py" --dense-selected --fa3-selected >"${ART}/sm90-megamoe-gate.log" 2>&1
wait "${fa3_selection}"
fi

if [[ ! -s "${ROOT}/source/reused-kernel-selection.json" && ! -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then
# Measure all twelve physical Prefill GPUs before model load. The two head
# groups and one tail group run concurrently; serving then reuses these GPUs.
prefill_selections=()
for worker in 0 1 2; do
  node=${HEAD}
  if (( worker == 2 )); then node=${TAIL}; fi
  srun -N1 -n1 --exact -w "${node}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts prefill-selection-${worker})" env TRITON_CACHE_DIR="/tmp/qwen-prefill-select-${SLURM_JOB_ID}-${worker}" bash "${ROOT}/scripts/select-firmus-prefill-kernels.sh" "${ROOT}" "prefill-${worker}" >"${ART}/prefill-selection-${worker}.log" 2>&1 &
  prefill_selections+=("$!")
done
for pid in "${prefill_selections[@]}"; do wait "${pid}" || true; done
uv run --no-project --python 3.12 python "${ROOT}/scripts/select-firmus-prefill-fa3.py" --aggregate "${ROOT}/results/prefill-selection" --output "${ROOT}/results/prefill-selection/selection.json" --env-output "${ROOT}/source/prefill-kernels.env"
fi

# Default off. The pair-only FA3 gate may replace the common pin before model load.
if [[ "${prefill_pp}" == 4 && "$(jq -r '.prefill.fa3_pair_gate // false' "${ROOT}/source/firmus-framework.json")" == true ]]; then
  test -s "${ROOT}/source/pp4-kernel-settings.json"
  mkdir -p "${ROOT}/results/prefill-fa3-pair"
  prefill_fa3_pair_started=$(date +%s)
  prefill_fa3_pair_deadline=$((prefill_fa3_pair_started + 155))
  prefill_fa3_pair_selections=()
  # Three four-GPU workers in parallel. Estimate 60-85 s including cold imports/JIT and endurance.
  # Bound each whole srun to 150 s plus 2 s kill grace (88 s left no margin over the estimate, and a timeout
  # would spend the build and the run without an answer); timeout retains the control without failing serving.
  for worker in 0 1 2; do
    node=${HEAD}
    if (( worker == 2 )); then node=${TAIL}; fi
    timeout --kill-after=2 150 srun -N1 -n1 --exact -w "${node}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts prefill-fa3-pair-${worker})" env TRITON_CACHE_DIR="/tmp/qwen-prefill-fa3-pair-${SLURM_JOB_ID}-${worker}/triton" TVM_FFI_CACHE_DIR="/tmp/qwen-prefill-fa3-pair-${SLURM_JOB_ID}-${worker}/tvm" /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${ROOT}/scripts/select-firmus-prefill-fa3-pair.py" --output "${ROOT}/results/prefill-fa3-pair/prefill-${worker}" --run-id "${SLURM_JOB_ID}-${prefill_fa3_pair_started}" --deadline "${prefill_fa3_pair_deadline}" >"${ART}/prefill-fa3-pair-${worker}.log" 2>&1 &
    prefill_fa3_pair_selections+=("$!")
  done
  prefill_fa3_pair_ok=1
  for pid in "${prefill_fa3_pair_selections[@]}"; do wait "${pid}" || prefill_fa3_pair_ok=0; done
  uv run --no-project --python 3.12 python "${ROOT}/scripts/select-firmus-prefill-fa3-pair.py" --aggregate "${ROOT}/results/prefill-fa3-pair" --output "${ROOT}/results/prefill-fa3-pair/selection.json" --env-output "${ROOT}/source/prefill-kernels.env" --run-id "${SLURM_JOB_ID}-${prefill_fa3_pair_started}" --deadline "${prefill_fa3_pair_deadline}" --workers-ok "${prefill_fa3_pair_ok}"
fi

# P is revalidated; optional P3 follows the full P gate on the same GPUs.
if [[ "${prefill_pp}" == 4 && "$(jq -r '.prefill.fa3_prepack_gate // false' "${ROOT}/source/firmus-framework.json")" == true ]]; then
  mkdir -p "${ROOT}/results/prefill-fa3-prepack"
  prefill_fa3_prepack_started=$(date +%s)
  prefill_fa3_prepack_deadline=$((prefill_fa3_prepack_started + 155))
  prefill_fa3_prepack_selections=()
  prefill_fa3_prepack_ok=0
  if [[ -s "${ROOT}/results/prefill-fa3-pair/selection.json" ]] && jq -e '.adopted == true' "${ROOT}/results/prefill-fa3-pair/selection.json" >/dev/null; then
    prefill_fa3_prepack_ok=1
    # Estimated group wall <=100 s: 150 s timeout, 2 s kill grace, aggregate by start+155.
    # Round 14: rank gate about 36 s, job 742 s, READY 449 s. See round-17 note for replacement budget.
    for worker in 0 1 2; do
      node=${HEAD}
      if (( worker == 2 )); then node=${TAIL}; fi
      prefill_fa3_prepack_remaining=$((prefill_fa3_prepack_started + 150 - $(date +%s)))
      if (( prefill_fa3_prepack_remaining <= 0 )); then prefill_fa3_prepack_ok=0; break; fi
      timeout --kill-after=2 "${prefill_fa3_prepack_remaining}" srun -N1 -n1 --exact -w "${node}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts prefill-fa3-prepack-${worker})" /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${ROOT}/scripts/select-firmus-prefill-fa3-prepack.py" --output "${ROOT}/results/prefill-fa3-prepack/prefill-${worker}" --run-id "${SLURM_JOB_ID}-${prefill_fa3_prepack_started}" --deadline "$((prefill_fa3_prepack_started + 150))" --p3 "$(jq -r '.prefill.fa3_v3_gate // false' "${ROOT}/source/firmus-framework.json")" --image-sha256 "$(jq -r '.images.sglang.sha256' "${ROOT}/source/firmus-framework.json")" >"${ART}/prefill-fa3-prepack-${worker}.log" 2>&1 &
      prefill_fa3_prepack_selections+=("$!")
    done
    while (( ${#prefill_fa3_prepack_selections[@]} )); do
      prefill_fa3_prepack_finished=
      if wait -n -p prefill_fa3_prepack_finished "${prefill_fa3_prepack_selections[@]}"; then
        prefill_fa3_prepack_pending=()
        for pid in "${prefill_fa3_prepack_selections[@]}"; do if [[ "${pid}" != "${prefill_fa3_prepack_finished}" ]]; then prefill_fa3_prepack_pending+=("${pid}"); fi; done
        prefill_fa3_prepack_selections=("${prefill_fa3_prepack_pending[@]}")
      else
        prefill_fa3_prepack_ok=0
        for pid in "${prefill_fa3_prepack_selections[@]}"; do kill "${pid}" 2>/dev/null || true; done
        for pid in "${prefill_fa3_prepack_selections[@]}"; do wait "${pid}" || true; done
        break
      fi
    done
  fi
  # Bound aggregation too; an incomplete selection leaves the pair env untouched.
  prefill_fa3_prepack_remaining=$((prefill_fa3_prepack_deadline - $(date +%s)))
  if (( prefill_fa3_prepack_remaining > 0 )); then
    timeout --signal=KILL "${prefill_fa3_prepack_remaining}" uv run --no-project --python 3.12 python "${ROOT}/scripts/select-firmus-prefill-fa3-prepack.py" --aggregate "${ROOT}/results/prefill-fa3-prepack" --output "${ROOT}/results/prefill-fa3-prepack/selection.json" --env-output "${ROOT}/source/prefill-kernels.env" --pair-selection "${ROOT}/results/prefill-fa3-pair/selection.json" --run-id "${SLURM_JOB_ID}-${prefill_fa3_prepack_started}" --deadline "${prefill_fa3_prepack_deadline}" --workers-ok "${prefill_fa3_prepack_ok}" --p3 "$(jq -r '.prefill.fa3_v3_gate // false' "${ROOT}/source/firmus-framework.json")" --image-sha256 "$(jq -r '.images.sglang.sha256' "${ROOT}/source/firmus-framework.json")" || true
  fi
fi


# Decode greedy gate begin: entirely after pair/prepack, before any model starts.
# Estimated <=60 s, workers 85 + kill 2 + aggregate 3 <=90 s (>=1.5x).
# After the gate, keep 660 s for model start, warmup, measurement and collection (634 s measured after
# the prepack group in round 21, 41327492259, which already includes a fresh-image start).
if [[ "${prefill_pp}" == 4 && "$(jq -r '.decode_greedy_verify_gate // false' "${ROOT}/source/firmus-framework.json")" == true ]]; then
  mkdir -p "${ROOT}/results/decode-greedy"
  if (( 1020 - SECONDS >= 90 + 660 )); then
    decode_greedy_started=$(date +%s)
    decode_greedy_ok=0
    # Reserve g4's eight devices only during this step; execute solely on physical 4-7.
    # The launcher resolves physical indices to UUIDs before torchrun, independently of CUDA order.
    if timeout --kill-after=2 85 srun -N1 -n1 --exact -w "${TAIL}" --gres=gpu:8 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts decode-greedy)" /opt/sglang/venv/bin/python "${ROOT}/scripts/select-firmus-decode-greedy.py" --model-path "${MODEL}" --launch --output "${ROOT}/results/decode-greedy" --run-id "${SLURM_JOB_ID}-${decode_greedy_started}" --deadline "$((decode_greedy_started + 85))" --image-sha256 "$(jq -r '.images.sglang.sha256' "${ROOT}/source/firmus-framework.json")" >"${ART}/decode-greedy.log" 2>&1; then decode_greedy_ok=1; fi
    decode_greedy_remaining=$((decode_greedy_started + 90 - $(date +%s)))
    if (( decode_greedy_remaining > 0 )); then
      cp "${ROOT}/source/decode-kernels.env" "${ROOT}/source/decode-kernels.greedy.env"
      if timeout --signal=KILL "${decode_greedy_remaining}" uv run --no-project --python 3.12 python "${ROOT}/scripts/select-firmus-decode-greedy.py" --model-path "${MODEL}" --aggregate "${ROOT}/results/decode-greedy" --output "${ROOT}/results/decode-greedy/selection.json" --env-output "${ROOT}/source/decode-kernels.greedy.env" --run-id "${SLURM_JOB_ID}-${decode_greedy_started}" --deadline "$((decode_greedy_started + 90))" --workers-ok "${decode_greedy_ok}" --image-sha256 "$(jq -r '.images.sglang.sha256' "${ROOT}/source/firmus-framework.json")" && jq -e '.adopted == true' "${ROOT}/results/decode-greedy/selection.json" >/dev/null; then
        mv "${ROOT}/source/decode-kernels.greedy.env" "${ROOT}/source/decode-kernels.env"
      fi
      rm -f "${ROOT}/source/decode-kernels.greedy.env" "${ROOT}/source/decode-kernels.greedy.greedy.tmp"
    fi
  else
    printf '%s\n' '{"adopted":false,"reason":"insufficient remaining serving budget"}' >"${ROOT}/results/decode-greedy/selection.json"
  fi
fi
# Decode greedy gate end.

# Round 32 candidate / round 33 pooled gate; keep 660 s for serving and collection.
if [[ "${prefill_pp}" == 4 && "$(jq -r '.prefill.megamoe_fc1_first_k_gate // false' "${ROOT}/source/firmus-framework.json")" == true ]]; then
  mkdir -p "${ROOT}/results/prefill-megamoe-fc1-first-k"
  if (( SECONDS + 105 + 660 <= 1020 )) && [[ -s "${ROOT}/results/prefill-fa3-prepack/selection.json" ]] && jq -e '.adopted == true' "${ROOT}/results/prefill-fa3-prepack/selection.json" >/dev/null; then
    prefill_fc1_first_k_started=$(date +%s)
    prefill_fc1_first_k_ok=1
    prefill_fc1_first_k_selections=()
    for worker in 0 1 2; do
      node=${HEAD}
      if (( worker == 2 )); then node=${TAIL}; fi
      prefill_fc1_first_k_remaining=$((prefill_fc1_first_k_started + 95 - $(date +%s)))
      if (( prefill_fc1_first_k_remaining <= 0 )); then prefill_fc1_first_k_ok=0; break; fi
      timeout --kill-after=2 "${prefill_fc1_first_k_remaining}" srun -N1 -n1 --exact -w "${node}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts prefill-megamoe-fc1-first-k-${worker})" env DG_JIT_CACHE_DIR="/tmp/sglang-r32-${SLURM_JOB_ID}-${worker}/deepgemm" TRITON_CACHE_DIR="/tmp/sglang-r32-${SLURM_JOB_ID}-${worker}/triton" TVM_FFI_CACHE_DIR="/tmp/sglang-r32-${SLURM_JOB_ID}-${worker}/tvm" /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${ROOT}/scripts/select-firmus-prefill-megamoe.py" --fc1-first-k --output "${ROOT}/results/prefill-megamoe-fc1-first-k/prefill-${worker}" --routing "${ROOT}/source/megamoe-local-ep1-routing.npz" --run-id "${SLURM_JOB_ID}-${prefill_fc1_first_k_started}" --deadline "$((prefill_fc1_first_k_started + 95))" --image-sha256 "$(jq -r '.images.sglang.sha256' "${ROOT}/source/firmus-framework.json")" >"${ART}/prefill-megamoe-fc1-first-k-${worker}.log" 2>&1 &
      prefill_fc1_first_k_selections+=("$!")
    done
    for pid in "${prefill_fc1_first_k_selections[@]}"; do wait "${pid}" || prefill_fc1_first_k_ok=0; done
    prefill_fc1_first_k_remaining=$((prefill_fc1_first_k_started + 105 - $(date +%s)))
    if (( prefill_fc1_first_k_remaining > 0 )); then
      cp "${ROOT}/source/prefill-kernels.env" "${ROOT}/source/prefill-kernels.r32.env"
      if timeout --signal=KILL "${prefill_fc1_first_k_remaining}" uv run --no-project --python 3.12 python "${ROOT}/scripts/select-firmus-prefill-megamoe.py" --fc1-first-k --aggregate "${ROOT}/results/prefill-megamoe-fc1-first-k" --output "${ROOT}/results/prefill-megamoe-fc1-first-k/selection.json" --env-output "${ROOT}/source/prefill-kernels.r32.env" --pair-selection "${ROOT}/results/prefill-fa3-pair/selection.json" --run-id "${SLURM_JOB_ID}-${prefill_fc1_first_k_started}" --deadline "$((prefill_fc1_first_k_started + 105))" --workers-ok "${prefill_fc1_first_k_ok}" --image-sha256 "$(jq -r '.images.sglang.sha256' "${ROOT}/source/firmus-framework.json")" && jq -e '.adopted == true' "${ROOT}/results/prefill-megamoe-fc1-first-k/selection.json" >/dev/null; then
        mv "${ROOT}/source/prefill-kernels.r32.env" "${ROOT}/source/prefill-kernels.env"
      fi
      rm -f "${ROOT}/source/prefill-kernels.r32.env" "${ROOT}/source/prefill-kernels.r32.env.fc1-first-k.tmp"
    fi
  else
    printf '{"adopted":false,"reason":"insufficient budget or incomplete P gate"}\n' >"${ROOT}/results/prefill-megamoe-fc1-first-k/selection.json"
  fi
fi


srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts prefill-0)" bash "${ROOT}/source/worker.sh" prefill-0 30100 "${MODEL}" "${prefill_tp}" "${HEAD_IP}" "${speculation[@]}" "${tokenizer_args[@]}" "${prefill_args[@]}" --disaggregation-bootstrap-port 9000 >"${ART}/prefill-0.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts prefill-1)" bash "${ROOT}/source/worker.sh" prefill-1 30101 "${MODEL}" "${prefill_tp}" "${HEAD_IP}" "${speculation[@]}" "${tokenizer_args[@]}" "${prefill_args[@]}" --disaggregation-bootstrap-port 9001 >"${ART}/prefill-1.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts prefill-2)" bash "${ROOT}/source/worker.sh" prefill-2 30102 "${MODEL}" "${prefill_tp}" "${TAIL_IP}" "${speculation[@]}" "${tokenizer_args[@]}" "${prefill_args[@]}" --disaggregation-bootstrap-port 9002 >"${ART}/prefill-2.log" 2>&1 &
worker_pids+=("$!")
if [[ -n "${dense_selection:-}" ]]; then
  wait "${dense_selection}"
  cat "${ROOT}/results/decode-selection/fa3.env" "${ROOT}/results/decode-selection/dense.env" >"${ROOT}/source/decode-kernels.env"
fi
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts decode-0)" bash "${ROOT}/source/worker.sh" decode-0 30200 "${MODEL}" 4 "${TAIL_IP}" "${speculation[@]}" "${tokenizer_args[@]}" "${decode_graph[@]}" --ep-size 4 --dp-size "$(jq -er .decode_dp_size "${ROOT}/source/firmus-framework.json")" --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1 --moe-a2a-backend megamoe --moe-runner-backend deep_gemm --chunked-prefill-size 8192 --max-running-requests "$(jq -er .decode_max_running_requests "${ROOT}/source/firmus-framework.json")" --disaggregation-decode-extra-slots "$(jq -er .decode_extra_slots "${ROOT}/source/firmus-framework.json")" --disaggregation-decode-polling-interval "$(jq -er .decode_polling_interval "${ROOT}/source/firmus-framework.json")" --max-total-tokens "$(jq -er .decode_max_total_tokens "${ROOT}/source/firmus-framework.json")" >"${ART}/decode-0.log" 2>&1 &
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
# Every archived run shows the Decode output stream stopping for ~0.33 s every 14-19 s while the Decode
# scheduler keeps stepping, which starves the Prefill queue (2026-10-30-pp4-system-r3.md). Freeze the startup
# heap of every server process after the warm-up request (SGLang /freeze_gc: tokenizer, schedulers,
# detokenizer) so that Python's full collections scan only request state. A failed call keeps GC as it was.
while IFS=$'\t' read -r worker host port _; do curl --fail --silent --show-error --max-time 30 -X POST "http://${host}:${port}/freeze_gc" >>"${ART}/freeze-gc.log" 2>&1 || printf 'freeze_gc failed on %s\n' "${worker}" >>"${ART}/freeze-gc.log"; done <"${ROOT}/source/worker-endpoints.tsv"
benchmark_status=0
if [[ "${profile_only}" != true ]]; then
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${ART}/aiperf.log" 2>&1 || benchmark_status=$?
test -s "${ROOT}/results/score.json"
for worker in 0 1 2; do
  grep -a -h -o -E 'Prefill r32 fc1-first-k (prewarm done|using candidate|using control).*' "${ART}/prefill-${worker}.log" >"${ROOT}/results/prefill-r32-dispatch-${worker}.txt" || printf 'no Prefill r32 fc1-first-k dispatch line\n' >"${ROOT}/results/prefill-r32-dispatch-${worker}.txt"
done
# Round 22 adopted a greedy fusion that serving never dispatched; keep each Decode rank's once-per-process decision with the score.
grep -a -h -o -E 'Decode greedy verify (using fused kernels|fallback).*' "${ART}/decode-0.log" | sort -u >"${ROOT}/results/decode-greedy-dispatch.txt" || printf 'no Decode greedy verify dispatch line in decode-0.log\n' >"${ROOT}/results/decode-greedy-dispatch.txt"
if (( benchmark_status == 0 )); then
  while IFS=$'\t' read -r worker host port _; do curl --fail --silent --show-error --max-time 15 "http://${host}:${port}/get_server_info" >"${ROOT}/results/server-info-${worker}.json"; done <"${ROOT}/source/worker-endpoints.tsv"
  uv run --no-project --python 3.12 python "${ROOT}/scripts/collect-eagle-acceptance.py" --cluster firmus --config "${ROOT}/source/firmus-framework.json" --endpoints "${ROOT}/source/worker-endpoints.tsv" --server-info-dir "${ROOT}/results" --log-dir "${ROOT}/logs" --output "${ROOT}/results/eagle-acceptance.json"
fi
fi
if [[ "${PROFILE}" == true ]]; then
  # A completed but SLA-ineligible ordinary run still needs the requested trace.
  # The original benchmark status is retained and returned after collection.
  if (( routing_capture_samples > 0 )); then
    for worker in prefill-0 prefill-1 prefill-2; do touch "${ROOT}/results/prefill-routing/${worker}/active"; done
  fi
  "${ROOT}/scripts/capture-firmus-profile.sh" "${ROOT}" >"${ART}/profile-capture.log" 2>&1 &
  profile_capture_pid=$!
  # A failed profile client must not cancel capture or report flushing; keep both statuses.
  profile_client_status=0
  srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" profile >"${ART}/profile-aiperf.log" 2>&1 || profile_client_status=$?
  profile_capture_status=0
  wait "${profile_capture_pid}" || profile_capture_status=$?
  profile_capture_pid=
  if (( routing_capture_samples > 0 )); then
    for worker in prefill-0 prefill-1 prefill-2; do rm -f "${ROOT}/results/prefill-routing/${worker}/active"; done
    srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-image="${SGLANG}" --container-mounts="$(mounts routing-summary)" /opt/sglang/venv/bin/python "${ROOT}/scripts/summarize-prefill-routing.py" --directory "${ROOT}/results/prefill-routing" --placement "${ROOT}/source/prefill-expert-location.json" --output "${ROOT}/results/prefill-routing-summary.json" --expected-workers 3 --expected-ranks 4 --samples-per-rank "${routing_capture_samples}" >"${ART}/prefill-routing-summary.log" 2>&1
  fi
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
# Round 32 submit: FC1 first-K128 specialization with its twelve-GPU gate.
