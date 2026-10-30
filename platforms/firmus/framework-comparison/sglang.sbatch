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
  if [[ "${profile_only}" == true ]]; then printf 'Profile-only recovery: ordinary benchmark is omitted. No new official score. Client uses 20/30/90 seconds; capture is instrumented.\n' >"${ROOT}/profile/interpretation.txt"; fi
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
  runtime_dir=$(dirname "$(dirname "$0")")
  export QWEN_PROFILER_NSYS_BIN=/opt/nsys/bin/nsys
  export QWEN_PROFILER_INVOCATION_PATH="${runtime_dir}/profile/nsys-${name}.argv"
  command=(bash "${runtime_dir}/scripts/exec-profiler.sh" "${command[@]}")
fi
exec "${command[@]}"
EOF

# One low-frequency NVML query process per node, outside the serving workers.
# Explicit GPU UUIDs identify all eight GPUs on each node without rank duplication.
for host in "${HEAD}" "${TAIL}"; do
  srun --overlap --exact -N1 -n1 -w "${host}" --gres=gpu:8 --cpus-per-task=1 nvidia-smi --query-gpu=timestamp,index,uuid,power.draw,clocks.sm,clocks.mem,clocks_throttle_reasons.active,temperature.gpu --format=csv --loop=1 >"${ART}/gpu-telemetry-${host}.csv" 2>"${ART}/gpu-telemetry-${host}.stderr.log" &
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
  test -s "${ROOT}/source/prefill-kernels.env" && test -s "${ROOT}/source/decode-kernels.env"
  printf 'PP4 serving trial: retain prior kernel settings and the pinned PP4 selections; gate the EP1 MegaMoE variants and its combine norm, and the two-heads-per-CTA Decode FA3 rows, the fused Decode MoE input and the Decode QKV BLOCK_M 192 tile against the pinned kernels.\n'
  # prepare-firmus-pp4-settings.py pins the Prefill FA3 row and the Decode norm/quant kernel (same winner in
  # 40849459077 and 40883389891), the FP32 Prefill Q/K norm writer (same winner in 40911121619 and 40925558339)
  # and the one-pass Prefill MoE input (13.6% faster on every GPU in 40964714831); their selections no longer
  # run here. The pipelined-tile Prefill FA3 row (-3% on H100) is no longer screened after 40991744289 stopped
  # serving with it and the split MegaMoE both adopted (0.2-0.35 ms per period, one fewer suspect).
  # The four Decode GPUs are idle before model load. One after the other, while the Prefill GPUs run the
  # MegaMoE selection and its endurance gate below: (1) the two-heads-per-CTA (256 B row read) FA3 verify rows
  # against the pinned FA3 candidate (~20-40 s, 175 s timeout): adopted only if bit-exact where the split
  # layout is the pinned one and >=1 us/layer faster on every GPU at cap/4-4, cap/4-2 and cap/4 (44, 46 and
  # 48 requests per rank, the Decode operating range at cap 48); (2) the PP4 copy-engine proxy transport
  # (check-firmus-pp4-copy-engine.py, ~20 s, 60 s timeout): the patch's IPC buffer/event calls on four GPUs,
  # adopted only if every transfer was exact; (3) the Decode TARGET_VERIFY MegaMoE input
  # (select-firmus-decode-moe-input.py, ~30-45 s with the Triton and pre-dispatch JIT, 150 s timeout): the
  # post-attention norm writes the FP8 input and the router the top-k rows (no pre-dispatch kernel), adopted
  # only if the residual, FP8 input, scales and top-k equal what pre-dispatch writes from the same y and it
  # saves max(1 us, 2%) per layer at 44, 46 and 48 requests per rank in both blocks on every GPU.
  # (4) the Decode QKV DeepGEMM BLOCK_M 192 tile for 129..192 rows (select-firmus-decode-dense.py --pinned-env,
  # ~40-60 s with two DeepGEMM JIT compiles, 420 s timeout): adopted only if bit-identical to the pinned QKV
  # config and >=0.5 us faster (pooled, both blocks) at each of 176, 184 and 192 rows; only the changed
  # DG_SM90_1D2D_DECODE_CONFIG line is appended to decode-kernels.env.
  # All results are taken before the workers start; a failed, missing or late result keeps the pinned
  # decode-kernels.env (pre-dispatch Decode MoE input) and in-stream PyNccl.
  {
    srun -N1 -n1 --exact -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts decode-selection-fa3)" bash "${ROOT}/scripts/select-firmus-decode-kernels.sh" "${ROOT}" fa3 --pinned-env "${ROOT}/source/decode-kernels.env" >"${ART}/decode-selection-fa3.log" 2>&1 || true
    srun -N1 -n1 --exact -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts pp4-copy-engine)" timeout --kill-after=5 60 /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${ROOT}/scripts/check-firmus-pp4-copy-engine.py" --output "${ROOT}/results/pp4-copy-engine" >"${ART}/pp4-copy-engine.log" 2>&1 || true
    srun -N1 -n1 --exact -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts decode-selection-moe-input)" bash "${ROOT}/scripts/select-firmus-decode-kernels.sh" "${ROOT}" moe-input >"${ART}/decode-selection-moe-input.log" 2>&1 || true
    srun -N1 -n1 --exact -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts decode-selection-dense)" bash "${ROOT}/scripts/select-firmus-decode-kernels.sh" "${ROOT}" dense --pinned-env "${ROOT}/source/decode-kernels.env" >"${ART}/decode-selection-dense.log" 2>&1 || true
  } &
  decode_gpu_selections=$!
  # EP1 MegaMoE on all twelve Prefill GPUs before model load (select-firmus-prefill-megamoe.py, about 1.5-2 min):
  # the control must match an FP32 reference (a failure stops the job); the four-stage, lean, lean + paired-tail and
  # lean + split-accumulator (with SS or ldmatrix A) variants are adopted only if they are bit-exact on the two timed routings and the untimed edge routing and
  # save the summed margin over all GPUs and timed routings (the fastest one wins).
  # A missing or late result keeps three stages. The selection, the endurance gate and the combine-norm gate below
  # (~2.6-3.4 min together) end within the Decode-GPU selections above (~4-4.5 min), so they add no job time.
  prefill_gpu_selections=()
  for worker in 0 1 2; do
    node=${HEAD}
    if (( worker == 2 )); then node=${TAIL}; fi
    srun -N1 -n1 --exact -w "${node}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts prefill-megamoe-${worker})" env TRITON_CACHE_DIR="/tmp/qwen-prefill-megamoe-${SLURM_JOB_ID}-${worker}/triton" DG_JIT_CACHE_DIR="/tmp/qwen-prefill-megamoe-${SLURM_JOB_ID}-${worker}/deep-gemm" timeout --kill-after=5 240 /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${ROOT}/scripts/select-firmus-prefill-megamoe.py" --ranks 1 --output "${ROOT}/results/prefill-megamoe/prefill-${worker}" >"${ART}/prefill-megamoe-${worker}.log" 2>&1 &
    prefill_gpu_selections+=("$!")
  done
  for pid in "${prefill_gpu_selections[@]}"; do wait "${pid}" || true; done
  # MegaMoE endurance gate (select-firmus-prefill-megamoe.py --endurance, ~20-40 s, 90 s timeout per worker, the
  # three workers in parallel): 40991744289 adopted the split accumulator after ~30 selection calls per GPU and all
  # three workers stopped after ~300 serving calls per GPU. Each GPU recomputes the winner from all twelve selection
  # results; if it has not served before (not control, four-stage, four-stage-lean or, since 41005872470, four-stage-lean-split), it runs it for 600 back-to-back
  # calls over the three routings and compares each routing's last output with the control bit for bit. The aggregate
  # keeps such a winner only if all twelve GPUs finished and matched; a hung, failed or late gate takes the fastest
  # eligible served variant instead and records why. The JIT cache of the selection step is reused.
  prefill_gpu_gates=()
  for worker in 0 1 2; do
    node=${HEAD}
    if (( worker == 2 )); then node=${TAIL}; fi
    srun -N1 -n1 --exact -w "${node}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts prefill-megamoe-endurance-${worker})" env TRITON_CACHE_DIR="/tmp/qwen-prefill-megamoe-${SLURM_JOB_ID}-${worker}/triton" DG_JIT_CACHE_DIR="/tmp/qwen-prefill-megamoe-${SLURM_JOB_ID}-${worker}/deep-gemm" timeout --kill-after=5 90 /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${ROOT}/scripts/select-firmus-prefill-megamoe.py" --endurance "${ROOT}/results/prefill-megamoe" --output "${ROOT}/results/prefill-megamoe-endurance/prefill-${worker}" >"${ART}/prefill-megamoe-endurance-${worker}.log" 2>&1 &
    prefill_gpu_gates+=("$!")
  done
  for pid in "${prefill_gpu_gates[@]}"; do wait "${pid}" || true; done
  uv run --no-project --python 3.12 python "${ROOT}/scripts/select-firmus-prefill-megamoe.py" --aggregate "${ROOT}/results/prefill-megamoe" --endurance "${ROOT}/results/prefill-megamoe-endurance" --output "${ROOT}/results/prefill-megamoe/selection.json" --env-output "${ROOT}/source/prefill-kernels.env"
  # Combine norm (select-firmus-prefill-megamoe.py --combine-norm, ~30-45 s with one DeepGEMM and one Triton JIT, 90 s
  # timeout per worker, the three workers in parallel): with the adopted variant on the lean path, each GPU
  # runs MegaMoE + the next layer's input add_rmsnorm_group_quant_fp8 (control) against fp8_mega_moe_norm, whose
  # combine updates the residual and writes the QKV input itself. The residual must be bit-equal and (q, scale)
  # within the sum-of-squares order bounds on the balanced, hot and edge routings; both are timed, then the fused
  # call runs 600 times back to back and must reproduce its first outputs bit for bit. SGLANG_PREFILL_COMBINE_NORM=1
  # is appended only if all twelve GPUs passed and the summed time saved the margin; otherwise every layer keeps
  # the separate norm. The JIT caches of the selection step are reused.
  prefill_gpu_norms=()
  for worker in 0 1 2; do
    node=${HEAD}
    if (( worker == 2 )); then node=${TAIL}; fi
    srun -N1 -n1 --exact -w "${node}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${SGLANG}" --container-mounts="$(mounts prefill-combine-norm-${worker})" env TRITON_CACHE_DIR="/tmp/qwen-prefill-megamoe-${SLURM_JOB_ID}-${worker}/triton" DG_JIT_CACHE_DIR="/tmp/qwen-prefill-megamoe-${SLURM_JOB_ID}-${worker}/deep-gemm" timeout --kill-after=5 90 /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${ROOT}/scripts/select-firmus-prefill-megamoe.py" --combine-norm "${ROOT}/results/prefill-megamoe/selection.json" --output "${ROOT}/results/prefill-combine-norm/prefill-${worker}" >"${ART}/prefill-combine-norm-${worker}.log" 2>&1 &
    prefill_gpu_norms+=("$!")
  done
  for pid in "${prefill_gpu_norms[@]}"; do wait "${pid}" || true; done
  uv run --no-project --python 3.12 python "${ROOT}/scripts/select-firmus-prefill-megamoe.py" --aggregate-combine-norm "${ROOT}/results/prefill-combine-norm" --combine-norm "${ROOT}/results/prefill-megamoe/selection.json" --output "${ROOT}/results/prefill-combine-norm/selection.json" --env-output "${ROOT}/source/prefill-kernels.env"
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

# PP4: the Decode-GPU selections started above (the copy-engine winner is appended to prefill-kernels.env
# after the MegaMoE and FA3 lines; Decode FA3, MoE-input and dense winners are appended to the pinned
# decode-kernels.env, overriding the pinned values when the worker sources the file).
if [[ -n "${decode_gpu_selections:-}" ]]; then
  wait "${decode_gpu_selections}" || true
  uv run --no-project --python 3.12 python "${ROOT}/scripts/check-firmus-pp4-copy-engine.py" --aggregate "${ROOT}/results/pp4-copy-engine" --output "${ROOT}/results/pp4-copy-engine/selection.json" --env-output "${ROOT}/source/prefill-kernels.env"
  if [[ -s "${ROOT}/results/decode-selection/fa3.env" ]]; then cat "${ROOT}/results/decode-selection/fa3.env" >>"${ROOT}/source/decode-kernels.env"; fi
  if [[ -s "${ROOT}/results/decode-selection/moe-input.env" ]]; then cat "${ROOT}/results/decode-selection/moe-input.env" >>"${ROOT}/source/decode-kernels.env"; fi
  if [[ -s "${ROOT}/results/decode-selection/dense.env" ]]; then cat "${ROOT}/results/decode-selection/dense.env" >>"${ROOT}/source/decode-kernels.env"; fi
fi

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts prefill-0)" bash "${ROOT}/source/worker.sh" prefill-0 30100 "${MODEL}" "${prefill_tp}" "${HEAD_IP}" "${speculation[@]}" "${prefill_args[@]}" --disaggregation-bootstrap-port 9000 >"${ART}/prefill-0.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts prefill-1)" bash "${ROOT}/source/worker.sh" prefill-1 30101 "${MODEL}" "${prefill_tp}" "${HEAD_IP}" "${speculation[@]}" "${prefill_args[@]}" --disaggregation-bootstrap-port 9001 >"${ART}/prefill-1.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts prefill-2)" bash "${ROOT}/source/worker.sh" prefill-2 30102 "${MODEL}" "${prefill_tp}" "${TAIL_IP}" "${speculation[@]}" "${prefill_args[@]}" --disaggregation-bootstrap-port 9002 >"${ART}/prefill-2.log" 2>&1 &
worker_pids+=("$!")
if [[ -n "${dense_selection:-}" ]]; then
  wait "${dense_selection}"
  cat "${ROOT}/results/decode-selection/fa3.env" "${ROOT}/results/decode-selection/dense.env" >"${ROOT}/source/decode-kernels.env"
fi
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
# Every archived run shows the Decode output stream stopping for ~0.33 s every 14-19 s while the Decode
# scheduler keeps stepping, which starves the Prefill queue (2026-10-30-pp4-system-r3.md). Freeze the startup
# heap of every server process after the warm-up request (SGLang /freeze_gc: tokenizer, schedulers,
# detokenizer) so that Python's full collections scan only request state. A failed call keeps GC as it was.
while IFS=$'\t' read -r worker host port _; do curl --fail --silent --show-error --max-time 30 -X POST "http://${host}:${port}/freeze_gc" >>"${ART}/freeze-gc.log" 2>&1 || printf 'freeze_gc failed on %s\n' "${worker}" >>"${ART}/freeze-gc.log"; done <"${ROOT}/source/worker-endpoints.tsv"
benchmark_status=0
if [[ "${profile_only}" != true ]]; then
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${ART}/aiperf.log" 2>&1 || benchmark_status=$?
test -s "${ROOT}/results/score.json"
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
