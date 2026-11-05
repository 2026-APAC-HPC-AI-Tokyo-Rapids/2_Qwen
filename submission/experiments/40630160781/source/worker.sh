set -euo pipefail
name=$1
port=$2
model=$3
tp=$4
export SGLANG_HOST_IP=$5
shift 5
role=${name%%-*}
runtime_dir=$(dirname "$(dirname "$0")")
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
