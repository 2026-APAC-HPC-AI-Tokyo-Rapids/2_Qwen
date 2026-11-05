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
# Both endpoints now own all KV heads (attention TP1). Use direct NIXL
# page transfer; heterogeneous-TP staging would only allocate unused pools.
export SGLANG_DISAGG_STAGING_BUFFER=0
export SGLANG_DG_CACHE_DIR="${SLURM_SUBMIT_DIR}/home/deep-gemm"
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096 SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1
export TRITON_CACHE_DIR=/tmp/sglang-$name/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-$name/inductor
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
echo "$name host=$(hostname) ip=$SGLANG_HOST_IP hca=$UCX_IB gpu=$CUDA_VISIBLE_DEVICES tp=$tp shuffle=${SGLANG_PREFILL_FUSED_FP8_SHUFFLE:-0} routing=${SGLANG_PREFILL_TILED_ROUTING:-0}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit --format=csv
# Queue-aware DP routing replaces bootstrap-room hashing on P and round-robin
# on D. Publish every iteration so completed requests release routing budget.
# Native P registration and D batched lookup preserve the actual KV source rank.
command=(/opt/sglang/venv/bin/python -m sglang.launch_server --model-path "$model" --served-model-name qwen --trust-remote-code --context-length 8192 --tp-size "$tp" --mem-fraction-static 0.90 --page-size 64 --kv-cache-dtype fp8_e4m3 --attention-backend fa3 --nccl-port "$((port + 1000))" --disaggregation-mode "$role" --disaggregation-transfer-backend nixl --load-balance-method total_requests --load-snapshot-publish-interval 1 "$@" --host 0.0.0.0 --port "$port")
# DP attention derives six TCP endpoints from this address.
# Adjacent HTTP ports need separate endpoint ranges on the same node.
if [[ "$role" == prefill ]]; then
  command+=(--dist-init-addr "127.0.0.1:$((32000 + ${name##*-} * 100))")
else
  command+=(--dist-init-addr "127.0.0.1:33000")
fi
if [[ -x /opt/nsys/bin/nsys ]]; then
  runtime_dir=$(dirname "$(dirname "$0")")
  export QWEN_PROFILER_NSYS_BIN=/opt/nsys/bin/nsys
  export QWEN_PROFILER_INVOCATION_PATH="${runtime_dir}/profile/nsys-${name}.argv"
  command=(bash "${runtime_dir}/scripts/exec-profiler.sh" "${command[@]}")
fi
exec "${command[@]}"
