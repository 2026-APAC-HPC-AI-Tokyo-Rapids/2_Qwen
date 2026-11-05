set -euo pipefail
name=$1
port=$2
model=$3
tp=$4
export VLLM_NIXL_SIDE_CHANNEL_HOST=$5
shift 5
role=${name%%-*}
export VLLM_NIXL_SIDE_CHANNEL_PORT="$((port - 24500))"
export VLLM_USE_NCCL_SYMM_MEM=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
# Serve the published FP32 block scales: with E8M0 on, the pinned vLLM re-quantizes DeepGEMM dense weights to power-of-two scales on Hopper.
export VLLM_USE_DEEP_GEMM_E8M0=0
role_flags=()
blocks=2048
if [[ "$role" == prefill ]]; then
  # Match the verified SGLang prefill token buckets, using native breakable PIECEWISE graphs.
  role_flags=(--compilation-config "$(cat "$(dirname "$0")/prefill-cudagraph.json")")
  export VLLM_USE_BREAKABLE_CUDAGRAPH=1
  # Keep dense FP8 outside the FlashInfer JIT path during prefill capture.
  export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
  export VLLM_ALLREDUCE_USE_SYMM_MEM=1
  export VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0
  KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
else
  blocks=8192
  KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
fi
UCX_IB=$(for d in /sys/class/infiniband/*; do
  [ "$(cat "$d/ports/1/state" 2>/dev/null)" = "4: ACTIVE" ] || continue
  echo "$(basename "$d"):1"
done | paste -sd,)
[ -n "$UCX_IB" ] || { echo "FATAL: no active HCA on $(hostname)"; exit 1; }
export UCX_NET_DEVICES=$UCX_IB
export UCX_POSIX_USE_PROC_LINK=n
export TRITON_CACHE_DIR=/tmp/vllm-$name/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/vllm-$name/inductor
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SERVER_DEV_MODE=1  # Required by the pinned image's /server_info registration gate.
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
echo "$name host=$(hostname) ip=$VLLM_NIXL_SIDE_CHANNEL_HOST hca=$UCX_IB gpu=$CUDA_VISIBLE_DEVICES tp=$tp symm=$VLLM_ALLREDUCE_USE_SYMM_MEM"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit --format=csv
worker_command=(/usr/local/bin/vllm serve "$model" --served-model-name qwen --trust-remote-code --dtype bfloat16 --max-model-len 8192 --tensor-parallel-size "$tp" --gpu-memory-utilization 0.90 --block-size 64 --num-gpu-blocks-override "$blocks" --max-num-batched-tokens 8192 --enable-chunked-prefill --kv-cache-dtype fp8_e4m3 --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":3}' --kv-transfer-config "$KV_CONFIG" "${role_flags[@]}" "$@" --host 0.0.0.0 --port "$port")
printf '%q ' "${worker_command[@]}" >"$(dirname "$0")/argv-${name}.txt"
printf '\n' >>"$(dirname "$0")/argv-${name}.txt"
if [[ -x /opt/nsys/bin/nsys ]]; then
  export QWEN_PROFILER_NSYS_BIN=/opt/nsys/bin/nsys
  export QWEN_PROFILER_INVOCATION_PATH="$(dirname "$0")/../profile/nsys-${name}.argv"
  worker_command+=(--profiler-config '{"profiler":"cuda"}')
  exec bash "$(dirname "$0")/../scripts/exec-profiler.sh" "${worker_command[@]}"
fi
exec "${worker_command[@]}"
