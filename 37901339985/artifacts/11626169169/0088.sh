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
export VLLM_NIXL_SIDE_CHANNEL_HOST="$(getent ahostsv4 "$(hostname -s)" | awk '$2 == "STREAM" {print $1; exit}')"
export VLLM_NIXL_SIDE_CHANNEL_PORT="$((port - 24500))"
if [[ "$role" == prefill ]]; then
  KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
else
  KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
fi
runtime_dir=$(dirname "$(dirname "$0")")
worker_command=(/usr/local/bin/vllm serve "$model" --served-model-name qwen --trust-remote-code --host 0.0.0.0 --port "$port" --tensor-parallel-size "$tp" --max-model-len 8192 --gpu-memory-utilization 0.90 --kv-transfer-config "$KV_CONFIG" "$@")
if [[ -x /opt/nsys/bin/nsys ]]; then
  export QWEN_PROFILER_NSYS_BIN=/opt/nsys/bin/nsys
  export QWEN_PROFILER_INVOCATION_PATH="${runtime_dir}/profile/nsys-${name}.argv"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  worker_command=(bash "${runtime_dir}/scripts/exec-profiler.sh" "${worker_command[@]}" --profiler-config '{"profiler":"cuda"}')
fi
exec "${worker_command[@]}"
