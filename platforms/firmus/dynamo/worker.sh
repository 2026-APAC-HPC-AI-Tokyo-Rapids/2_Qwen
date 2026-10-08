#!/usr/bin/env bash
set -euo pipefail
framework=$1
name=$2
port=$3
model=$4
tp=$5
role=${name%%-*}
runtime_dir=$(dirname "$(dirname "$0")")
node_ip=$(getent ahostsv4 "$(hostname -s)" | awk '$2 == "STREAM" {print $1; exit}')
export DYN_TCP_RPC_HOST=${node_ip}
export DYN_TCP_RESPONSE_STREAM_HOST=${node_ip}
export DYN_SYSTEM_HOST=0.0.0.0
export DYN_SYSTEM_PORT=${port}
export SGLANG_HOST_IP=${node_ip}
export VLLM_NIXL_SIDE_CHANNEL_HOST=${node_ip}
export VLLM_NIXL_SIDE_CHANNEL_PORT=$((port - 24500))
export UCX_NET_DEVICES=$(for device in /sys/class/infiniband/*; do if [[ "$(cat "${device}/ports/1/link_layer" 2>/dev/null)" == InfiniBand && "$(cat "${device}/ports/1/state" 2>/dev/null)" == '4: ACTIVE' ]]; then printf '%s:1\n' "${device##*/}"; fi; done | paste -sd,)
[[ -n "${UCX_NET_DEVICES}" ]]
export UCX_POSIX_USE_PROC_LINK=n
export TRITON_CACHE_DIR=${runtime_dir}/home/${name}/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=${runtime_dir}/home/${name}/.cache/inductor
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
printf 'worker=%s host=%s gpu=%s tp=%s ib=%s\n' "${name}" "${node_ip}" "${CUDA_VISIBLE_DEVICES}" "${tp}" "${UCX_NET_DEVICES}"
if [[ "${framework}" == vllm ]]; then
  worker_command=(python3 -m dynamo.vllm --model "${model}" --served-model-name qwen --trust-remote-code --tensor-parallel-size "${tp}" --max-model-len 8192 --gpu-memory-utilization 0.90 --disaggregation-mode "${role}" --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}')
  if [[ "${role}" == decode ]]; then worker_command+=(--enable-expert-parallel --max-cudagraph-capture-size 64 --max-num-seqs 256 --no-enable-flashinfer-autotune); fi
elif [[ "${framework}" == sglang ]]; then
  worker_command=(python3 -m dynamo.sglang --model-path "${model}" --served-model-name qwen --trust-remote-code --context-length 8192 --tp-size "${tp}" --mem-fraction-static 0.90 --disaggregation-mode "${role}" --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port "$((port - 21100))" --host 0.0.0.0 --port "$((port + 1000))" --nccl-port "$((port + 2000))" --enable-metrics)
  if [[ "${role}" == decode ]]; then worker_command+=(--ep-size 4 --max-running-requests 256 --cuda-graph-max-bs-decode 64 --disable-flashinfer-autotune); fi
else
  exit 64
fi
printf '%q ' "${worker_command[@]}" >"${runtime_dir}/source/${name}.argv"
printf '\n' >>"${runtime_dir}/source/${name}.argv"
exec "${worker_command[@]}"
