#!/bin/bash
#SBATCH -J vllm-firmus-native-ep4-c108
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:8
#SBATCH --propagate=NONE
#SBATCH --no-requeue
#SBATCH -o runs/sglang-%j.out
#SBATCH -e runs/sglang-%j.err
# H200: six TP2/EP1 prefills, one attention TP4/MoE TP1/EP4 decoder; unchanged FP8 weights.
# Native full-route EP4 versus E4/c108/E8M0-off. Wider expert GEMMs may improve decoder capacity; not compact routes.
# Diagnostic: native EP4 decoder only; ordinary reduction and the previous clean configuration retained.
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR}
VLLM=${ROOT}/vllm.sqsh
AIPERF=${ROOT}/sglang.sqsh
MODEL=/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8
R_PORT=8000
READY_SECS=600  # ASPIRE full stack took 7m09s; leave time for the 60/120/30 client.
N=$(jq -er .concurrency "${ROOT}/source/firmus-framework.json")
# Parse configuration on the host; the pinned serving image does not include jq.
speculation=(--speculative-config '{"method":"eagle3","model":"/draft","num_speculative_tokens":2}')
# Keep PIECEWISE coverage for mixed speculative batches; FULL alone requires uniform three-token requests.
decode_graph=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":768}' --cudagraph-capture-sizes)
while read -r size; do decode_graph+=("${size}"); done < <(jq -er '.decode_cuda_graph_bs[] | select(. <= 256) | . * 3' "${ROOT}/source/firmus-framework.json")
jq -c '{cudagraph_mode:"PIECEWISE",cudagraph_capture_sizes:.prefill_cuda_graph_tokens,max_cudagraph_capture_size:8192}' "${ROOT}/source/firmus-framework.json" >"${ROOT}/source/prefill-cudagraph.json"
ART=${ROOT}/logs
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
export MELLANOX_VISIBLE_DEVICES=all MELLANOX_MOUNT_DRIVER=yes
[[ "${PROFILE:-false}" == false || "${PROFILE}" == true ]]
jq -e '.framework == "vllm" and .workload == "official"' "${ROOT}/source/firmus-framework.json" >/dev/null

NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD=${NODES[0]}
TAIL=${NODES[1]}
ip_of() { getent ahostsv4 "$1" | awk '{print $1; exit}'; }
HEAD_IP=$(ip_of "${HEAD}")
TAIL_IP=$(ip_of "${TAIL}")

mounts() {
  mkdir -p "${ROOT}/home/$1"
  local profiler_mount=
  if [[ "${PROFILE:-false}" == true && ( "$1" == decode-0 || "$1" == profile-validation ) ]]; then profiler_mount=",${ROOT}/nsys:/opt/nsys:ro"; fi
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
echo "head=${HEAD}(${HEAD_IP}) tail=${TAIL}(${TAIL_IP}) model=${MODEL} isl=4000 osl=200 N=${N} topology=P6TP2EP1-D1AttentionTP4MoETP1EP4 kv=fp8_e4m3 speculation=EAGLE3 kernels=vllm-native"
test -r "${VLLM}" && test -r "${AIPERF}" && test -f "${MODEL}/config.json" && test -s "${ROOT}/draft/model.safetensors"
mkdir -p "${ART}"
worker_pids=()
cleanup_all() {
  status=$?
  trap - EXIT INT TERM
  for pid in "${worker_pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${worker_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  if [[ -n "${profile_capture_pid:-}" ]]; then kill "${profile_capture_pid}" 2>/dev/null || true; wait "${profile_capture_pid}" 2>/dev/null || true; fi
  exit "${status}"
}
trap cleanup_all EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for worker in 0 1 2 3; do printf 'prefill-%s\t%s\t%s\t2\n' "${worker}" "${HEAD_IP}" "$((30100 + worker))"; done >"${ROOT}/source/worker-endpoints.tsv"
for worker in 4 5; do printf 'prefill-%s\t%s\t%s\t2\n' "${worker}" "${TAIL_IP}" "$((30100 + worker))"; done >>"${ROOT}/source/worker-endpoints.tsv"
for worker in 0; do printf 'decode-%s\t%s\t%s\t4\n' "${worker}" "${TAIL_IP}" "$((30200 + worker))"; done >>"${ROOT}/source/worker-endpoints.tsv"

cat >"${ROOT}/source/worker.sh" <<'EOF'
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
EOF

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:2 --cpus-per-task=16 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${VLLM}" --container-mounts="$(mounts prefill-0)" bash "${ROOT}/source/worker.sh" prefill-0 30100 "${MODEL}" 2 "${HEAD_IP}" "${speculation[@]}" --disable-custom-all-reduce --max-num-seqs "${N}" --moe-backend deep_gemm >"${ART}/prefill-0.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:2 --cpus-per-task=16 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${VLLM}" --container-mounts="$(mounts prefill-1)" bash "${ROOT}/source/worker.sh" prefill-1 30101 "${MODEL}" 2 "${HEAD_IP}" "${speculation[@]}" --disable-custom-all-reduce --max-num-seqs "${N}" --moe-backend deep_gemm >"${ART}/prefill-1.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:2 --cpus-per-task=16 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${VLLM}" --container-mounts="$(mounts prefill-2)" bash "${ROOT}/source/worker.sh" prefill-2 30102 "${MODEL}" 2 "${HEAD_IP}" "${speculation[@]}" --disable-custom-all-reduce --max-num-seqs "${N}" --moe-backend deep_gemm >"${ART}/prefill-2.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:2 --cpus-per-task=16 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${VLLM}" --container-mounts="$(mounts prefill-3)" bash "${ROOT}/source/worker.sh" prefill-3 30103 "${MODEL}" 2 "${HEAD_IP}" "${speculation[@]}" --disable-custom-all-reduce --max-num-seqs "${N}" --moe-backend deep_gemm >"${ART}/prefill-3.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:2 --cpus-per-task=16 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${VLLM}" --container-mounts="$(mounts prefill-4)" bash "${ROOT}/source/worker.sh" prefill-4 30104 "${MODEL}" 2 "${TAIL_IP}" "${speculation[@]}" --disable-custom-all-reduce --max-num-seqs "${N}" --moe-backend deep_gemm >"${ART}/prefill-4.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:2 --cpus-per-task=16 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${VLLM}" --container-mounts="$(mounts prefill-5)" bash "${ROOT}/source/worker.sh" prefill-5 30105 "${MODEL}" 2 "${TAIL_IP}" "${speculation[@]}" --disable-custom-all-reduce --max-num-seqs "${N}" --moe-backend deep_gemm >"${ART}/prefill-5.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${VLLM}" --container-mounts="$(mounts decode-0)" bash "${ROOT}/source/worker.sh" decode-0 30200 "${MODEL}" 4 "${TAIL_IP}" "${speculation[@]}" "${decode_graph[@]}" --max-num-seqs 256 --no-enable-prefix-caching --enable-expert-parallel --moe-backend triton >"${ART}/decode-0.log" 2>&1 &
worker_pids+=("$!")

wait_http "http://${HEAD_IP}:30100/health"; echo "prefill-0 ready"
wait_http "http://${HEAD_IP}:30101/health"; echo "prefill-1 ready"
wait_http "http://${HEAD_IP}:30102/health"; echo "prefill-2 ready"
wait_http "http://${HEAD_IP}:30103/health"; echo "prefill-3 ready"
wait_http "http://${TAIL_IP}:30200/health"; echo "decode-0 ready"
wait_http "http://${TAIL_IP}:30104/health"; echo "prefill-4 ready"
wait_http "http://${TAIL_IP}:30105/health"; echo "prefill-5 ready"

srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${VLLM}" --container-mounts="$(mounts router)" /usr/bin/python3 "${ROOT}/vllm-proxy.py" --host 0.0.0.0 --port "${R_PORT}" --prefiller-hosts "${HEAD_IP}" "${HEAD_IP}" "${HEAD_IP}" "${HEAD_IP}" "${TAIL_IP}" "${TAIL_IP}" --prefiller-ports 30100 30101 30102 30103 30104 30105 --decoder-hosts "${TAIL_IP}" --decoder-ports 30200 >"${ART}/router.log" 2>&1 &
worker_pids+=("$!")

wait_http "http://${HEAD_IP}:${R_PORT}/v1/models"
echo "READY after ${SECONDS}s"

while IFS=$'\t' read -r worker host port _tp; do
  curl --fail --silent --show-error --max-time 30 "http://${host}:${port}/server_info?config_format=json" >"${ROOT}/results/server-info-${worker}.json"
  jq -e --arg role "${worker%%-*}" --argjson expected_tp "${_tp}" --slurpfile config "${ROOT}/source/firmus-framework.json" '.vllm_config as $v | $v.parallel_config.tensor_parallel_size == $expected_tp and $v.kernel_config.moe_backend == (if $role == "prefill" then "deep_gemm" else "triton" end) and $v.parallel_config.enable_expert_parallel == ($role == "decode") and $v.parallel_config.data_parallel_size == 1 and $v.cache_config.cache_dtype == "fp8_e4m3" and $v.cache_config.block_size == 64 and ($v.cache_config.block_size * $v.cache_config.num_gpu_blocks_override) == (if $role == "prefill" then 131072 else $config[0].decode_max_total_tokens end) and $v.model_config.max_model_len == 8192 and $v.scheduler_config.max_num_batched_tokens == 8192 and (if $role == "prefill" then $v.compilation_config.cudagraph_mode == 1 and $v.compilation_config.cudagraph_capture_sizes == $config[0].prefill_cuda_graph_tokens and $v.compilation_config.max_cudagraph_capture_size == 8192 else $v.cache_config.enable_prefix_caching == false and $v.compilation_config.max_cudagraph_capture_size == 768 and $v.compilation_config.cudagraph_mode == [2,1] and $v.compilation_config.cudagraph_capture_sizes == [$config[0].decode_cuda_graph_bs[] | select(. <= 256) | . * 3] end) and $v.speculative_config.draft_parallel_config.enable_expert_parallel == false and $v.speculative_config.draft_parallel_config.tensor_parallel_size == $expected_tp and $v.speculative_config.method == "eagle3" and $v.speculative_config.model == "/draft" and $v.speculative_config.num_speculative_tokens == 2 and $v.scheduler_config.max_num_seqs == (if $role == "prefill" then $config[0].concurrency else 256 end)' "${ROOT}/results/server-info-${worker}.json" >/dev/null
done <"${ROOT}/source/worker-endpoints.tsv"
# Profile-only execution is independent of official eligibility.
if [[ "${PROFILE:-false}" == true ]]; then
  mkdir -p "${ROOT}/profile/client"
  grep -F $'decode-0\t' "${ROOT}/source/worker-endpoints.tsv" >"${ROOT}/source/profile-worker-endpoints.tsv"
  printf 'Decoder-only native EP4 diagnostic. All seven workers serve the unchanged c108/spec2 workload, but only decode-0 TP4 is instrumented. Profile-only20/30/30 client pass; no official benchmark or throughput score. profile/client/ is nonofficial. CUDA event completion tracing is disabled because graph/IPC events are unsupported; kernel node tracing remains enabled. No complete critical-path or hardware-saturation claim is permitted.\n' >"${ROOT}/profile/interpretation.txt"
  bash "${ROOT}/scripts/capture-firmus-profile.sh" "${ROOT}" "${ROOT}/profile/client/aiperf.log" >"${ROOT}/logs/profile-capture.log" 2>&1 &
  profile_capture_pid=$!
  srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" profile >"${ROOT}/profile/client/aiperf.log" 2>&1
  wait "${profile_capture_pid}"
  profile_capture_pid=
  srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-image="${VLLM}" --container-mounts="$(mounts profile-validation)" /usr/bin/python3 "${ROOT}/scripts/validate-firmus-profile.py" "${ROOT}" >"${ROOT}/logs/profile-validation.log" 2>&1
  touch "${ROOT}/run-success"
  exit 0
fi

# Prometheus snapshots bracket the official pass; their deltas split TTFT into prefill queue, prefill, KV transfer and decoder admission without touching measured traffic.
while IFS=$'\t' read -r worker host port _tp; do curl --fail --silent --show-error --max-time 30 "http://${host}:${port}/metrics" >"${ROOT}/results/metrics-before-${worker}.prom"; done <"${ROOT}/source/worker-endpoints.tsv"
aiperf_status=0
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${ART}/aiperf.log" 2>&1 || aiperf_status=$?
while IFS=$'\t' read -r worker host port _tp; do curl --fail --silent --show-error --max-time 30 "http://${host}:${port}/metrics" >"${ROOT}/results/metrics-after-${worker}.prom"; done <"${ROOT}/source/worker-endpoints.tsv"
# The scheduler log reaches the Actions log, so print the decisive numbers there as well as in results/.
jq -c -n --slurpfile s "${ROOT}/results/score.json" --slurpfile e "${ROOT}/results/profile_export_aiperf.json" '{score:$s[0].score_output_tokens_per_second, eligible:$s[0].eligible, good_fraction:$s[0].good_request_fraction, requests:$s[0].request_count, good:$s[0].good_request_count, ttft_ms:($e[0].time_to_first_token | {avg, p50, p90, p99}), itl_ms:($e[0].inter_token_latency | {avg, p50, p90, p99})}' || true
(cd "${ROOT}/results" && grep -E '^vllm:(((request_queue_time|request_prefill_time|request_decode_time|time_to_first_token|inter_token_latency|e2e_request_latency|nixl_xfer_time|nixl_post_time)_seconds|nixl_bytes_transferred|nixl_num_descriptors|iteration_tokens_total)_(sum|count)|(prompt_tokens|generation_tokens|num_preemptions|spec_decode_num_drafts|spec_decode_num_draft_tokens|spec_decode_num_accepted_tokens)_total)[{ ]' metrics-before-*.prom metrics-after-*.prom) || true
(cd "${ART}" && grep -o 'DeepGEMM E8M0 [a-z]*' prefill-*.log decode-0.log | sort | uniq -c) || true
grep -hE 'Avg prompt throughput|SpecDecoding metrics|KV Transfer metrics' "${ART}/decode-0.log" | tail -n 30 || true
grep -hE 'Avg prompt throughput' "${ART}/prefill-0.log" "${ART}/prefill-4.log" | tail -n 30 || true
(( aiperf_status == 0 ))
echo "AIPERF ok"

touch "${ROOT}/run-success"
