#!/bin/bash
#SBATCH -J run
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:8
#SBATCH --propagate=NONE
#SBATCH --no-requeue
#SBATCH -o runs/sglang-%j.out
#SBATCH -e runs/sglang-%j.err
# Komori qualified6344.96 architecture: P3 DP4/EP4 plus D1 DP4/EP4, attention TP1.
# Exact modified SM90 MegaMoE owns dispatch/math/combine; KV budgets are per DP rank.
# Profile allocations validate adapter math before loading the full serving topology.
# Reviewed capture uses the exact Komori client and PDL-on image pinned in config.
# Keep this P48 diagnostic separate from the uninstrumented baseline and P2 hypothesis.
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR}
VLLM=${ROOT}/vllm.sqsh
AIPERF=${ROOT}/sglang.sqsh
MODEL=/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8
R_PORT=8000
READY_SECS=600  # ASPIRE full stack took 7m09s; leave time for the 60/120/30 client.
N=$(jq -er .concurrency "${ROOT}/source/firmus-framework.json")
# Parse configuration on the host; the pinned serving image does not include jq.
speculation=(--speculative-config '{"method":"eagle3","model":"/draft","num_speculative_tokens":3}')
# Native mixed EAGLE batches require PIECEWISE as well as full decode graphs.
decode_graph=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":128}' --cudagraph-capture-sizes)
while read -r size; do decode_graph+=("${size}"); done < <(jq -er '.decode_cuda_graph_bs[] | . * 4' "${ROOT}/source/firmus-framework.json")
ART=${ROOT}/logs
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
export MELLANOX_VISIBLE_DEVICES=all MELLANOX_MOUNT_DRIVER=yes
[[ "${PROFILE:-false}" == false || "${PROFILE:-false}" == true ]]
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
  if [[ "${PROFILE:-false}" == true && ( "$1" == prefill-* || "$1" == decode-* || "$1" == profile-validation ) ]]; then profiler_mount=",${ROOT}/nsys:/opt/nsys:ro"; fi
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
echo "head=${HEAD}(${HEAD_IP}) tail=${TAIL}(${TAIL_IP}) model=${MODEL} isl=4000 osl=200 N=${N} topology=P3DP4EP4-D1DP4EP4-AttentionTP1 kv=fp8_e4m3 speculation=EAGLE3 kernels=apac_sm90_megamoe"
test -r "${VLLM}" && test -r "${AIPERF}" && test -f "${MODEL}/config.json" && test -s "${ROOT}/draft/model.safetensors"
mkdir -p "${ART}"
worker_pids=()
capture_pid=
cleanup_all() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${capture_pid}" ]]; then kill "${capture_pid}" 2>/dev/null || true; wait "${capture_pid}" 2>/dev/null || true; fi
  for pid in "${worker_pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${worker_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup_all EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# The PDL-on installed adapter passed 64 numerical/graph records in 38801099834.
# Keep that finite check on the profile path; normal official runs stay uninstrumented.
if [[ "${PROFILE:-false}" == true ]]; then
  # Reuse this allocation and exact serving image; no profiler during numerical validation.
  NVIDIA_VISIBLE_DEVICES=0,1,2,3 timeout --signal=TERM --kill-after=30s 300s srun --mpi=none --export=ALL --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=16 --gpu-bind=none --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${VLLM}" --container-mounts="$(mounts adapter-check)" bash -c 'export CUDA_VISIBLE_DEVICES=0,1,2,3; export HOME="$1/home/adapter-check" XDG_CACHE_HOME="$1/home/adapter-check/.cache" TRITON_CACHE_DIR="$1/home/adapter-check/.triton"; exec torchrun --standalone --nproc_per_node=4 "$1/scripts/megamoe/adapter_gpu_gate.py" --output "$1/results/gate.json"' gate "${ROOT}" >"${ART}/adapter-check.log" 2>&1
  jq -e '.completed == true and .correctness_passed == true and .graph_passed == true and .native_factory_runner == true and .capacity_requested == 4096 and (.cases | length) == 4 and ([.cases[].records[]] | length) == 64' "${ROOT}/results/gate.json" >/dev/null
fi
for worker in 0 1; do printf 'prefill-%s\t%s\t%s\t1\n' "${worker}" "${HEAD_IP}" "$((30100 + worker * 10))"; done >"${ROOT}/source/worker-endpoints.tsv"
printf 'prefill-2\t%s\t30120\t1\n' "${TAIL_IP}" >>"${ROOT}/source/worker-endpoints.tsv"
printf 'decode-0\t%s\t30200\t1\n' "${TAIL_IP}" >>"${ROOT}/source/worker-endpoints.tsv"

cat >"${ROOT}/source/worker.sh" <<'EOF'
set -euo pipefail
name=$1
port=$2
model=$3
tp=$4
export VLLM_NIXL_SIDE_CHANNEL_HOST=$5
shift 5
[[ -r /opt/apac/prefill-handoff-repair.json ]] || { echo "FATAL: serving image lacks the reviewed prefill handoff repair"; exit 1; }
python3 -c 'import json, sys; r = json.load(open(sys.argv[1])); sys.exit(not (r.get("push_worker_source_sha256") == "87de3ea383dd3aa95fb2c287ea26ded8a330859cc1295f9faa69047e80865c2c" and r.get("push_release_cpu_contract", {}).get("passed") is True))' /opt/apac/prefill-handoff-repair.json || { echo "FATAL: serving image lacks the producer WRITE release repair"; exit 1; }
role=${name%%-*}
engine_id="$(basename "$(dirname "$(dirname "$0")")")-${name}"
export VLLM_NIXL_SIDE_CHANNEL_PORT="$((port - 24500))"
export VLLM_USE_NCCL_SYMM_MEM=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
# Serve the published FP32 block scales: with E8M0 on, the pinned vLLM re-quantizes DeepGEMM dense weights to power-of-two scales on Hopper.
export VLLM_USE_DEEP_GEMM_E8M0=0
# Normal decoder startup fails at the first M<32 FlashInfer swapAB shape.
# Use native DeepGEMM dense FP8 for both roles; MegaMoE stays unchanged.
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
role_flags=()
blocks=2048
batch_tokens=4096
if [[ "$role" == prefill ]]; then
  # The winning SGLang prefill is eager; disable target and draft graphs here.
  role_flags=(--enforce-eager)
  export VLLM_ALLREDUCE_USE_SYMM_MEM=1
  export VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0
  KV_CONFIG="{\"kv_connector\":\"NixlPushConnector\",\"kv_role\":\"kv_producer\",\"engine_id\":\"${engine_id}\"}"
else
  # The winning decoder pool is393216 tokens PER DP rank, not divided by4.
  blocks=6144
  batch_tokens=2048
  KV_CONFIG="{\"kv_connector\":\"NixlPushConnector\",\"kv_role\":\"kv_consumer\",\"engine_id\":\"${engine_id}\"}"
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
if [[ -x /opt/nsys/bin/nsys ]]; then
  # Keep normal serving log verbosity after the dense-kernel startup repair.
  # Inspect both nodes before serving; a separate login-launched step could not start.
  if [[ "$name" == prefill-0 || "$name" == decode-0 ]]; then
    for device in /sys/class/infiniband/*; do
      [[ -d "$device/ports/1" ]] || continue
      printf 'NIC device=%s link_layer=%s state=%s rate=%s\n' "${device##*/}" "$(cat "$device/ports/1/link_layer")" "$(cat "$device/ports/1/state")" "$(cat "$device/ports/1/rate")"
      for gid in "$device/ports/1/gids/"*; do
        [[ -r "$gid" ]] || continue
        value=$(cat "$gid")
        [[ -n "${value//[0:]/}" ]] || continue
        index=${gid##*/}
        printf 'NIC_GID device=%s index=%s value=%s netdev=%s type=%s\n' "${device##*/}" "$index" "$value" "$(cat "$device/ports/1/gid_attrs/ndevs/$index" 2>/dev/null)" "$(cat "$device/ports/1/gid_attrs/types/$index" 2>/dev/null)"
      done
    done
  fi
fi
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
echo "$name host=$(hostname) ip=$VLLM_NIXL_SIDE_CHANNEL_HOST hca=$UCX_IB gpu=$CUDA_VISIBLE_DEVICES tp=$tp symm=$VLLM_ALLREDUCE_USE_SYMM_MEM"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit --format=csv
worker_command=(/usr/local/bin/vllm serve "$model" --served-model-name qwen --trust-remote-code --dtype bfloat16 --max-model-len 8192 --tensor-parallel-size "$tp" --gpu-memory-utilization 0.90 --block-size 64 --num-gpu-blocks-override "$blocks" --max-num-batched-tokens "$batch_tokens" --enable-chunked-prefill --kv-cache-dtype fp8_e4m3 --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":3}' --kv-transfer-config "$KV_CONFIG" "${role_flags[@]}" "$@" --host 0.0.0.0 --port "$port")
if [[ -x /opt/nsys/bin/nsys ]]; then worker_command+=(--profiler-config '{"profiler":"cuda"}'); fi
printf '%q ' "${worker_command[@]}" >"$(dirname "$0")/argv-${name}.txt"
printf '\n' >>"$(dirname "$0")/argv-${name}.txt"
if [[ -x /opt/nsys/bin/nsys ]]; then
  export QWEN_PROFILER_NSYS_BIN=/opt/nsys/bin/nsys
  export QWEN_PROFILER_INVOCATION_PATH="$(dirname "$(dirname "$0")")/profile/nsys-${name}.argv"
  exec bash "$(dirname "$(dirname "$0")")/scripts/exec-profiler.sh" "${worker_command[@]}"
fi
exec "${worker_command[@]}"
EOF

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${VLLM}" --container-mounts="$(mounts prefill-0)" bash "${ROOT}/source/worker.sh" prefill-0 30100 "${MODEL}" 1 "${HEAD_IP}" "${speculation[@]}" --disable-custom-all-reduce --enable-expert-parallel --data-parallel-size 4 --data-parallel-size-local 4 --data-parallel-backend mp --all2all-backend allgather_reducescatter --moe-backend apac_sm90_megamoe --max-num-seqs 48 >"${ART}/prefill-0.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${VLLM}" --container-mounts="$(mounts prefill-1)" bash "${ROOT}/source/worker.sh" prefill-1 30110 "${MODEL}" 1 "${HEAD_IP}" "${speculation[@]}" --disable-custom-all-reduce --enable-expert-parallel --data-parallel-size 4 --data-parallel-size-local 4 --data-parallel-backend mp --all2all-backend allgather_reducescatter --moe-backend apac_sm90_megamoe --max-num-seqs 48 >"${ART}/prefill-1.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${VLLM}" --container-mounts="$(mounts prefill-2)" bash "${ROOT}/source/worker.sh" prefill-2 30120 "${MODEL}" 1 "${TAIL_IP}" "${speculation[@]}" --disable-custom-all-reduce --enable-expert-parallel --data-parallel-size 4 --data-parallel-size-local 4 --data-parallel-backend mp --all2all-backend allgather_reducescatter --moe-backend apac_sm90_megamoe --max-num-seqs 48 >"${ART}/prefill-2.log" 2>&1 &
worker_pids+=("$!")
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${VLLM}" --container-mounts="$(mounts decode-0)" bash "${ROOT}/source/worker.sh" decode-0 30200 "${MODEL}" 1 "${TAIL_IP}" "${speculation[@]}" "${decode_graph[@]}" --disable-custom-all-reduce --enable-expert-parallel --data-parallel-size 4 --data-parallel-size-local 4 --data-parallel-backend mp --all2all-backend allgather_reducescatter --moe-backend apac_sm90_megamoe --max-num-seqs 32 --no-enable-prefix-caching >"${ART}/decode-0.log" 2>&1 &
worker_pids+=("$!")

wait_http "http://${HEAD_IP}:30100/health"; echo "prefill-0 ready"
wait_http "http://${HEAD_IP}:30110/health"; echo "prefill-1 ready"
wait_http "http://${TAIL_IP}:30120/health"; echo "prefill-2 ready"
wait_http "http://${TAIL_IP}:30200/health"; echo "decode-0 ready"

srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${VLLM}" --container-mounts="$(mounts router)" /usr/bin/python3 "${ROOT}/vllm-proxy.py" --host 0.0.0.0 --port "${R_PORT}" --prefiller-hosts "${HEAD_IP}" "${HEAD_IP}" "${TAIL_IP}" --prefiller-ports 30100 30110 30120 --prefiller-engine-ids "${ROOT##*/}-prefill-0" "${ROOT##*/}-prefill-1" "${ROOT##*/}-prefill-2" --decoder-hosts "${TAIL_IP}" --decoder-ports 30200 >"${ART}/router.log" 2>&1 &
worker_pids+=("$!")

wait_http "http://${HEAD_IP}:${R_PORT}/v1/models"
echo "READY after ${SECONDS}s"

while IFS=$'\t' read -r worker host port _tp; do
  curl --fail --silent --show-error --max-time 30 "http://${host}:${port}/server_info?config_format=json" >"${ROOT}/results/server-info-${worker}.json"
  jq -e --arg role "${worker%%-*}" --argjson expected_tp "${_tp}" --slurpfile config "${ROOT}/source/firmus-framework.json" '.vllm_config as $v | $expected_tp == 1 and $v.parallel_config.tensor_parallel_size == 1 and $v.kv_transfer_config.kv_connector == "NixlPushConnector" and $v.kernel_config.moe_backend == "apac_sm90_megamoe" and $v.parallel_config.enable_expert_parallel == true and $v.parallel_config.data_parallel_size == 4 and $v.parallel_config.all2all_backend == "allgather_reducescatter" and $v.cache_config.cache_dtype == "fp8_e4m3" and $v.cache_config.block_size == 64 and ($v.cache_config.block_size * $v.cache_config.num_gpu_blocks_override) == (if $role == "prefill" then 131072 else $config[0].decode_max_total_tokens end) and $v.model_config.max_model_len == 8192 and $v.scheduler_config.max_num_batched_tokens == (if $role == "prefill" then 4096 else 2048 end) and (if $role == "prefill" then $v.model_config.enforce_eager == true else $v.cache_config.enable_prefix_caching == false and $v.compilation_config.max_cudagraph_capture_size == 128 and $v.compilation_config.cudagraph_mode == [2,1] and $v.compilation_config.cudagraph_capture_sizes == [$config[0].decode_cuda_graph_bs[] | . * 4] end) and $v.speculative_config.draft_parallel_config.enable_expert_parallel == false and $v.speculative_config.draft_parallel_config.tensor_parallel_size == 1 and $v.speculative_config.method == "eagle3" and $v.speculative_config.model == "/draft" and $v.speculative_config.num_speculative_tokens == 3 and $v.scheduler_config.max_num_seqs == (if $role == "prefill" then 48 else 32 end)' "${ROOT}/results/server-info-${worker}.json" >/dev/null
done <"${ROOT}/source/worker-endpoints.tsv"
if [[ "${PROFILE:-false}" == true ]]; then
  mkdir -p "${ROOT}/profile/client"
  : >"${ROOT}/profile/client/aiperf.log"
  printf '%s\n' 'Instrumented diagnostic: all 16 GPUs, CUDA API range starts five seconds into measurement and ends after request/grace completion; one-second CUPTI flush interval; not an official throughput result.' >"${ROOT}/profile/interpretation.txt"
  bash "${ROOT}/scripts/capture-firmus-profile.sh" "${ROOT}" "${ROOT}/profile/client/aiperf.log" >"${ROOT}/profile/capture.log" 2>&1 &
  capture_pid=$!
  aiperf_status=0
  srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" profile >"${ROOT}/profile/client/aiperf.log" 2>&1 || aiperf_status=$?
  capture_status=0
  wait "${capture_pid}" || capture_status=$?
  capture_pid=
  validation_status=0
  srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${VLLM}" --container-mounts="$(mounts profile-validation)" python3 "${ROOT}/scripts/validate-firmus-profile.py" "${ROOT}" >"${ROOT}/profile/validation.log" 2>&1 || validation_status=$?
  (( aiperf_status == 0 && capture_status == 0 && validation_status == 0 ))
  touch "${ROOT}/run-success"
  exit 0
fi
# Prometheus snapshots bracket warmup plus measurement. Native intervals overlap;
# concurrent D queue includes P/KV waiting, not an additive critical path.
while IFS=$'\t' read -r worker host port _tp; do curl --fail --silent --show-error --max-time 30 "http://${host}:${port}/metrics" >"${ROOT}/results/metrics-before-${worker}.prom"; done <"${ROOT}/source/worker-endpoints.tsv"
aiperf_status=0
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${ART}/aiperf.log" 2>&1 || aiperf_status=$?
while IFS=$'\t' read -r worker host port _tp; do curl --fail --silent --show-error --max-time 30 "http://${host}:${port}/metrics" >"${ROOT}/results/metrics-after-${worker}.prom"; done <"${ROOT}/source/worker-endpoints.tsv"
# The scheduler log reaches the Actions log, so print the decisive numbers there as well as in results/.
jq -c -n --slurpfile s "${ROOT}/results/score.json" --slurpfile e "${ROOT}/results/profile_export_aiperf.json" '{good_output_tokens_per_second:($s[0].goodput_requests_per_second * 200), overall_output_tokens_per_second:$e[0].output_token_throughput.avg, local_gated_value:$s[0].score_output_tokens_per_second, eligible:$s[0].eligible, good_fraction:$s[0].good_request_fraction, requests:$s[0].request_count, good:$e[0].good_request_count.avg, ttft_ms:($e[0].time_to_first_token | {avg, p50, p90, p95, p99}), itl_ms:($e[0].inter_token_latency | {avg, p50, p90, p95, p99})}' || true
(cd "${ROOT}/results" && grep -E '^vllm:(((request_queue_time|request_prefill_time|request_decode_time|time_to_first_token|inter_token_latency|e2e_request_latency|nixl_xfer_time|nixl_post_time)_seconds|nixl_bytes_transferred|nixl_num_descriptors|iteration_tokens_total)_(sum|count)|(prompt_tokens|generation_tokens|num_preemptions|spec_decode_num_drafts|spec_decode_num_draft_tokens|spec_decode_num_accepted_tokens)_total)[{ ]' metrics-before-*.prom metrics-after-*.prom) || true
(cd "${ART}" && grep -o 'DeepGEMM E8M0 [a-z]*' prefill-*.log decode-0.log | sort | uniq -c) || true
grep -hE 'Avg prompt throughput|SpecDecoding metrics|KV Transfer metrics' "${ART}/decode-0.log" | tail -n 30 || true
grep -hE 'Avg prompt throughput' "${ART}/prefill-0.log" "${ART}/prefill-2.log" | tail -n 30 || true
(( aiperf_status == 0 ))
echo "AIPERF ok"

touch "${ROOT}/run-success"
