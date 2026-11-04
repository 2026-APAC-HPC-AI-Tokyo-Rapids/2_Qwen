#!/bin/bash
#SBATCH -J sglang-firmus-dp4-ep4
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:8
#SBATCH --propagate=NONE
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%j.log
#SBATCH -e logs/slurm-%j.stderr.log
# Qwen3 PD Bonus job on Firmus: serve the model on 2 nodes / 16 H200 and run the official AIPerf benchmark.
#
#   node 0: prefill-0, prefill-1 (each TP1 x PP4, one pipeline stage per GPU) and the router
#   node 1: prefill-2 and decode-0 (TP4, DP-attention 4, EP4)
#
# Submit from a run directory prepared by submission/run/prepare.sh: sbatch scripts/submit.sbatch
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR}
CONFIG=${ROOT}/source/firmus-framework.json
IMAGE=${ROOT}/sglang.sqsh
MODEL=$(jq -er .model.path "${ROOT}/source/model.json")
LOGS=${ROOT}/logs
RESULTS=${ROOT}/results
ROUTER_PORT=8000
# Seconds each worker and the router may take to become ready.
READY_SECS=600
# Host variables the container must see (Pyxis --container-env).
CONTAINER_ENV=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER

# The serving image has no jq, so the configuration is read here on the host.
config() { jq -er "$1" "${CONFIG}"; }
jq -e '.framework == "sglang" and .workload == "official" and .prefill.pp == 4 and .prefill.tp == 1 and .speculation.draft_quantization == "fp8"' "${CONFIG}" >/dev/null
N=$(config .concurrency)

# ---- Worker flags (worker.sh adds the flags every worker shares) ----

# EAGLE3 with the draft quantized to FP8 at load, and the configured tokenizer backend.
shared_args=(--speculative-algorithm EAGLE3 --speculative-draft-model-path /draft --speculative-num-steps "$(config .speculation.steps)" --speculative-eagle-topk 1 --speculative-num-draft-tokens "$(config .speculation.draft_tokens)")
shared_args+=(--speculative-draft-model-quantization fp8 --tokenizer-backend "$(config .tokenizer_backend)")

# Prefill: four pipeline stages with up to two micro-batches, no expert parallelism, MegaMoE, no CUDA graph.
prefill_args=(--pp-size 4 --pp-max-micro-batch-size 2 --pp-async-batch-depth "$(config .prefill.pp_async_batch_depth)" --ep-size 1 --dp-size 1)
prefill_args+=(--chunked-prefill-size 8192 --max-total-tokens 131072 --max-running-requests "${N}")
prefill_args+=(--moe-a2a-backend megamoe --moe-runner-backend deep_gemm --disable-cuda-graph --disable-overlap-schedule)
prefill_args+=(--init-expert-location trivial --ep-num-redundant-experts 0 --optimistic-prefill-attempts 0)

# Decode: DP attention and DP LM head on four GPUs, EP4 MegaMoE, CUDA graphs at the configured batch sizes.
mapfile -t decode_graph < <(config '.decode_cuda_graph_bs[]')
decode_args=(--cuda-graph-bs-decode "${decode_graph[@]}" --ep-size 4 --dp-size "$(config .decode_dp_size)" --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1)
decode_args+=(--moe-a2a-backend megamoe --moe-runner-backend deep_gemm --chunked-prefill-size 8192 --max-running-requests "$(config .decode_max_running_requests)")
decode_args+=(--disaggregation-decode-extra-slots "$(config .decode_extra_slots)" --disaggregation-decode-polling-interval "$(config .decode_polling_interval)" --max-total-tokens "$(config .decode_max_total_tokens)")

export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
export MELLANOX_VISIBLE_DEVICES=all MELLANOX_MOUNT_DRIVER=yes
# Idle Prefill ranks run MegaMoE with a fixed 256-row geometry (megamoe-idle-prefill-geometry.patch; Prefill only).
export SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY=1

# ---- Nodes and workers ----

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD=${NODES[0]}
TAIL=${NODES[1]}
ip_of() { getent ahostsv4 "$1" | awk '{print $1; exit}'; }
HEAD_IP=$(ip_of "${HEAD}")
TAIL_IP=$(ip_of "${TAIL}")

# name, node, address, HTTP port, bootstrap port (Prefill only)
WORKERS=("prefill-0 ${HEAD} ${HEAD_IP} 30100 9000" "prefill-1 ${HEAD} ${HEAD_IP} 30101 9001" "prefill-2 ${TAIL} ${TAIL_IP} 30102 9002" "decode-0 ${TAIL} ${TAIL_IP} 30200 -")
ENDPOINTS=${ROOT}/source/worker-endpoints.tsv

# Container mounts of one srun step: shared scratch, a per-step HOME, the run directory, the draft,
# InfiniBand, and the SGLang and DeepGEMM packages patched by prepare.sh over the image's copies.
mounts() {
  mkdir -p "${ROOT}/home/$1"
  echo "/scratch:/scratch,${ROOT}/home/$1:${HOME},${ROOT}:${ROOT},${ROOT}/draft:/draft,/dev/infiniband:/dev/infiniband,${ROOT}/overlay/sglang/python/sglang:/opt/sglang/source/python/sglang,${ROOT}/overlay/deep-gemm/deep_gemm:/opt/sglang/venv/lib/python3.12/site-packages/deep_gemm"
}

# Wait until URL answers; fail as soon as any started server exits, or after READY_SECS.
wait_http() {
  local url=$1 deadline=$((SECONDS + READY_SECS)) pid status
  while ! curl --fail --silent --max-time 2 -o /dev/null "${url}"; do
    for pid in "${server_pids[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        status=0
        wait "${pid}" || status=$?
        printf 'server_exited pid=%s status=%s waiting_for=%s\n' "${pid}" "${status}" "${url}" >&2
        return $(( status == 0 ? 1 : status ))
      fi
    done
    if (( SECONDS >= deadline )); then printf 'server_readiness_timeout url=%s\n' "${url}" >&2; return 124; fi
    sleep 2
  done
}

# Stop every server this job started, whatever ends the job.
server_pids=()
stop_servers() {
  local status=$? pid
  trap - EXIT INT TERM
  for pid in "${server_pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${server_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  exit "${status}"
}
trap stop_servers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} user=$(whoami)"
echo "head=${HEAD}(${HEAD_IP}) tail=${TAIL}(${TAIL_IP}) model=${MODEL} isl=4000 osl=200 N=${N} topology=P3-TP1-PP4-D1-TP4-DP4-EP4 kv=fp8_e4m3 speculation=EAGLE3 prefill=megamoe decode=megamoe router=live-prefill-min-load"
test -r "${IMAGE}" && test -f "${MODEL}/config.json" && test -s "${ROOT}/draft/model.safetensors"
test -s "${ROOT}/source/prefill-kernels.env" && test -s "${ROOT}/source/decode-kernels.env"

# ---- Start the workers and the router ----

for worker in "${WORKERS[@]}"; do
  read -r name node address port bootstrap <<<"${worker}"
  printf '%s\t%s\t%s\t4\n' "${name}" "${address}" "${port}"
done >"${ENDPOINTS}"

for worker in "${WORKERS[@]}"; do
  read -r name node address port bootstrap <<<"${worker}"
  if [[ "${name}" == prefill-* ]]; then tp=1; role_args=("${prefill_args[@]}" --disaggregation-bootstrap-port "${bootstrap}"); else tp=4; role_args=("${decode_args[@]}"); fi
  srun --exact -N1 -n1 -w "${node}" --gres=gpu:4 --cpus-per-task=32 --container-env="${CONTAINER_ENV}" --container-image="${IMAGE}" --container-mounts="$(mounts "${name}")" bash "${ROOT}/scripts/worker.sh" "${name}" "${port}" "${MODEL}" "${tp}" "${address}" "${shared_args[@]}" "${role_args[@]}" >"${LOGS}/${name}.log" 2>&1 &
  server_pids+=("$!")
done
for worker in "${WORKERS[@]}"; do
  read -r name node address port bootstrap <<<"${worker}"
  wait_http "http://${address}:${port}/health"
  echo "${name} ready"
done

# The router assigns each request to the Prefill worker with the least live load.
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 --container-env="${CONTAINER_ENV}" --container-image="${IMAGE}" --container-mounts="$(mounts router)" /opt/apac/bin/sglang-router launch --pd-disaggregation --prefill "http://${HEAD_IP}:30100" 9000 --prefill "http://${HEAD_IP}:30101" 9001 --prefill "http://${TAIL_IP}:30102" 9002 --decode "http://${TAIL_IP}:30200" --prefill-policy manual --assignment-mode min_load --host 0.0.0.0 --port "${ROUTER_PORT}" >"${LOGS}/router.log" 2>&1 &
server_pids+=("$!")
# /health is liveness only; /readiness waits until a Prefill and a Decode worker are registered.
wait_http "http://${HEAD_IP}:${ROUTER_PORT}/readiness"
echo "READY after ${SECONDS}s"

# ---- Before the benchmark ----

# Record each worker's effective settings, and check that the draft really loaded in FP8.
while IFS=$'\t' read -r name address port _; do
  curl --fail --silent --show-error --max-time 15 "http://${address}:${port}/get_server_info" >"${RESULTS}/server-info-${name}.json"
  jq -e '.speculative_draft_model_quantization == "fp8"' "${RESULTS}/server-info-${name}.json" >/dev/null
done <"${ENDPOINTS}"
# One connection warm-up request through the gateway (not from the benchmark dataset), kept for inspection.
curl --fail --silent --show-error --max-time 120 -H 'Content-Type: application/json' -d '{"model":"qwen","messages":[{"role":"user","content":"Name the capital of France and explain in two sentences why it is historically important."}],"max_tokens":64,"temperature":0}' "http://${HEAD_IP}:${ROUTER_PORT}/v1/chat/completions" >"${RESULTS}/sanity-chat.json"
# Freeze the startup heap of every server process (SGLang /freeze_gc) so Python's full collections scan only
# request state: they otherwise stop the Decode output stream for ~0.33 s every 14-19 s. A failed call keeps GC as it was.
while IFS=$'\t' read -r name address port _; do
  curl --fail --silent --show-error --max-time 30 -X POST "http://${address}:${port}/freeze_gc" >>"${LOGS}/freeze-gc.log" 2>&1 || printf 'freeze_gc failed on %s\n' "${name}" >>"${LOGS}/freeze-gc.log"
done <"${ENDPOINTS}"

# ---- Official benchmark ----

benchmark_status=0
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-env="${CONTAINER_ENV}" --container-image="${IMAGE}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${LOGS}/aiperf.log" 2>&1 || benchmark_status=$?
test -s "${RESULTS}/score.json"

# ---- After the benchmark ----

# Each Decode rank logs once whether it uses the fused greedy verify kernels or the fallback.
grep -a -h -o -E 'Decode greedy verify (using fused kernels|fallback).*' "${LOGS}/decode-0.log" | sort -u >"${RESULTS}/decode-greedy-dispatch.txt" || printf 'no Decode greedy verify dispatch line in decode-0.log\n' >"${RESULTS}/decode-greedy-dispatch.txt"
if (( benchmark_status == 0 )); then
  while IFS=$'\t' read -r name address port _; do
    curl --fail --silent --show-error --max-time 15 "http://${address}:${port}/get_server_info" >"${RESULTS}/server-info-${name}.json"
  done <"${ENDPOINTS}"
  # Team-side output check: eight fixed greedy prompts through the gateway, 64 tokens each, with token logprobs.
  srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 --container-env="${CONTAINER_ENV}" --container-image="${IMAGE}" --container-mounts="$(mounts greedy-probe)" /opt/sglang/venv/bin/python "${ROOT}/scripts/greedy-probe.py" "${RESULTS}" "http://${HEAD_IP}:${ROUTER_PORT}/v1/chat/completions" qwen >"${LOGS}/greedy-probe.log" 2>&1
fi
if (( benchmark_status != 0 )); then exit "${benchmark_status}"; fi
echo "AIPERF ok"
touch "${ROOT}/run-success"
