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
WARMUP_REQUEST='{"model":"qwen","messages":[{"role":"user","content":"Name the capital of France and explain in two sentences why it is historically important."}],"max_tokens":64,"temperature":0}'

# The serving image has no jq, so the configuration is read here on the host.
config() { jq -er "$1" "${CONFIG}"; }
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
for worker in "${WORKERS[@]}"; do
  read -r name node address port bootstrap <<<"${worker}"
  printf '%s\t%s\t%s\t4\n' "${name}" "${address}" "${port}"
done >"${ENDPOINTS}"

# Container options of every srun step; Pyxis passes CONTAINER_ENV from the host environment.
CONTAINER=(--container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${IMAGE}")
# Container mounts of one step: shared scratch, a per-step HOME, the run directory, the draft, InfiniBand,
# and the SGLang and DeepGEMM packages patched by prepare.sh over the image's copies.
mounts() {
  mkdir -p "${ROOT}/home/$1"
  echo "--container-mounts=/scratch:/scratch,${ROOT}/home/$1:${HOME},${ROOT}:${ROOT},${ROOT}/draft:/draft,${MODEL}:${MODEL}:ro,/dev/infiniband:/dev/infiniband,${ROOT}/overlay/sglang/python/sglang:/opt/sglang/source/python/sglang,${ROOT}/overlay/deep-gemm/deep_gemm:/opt/sglang/venv/lib/python3.12/site-packages/deep_gemm"
}
wait_http() { until curl --fail --silent --max-time 2 -o /dev/null "$1"; do sleep 2; done; }

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} user=$(whoami)"
echo "head=${HEAD}(${HEAD_IP}) tail=${TAIL}(${TAIL_IP}) model=${MODEL} isl=4000 osl=200 N=${N} topology=P3-TP1-PP4-D1-TP4-DP4-EP4 kv=fp8_e4m3 speculation=EAGLE3 prefill=megamoe decode=megamoe router=live-prefill-min-load"

# ---- Start the workers and the router ----

for worker in "${WORKERS[@]}"; do
  read -r name node address port bootstrap <<<"${worker}"
  if [[ "${name}" == prefill-* ]]; then tp=1; role_args=("${prefill_args[@]}" --disaggregation-bootstrap-port "${bootstrap}"); else tp=4; role_args=("${decode_args[@]}"); fi
  srun --exact -N1 -n1 -w "${node}" --gres=gpu:4 --cpus-per-task=32 "${CONTAINER[@]}" "$(mounts "${name}")" bash "${ROOT}/scripts/worker.sh" "${name}" "${port}" "${MODEL}" "${tp}" "${address}" "${shared_args[@]}" "${role_args[@]}" >"${LOGS}/${name}.log" 2>&1 &
done
for worker in "${WORKERS[@]}"; do
  read -r name node address port bootstrap <<<"${worker}"
  wait_http "http://${address}:${port}/health"
  echo "${name} ready"
done

# The router assigns each request to the Prefill worker with the least live load.
router=(--pd-disaggregation --prefill "http://${HEAD_IP}:30100" 9000 --prefill "http://${HEAD_IP}:30101" 9001 --prefill "http://${TAIL_IP}:30102" 9002 --decode "http://${TAIL_IP}:30200")
router+=(--prefill-policy manual --assignment-mode min_load --host 0.0.0.0 --port "${ROUTER_PORT}")
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 "${CONTAINER[@]}" "$(mounts router)" /opt/apac/bin/sglang-router launch "${router[@]}" >"${LOGS}/router.log" 2>&1 &
# /health is liveness only; /readiness waits until a Prefill and a Decode worker are registered.
wait_http "http://${HEAD_IP}:${ROUTER_PORT}/readiness"
echo "READY after ${SECONDS}s"

# ---- Before the benchmark ----

# Each worker's effective settings.
while IFS=$'\t' read -r name address port _; do
  curl --fail --silent --show-error --max-time 15 "http://${address}:${port}/get_server_info" >"${RESULTS}/server-info-${name}.json"
done <"${ENDPOINTS}"
# One connection warm-up request through the gateway (not from the benchmark dataset), kept for inspection.
curl --fail --silent --show-error --max-time 120 -H 'Content-Type: application/json' -d "${WARMUP_REQUEST}" "http://${HEAD_IP}:${ROUTER_PORT}/v1/chat/completions" >"${RESULTS}/sanity-chat.json"
# Freeze the startup heap of every server process (SGLang /freeze_gc) so Python's full collections scan only
# request state: they otherwise stop the Decode output stream for ~0.33 s every 14-19 s.
while IFS=$'\t' read -r name address port _; do
  curl --fail --silent --show-error --max-time 30 -X POST "http://${address}:${port}/freeze_gc" >>"${LOGS}/freeze-gc.log" 2>&1
done <"${ENDPOINTS}"

# ---- Official benchmark ----

srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 "${CONTAINER[@]}" "$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${LOGS}/aiperf.log" 2>&1

echo "AIPERF ok"
touch "${ROOT}/run-success"

# ---- Optional diagnostics after the completed benchmark ----

# Each Decode rank logs once whether it uses the fused greedy verify kernels.
grep -a -h -o -E 'Decode greedy verify (using fused kernels|fallback).*' "${LOGS}/decode-0.log" | sort -u >"${RESULTS}/decode-greedy-dispatch.txt" || echo "Decode dispatch extraction failed; see logs/decode-0.log" >&2
# Each Prefill stage logs its FC1 first-K MegaMoE prewarm and which kernel it dispatches.
for worker in 0 1 2; do grep -a -h -o -E 'Prefill r32 fc1-first-k (prewarm done|using candidate|using control).*' "${LOGS}/prefill-${worker}.log" >"${RESULTS}/prefill-r32-dispatch-${worker}.txt" || echo "Prefill dispatch extraction failed for worker ${worker}; see its log" >&2; done
# Team-side output check: eight fixed greedy prompts through the gateway, 64 tokens each, with token logprobs.
# A stalled diagnostic must finish before Slurm ends the allocation. SIGTERM to srun kills its tasks.
# https://slurm.schedmd.com/srun.html (SIGNALS); leave 90 seconds for a 60-second probe and cleanup.
if [[ -z "${SLURM_JOB_END_TIME:-}" ]]; then
  echo "Skipping greedy probe: SLURM_JOB_END_TIME is unavailable" >&2
elif (( SLURM_JOB_END_TIME - $(date +%s) >= 90 )); then
  timeout --kill-after=5s 60s srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 "${CONTAINER[@]}" "$(mounts greedy-probe)" /opt/sglang/venv/bin/python "${ROOT}/scripts/greedy-probe.py" "${RESULTS}" "http://${HEAD_IP}:${ROUTER_PORT}/v1/chat/completions" qwen >"${LOGS}/greedy-probe.log" 2>&1 || echo "Greedy probe failed; see logs/greedy-probe.log" >&2
else
  echo "Skipping greedy probe: less than 90 seconds left in the allocation" >&2
fi
