#!/bin/bash
#SBATCH -J qwen-basic
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:8
#SBATCH --propagate=NONE
#SBATCH --export=PATH,HOME,USER,LOGNAME
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%j.log
#SBATCH -e logs/slurm-%j.stderr.log
# Qwen3 PD Basic job on Firmus: unmodified SGLang 0.5.23 on 2 nodes / 16 H200, then the official AIPerf benchmark.
#
#   node 0: prefill-0, prefill-1 and the gateway; node 1: prefill-2 and decode-0 (four GPUs per worker)
#
# Submit from a run directory prepared by submission/basic/prepare.sh:
#   sbatch --export=PATH,HOME,USER,LOGNAME scripts/submit.sbatch
# scripts/basic-worker.py builds each worker's public launch_server command from source/firmus-framework.json.
set -euo pipefail

# Pass the whole job environment to every srun step.
export SLURM_EXPORT_ENV=ALL
ROOT=${SLURM_SUBMIT_DIR}
CONFIG=${ROOT}/source/firmus-framework.json
IMAGE=${ROOT}/sglang.sqsh
MODEL=$(jq -er .model.path "${ROOT}/source/model.json")
N=$(jq -er .concurrency "${CONFIG}")
GATEWAY_PORT=8000
# Seconds each worker and the gateway may take to become ready.
READY_SECS=720
# Host variables the container must see (Pyxis --container-env).
CONTAINER_ENV=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
export MELLANOX_VISIBLE_DEVICES=all MELLANOX_MOUNT_DRIVER=yes

# ---- Nodes and workers ----

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#NODES[@]} -eq 2 ]]
ip_of() { getent ahostsv4 "$1" | awk '{print $1; exit}'; }
HEAD=${NODES[0]}
TAIL=${NODES[1]}
HEAD_IP=$(ip_of "${HEAD}")
TAIL_IP=$(ip_of "${TAIL}")

# role, index, node, address; HTTP ports are 30100 + index for Prefill and 30200 for Decode.
WORKERS=("prefill 0 ${HEAD} ${HEAD_IP}" "prefill 1 ${HEAD} ${HEAD_IP}" "prefill 2 ${TAIL} ${TAIL_IP}" "decode 0 ${TAIL} ${TAIL_IP}")
ENDPOINTS=${ROOT}/source/worker-endpoints.tsv
printf 'prefill-0\t%s\t30100\t4\nprefill-1\t%s\t30101\t4\nprefill-2\t%s\t30102\t4\ndecode-0\t%s\t30200\t4\n' "${HEAD_IP}" "${HEAD_IP}" "${TAIL_IP}" "${TAIL_IP}" >"${ENDPOINTS}"

# Container mounts of one srun step: the run directory, a per-step HOME, the model and the draft (read-only), InfiniBand.
mounts() {
  mkdir -p "${ROOT}/home/$1"
  printf '%s\n' "${ROOT}:${ROOT},${ROOT}/home/$1:${HOME},${MODEL}:/model:ro,${ROOT}/draft:/draft:ro,/dev/infiniband:/dev/infiniband"
}

# Wait until URL answers; fail as soon as any started server exits, or after READY_SECS.
wait_http() {
  local url=$1 deadline=$((SECONDS + READY_SECS)) pid status
  until curl --fail --silent --max-time 2 -o /dev/null "${url}"; do
    for pid in "${pids[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        status=0
        wait "${pid}" || status=$?
        printf 'worker_exited pid=%s status=%s\n' "${pid}" "${status}" >&2
        return 1
      fi
    done
    if (( SECONDS >= deadline )); then printf 'readiness_timeout %s\n' "${url}" >&2; return 124; fi
    sleep 2
  done
}

# Fetch every worker's server info and check its release, layout and KV pools against the configuration.
readback() {
  while IFS=$'\t' read -r name host port _; do
    curl --fail --silent --show-error --max-time 15 "http://${host}:${port}/server_info" >"${ROOT}/results/server-info-${name}.json"
    python3 "${ROOT}/scripts/check-basic-server-info.py" "${CONFIG}" "${ROOT}/results/server-info-${name}.json" "${name%%-*}"
  done <"${ENDPOINTS}"
}

# Stop every server this job started and clear their node-local scratch, whatever ends the job.
pids=()
cleanup() {
  local status=$? pid node
  trap - EXIT
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  for node in "${NODES[@]}"; do srun --overlap -N1 -n1 -w "${node}" --gres=gpu:0 --cpus-per-task=1 bash -c 'rm -rf -- /tmp/qwen-basic-"$SLURM_JOB_ID"-*' || true; done
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- Start the workers and the gateway ----

for worker in "${WORKERS[@]}"; do
  read -r role index node host <<<"${worker}"
  name=${role}-${index}
  srun --exact -N1 -n1 -w "${node}" --gres=gpu:4 --cpus-per-task=32 --container-env="${CONTAINER_ENV}" --container-image="${IMAGE}" --container-mounts="$(mounts "${name}")" /opt/sglang/bin/python "${ROOT}/scripts/basic-worker.py" "${ROOT}" "${role}" "${index}" "${host}" >"${ROOT}/logs/${name}.log" 2>&1 &
  pids+=("$!")
done
while IFS=$'\t' read -r name host port _; do
  wait_http "http://${host}:${port}/health"
done <"${ENDPOINTS}"
# The gateway shipped in the official image, with round-robin assignment on both roles.
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 --container-env="${CONTAINER_ENV}" --container-image="${IMAGE}" --container-mounts="$(mounts router)" /opt/sglang/bin/sgl-model-gateway --pd-disaggregation --prefill "http://${HEAD_IP}:30100" 9000 --prefill "http://${HEAD_IP}:30101" 9001 --prefill "http://${TAIL_IP}:30102" 9002 --decode "http://${TAIL_IP}:30200" --prefill-policy round_robin --decode-policy round_robin --host 0.0.0.0 --port "${GATEWAY_PORT}" >"${ROOT}/logs/router.log" 2>&1 &
pids+=("$!")
wait_http "http://${HEAD_IP}:${GATEWAY_PORT}/readiness"
printf 'READY after %ss\n' "${SECONDS}"

# ---- Before the benchmark ----

readback
# One connection warm-up request through the gateway (not from the benchmark dataset), kept for inspection.
curl --fail --silent --show-error --max-time 120 -H 'Content-Type: application/json' -d '{"model":"qwen","messages":[{"role":"user","content":"Name the capital of France and explain its historical importance in two sentences."}],"max_tokens":64,"temperature":0}' "http://${HEAD_IP}:${GATEWAY_PORT}/v1/chat/completions" >"${ROOT}/results/sanity-chat.json"

# ---- Official benchmark ----

status=0
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=32 --container-env="${CONTAINER_ENV}" --container-image="${IMAGE}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${ROOT}/logs/aiperf.log" 2>&1 || status=$?

# ---- After the benchmark ----

readback
# EAGLE acceptance statistics from the server info and logs (diagnostic; not part of the score).
python3 "${ROOT}/scripts/collect-eagle-acceptance.py" --config "${CONFIG}" --endpoints "${ENDPOINTS}" --server-info-dir "${ROOT}/results" --log-dir "${ROOT}/logs" --output "${ROOT}/results/eagle-acceptance.json" --cluster firmus
(( status == 0 )) || exit "${status}"
test -s "${ROOT}/results/score.json"
echo 'AIPERF ok'
touch "${ROOT}/run-success"
