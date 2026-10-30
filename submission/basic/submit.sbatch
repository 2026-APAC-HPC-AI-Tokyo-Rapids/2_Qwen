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
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
ROOT=${SLURM_SUBMIT_DIR}
MODEL=$(jq -er .model.path "${ROOT}/source/model.json")
N=$(jq -er .concurrency "${ROOT}/source/firmus-framework.json")
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#NODES[@]} -eq 2 ]]
ip_of() { getent ahostsv4 "$1" | awk '{print $1; exit}'; }
HEAD=${NODES[0]}
TAIL=${NODES[1]}
HEAD_IP=$(ip_of "${HEAD}")
TAIL_IP=$(ip_of "${TAIL}")
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
export MELLANOX_VISIBLE_DEVICES=all MELLANOX_MOUNT_DRIVER=yes
pids=()
cleanup() {
  status=$?
  trap - EXIT
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  for node in "${NODES[@]}"; do srun --overlap -N1 -n1 -w "${node}" --gres=gpu:0 --cpus-per-task=1 bash -c 'rm -rf -- /tmp/qwen-basic-"$SLURM_JOB_ID"-*' || true; done
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mounts() {
  mkdir -p "${ROOT}/home/$1"
  printf '%s\n' "${ROOT}:${ROOT},${ROOT}/home/$1:${HOME},${MODEL}:/model:ro,${ROOT}/draft:/draft:ro,/dev/infiniband:/dev/infiniband"
}
wait_http() {
  local deadline=$((SECONDS + 720)) status
  until curl --fail --silent --max-time 2 -o /dev/null "$1"; do
    for pid in "${pids[@]}"; do if ! kill -0 "${pid}" 2>/dev/null; then status=0; wait "${pid}" || status=$?; printf 'worker_exited pid=%s status=%s\n' "${pid}" "${status}" >&2; return 1; fi; done
    if (( SECONDS >= deadline )); then printf 'readiness_timeout %s\n' "$1" >&2; return 124; fi
    sleep 2
  done
}
printf 'prefill-0\t%s\t30100\t4\nprefill-1\t%s\t30101\t4\nprefill-2\t%s\t30102\t4\ndecode-0\t%s\t30200\t4\n' "${HEAD_IP}" "${HEAD_IP}" "${TAIL_IP}" "${TAIL_IP}" >"${ROOT}/source/worker-endpoints.tsv"
for item in "prefill:0:${HEAD}:${HEAD_IP}" "prefill:1:${HEAD}:${HEAD_IP}" "prefill:2:${TAIL}:${TAIL_IP}" "decode:0:${TAIL}:${TAIL_IP}"; do
  IFS=: read -r role index node host <<<"${item}"
  name=${role}-${index}
  srun --exact -N1 -n1 -w "${node}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${ROOT}/sglang.sqsh" --container-mounts="$(mounts "${name}")" /opt/sglang/bin/python "${ROOT}/scripts/basic-worker.py" "${ROOT}" "${role}" "${index}" "${host}" >"${ROOT}/logs/${name}.log" 2>&1 &
  pids+=("$!")
done
while IFS=$'\t' read -r name host port _; do wait_http "http://${host}:${port}/health"; done <"${ROOT}/source/worker-endpoints.tsv"
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${ROOT}/sglang.sqsh" --container-mounts="$(mounts router)" /opt/sglang/bin/sgl-model-gateway --pd-disaggregation --prefill "http://${HEAD_IP}:30100" 9000 --prefill "http://${HEAD_IP}:30101" 9001 --prefill "http://${TAIL_IP}:30102" 9002 --decode "http://${TAIL_IP}:30200" --prefill-policy round_robin --decode-policy round_robin --host 0.0.0.0 --port 8000 >"${ROOT}/logs/router.log" 2>&1 &
pids+=("$!")
wait_http "http://${HEAD_IP}:8000/readiness"
printf 'READY after %ss\n' "${SECONDS}"
readback() {
  while IFS=$'\t' read -r name host port _; do
    curl --fail --silent --show-error --max-time 15 "http://${host}:${port}/server_info" >"${ROOT}/results/server-info-${name}.json"
    python3 "${ROOT}/scripts/check-basic-server-info.py" "${ROOT}/source/firmus-framework.json" "${ROOT}/results/server-info-${name}.json" "${name%%-*}"
  done <"${ROOT}/source/worker-endpoints.tsv"
}
readback
curl --fail --silent --show-error --max-time 120 -H 'Content-Type: application/json' -d '{"model":"qwen","messages":[{"role":"user","content":"Name the capital of France and explain its historical importance in two sentences."}],"max_tokens":64,"temperature":0}' "http://${HEAD_IP}:8000/v1/chat/completions" >"${ROOT}/results/sanity-chat.json"
status=0
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${ROOT}/sglang.sqsh" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${ROOT}/logs/aiperf.log" 2>&1 || status=$?
readback
python3 "${ROOT}/scripts/collect-eagle-acceptance.py" --config "${ROOT}/source/firmus-framework.json" --endpoints "${ROOT}/source/worker-endpoints.tsv" --server-info-dir "${ROOT}/results" --log-dir "${ROOT}/logs" --output "${ROOT}/results/eagle-acceptance.json" --cluster firmus
(( status == 0 )) || exit "${status}"
test -s "${ROOT}/results/score.json"
echo 'AIPERF ok'
touch "${ROOT}/run-success"
