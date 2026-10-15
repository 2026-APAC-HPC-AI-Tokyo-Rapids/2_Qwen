#!/usr/bin/env bash
#SBATCH -J run
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:8
#SBATCH --propagate=NONE
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR}
ART=${ROOT}/logs
IMAGE=${ROOT}/dynamo.sqsh
CLIENT=${ROOT}/sglang.sqsh
MODEL=/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8
framework=$(jq -er .framework "${ROOT}/source/firmus-framework.json")
NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD=${NODES[0]}
TAIL=${NODES[1]}
HEAD_IP=$(getent ahostsv4 "${HEAD}" | awk '{print $1; exit}')
TAIL_IP=$(getent ahostsv4 "${TAIL}" | awk '{print $1; exit}')
export ETCD_ENDPOINTS=http://${HEAD_IP}:23900
export DYN_DISCOVERY_BACKEND=etcd
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_NAMESPACE=qwen-${SLURM_JOB_ID}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
pids=()
cleanup() {
  status=$?
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mounts() { mkdir -p "${ROOT}/home/$1"; printf '%s\n' "/scratch:/scratch,${ROOT}/home/$1:${HOME},${ROOT}:${ROOT},/dev/infiniband:/dev/infiniband"; }
wait_http() { timeout 900 curl --fail --silent --show-error --retry-connrefused --retry-all-errors --retry-delay 2 --retry 450 --max-time 2 -o /dev/null "$1"; }
printf 'job=%s framework=%s head=%s tail=%s namespace=%s\n' "${SLURM_JOB_ID}" "${framework}" "${HEAD_IP}" "${TAIL_IP}" "${DYN_NAMESPACE}"
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=1 --container-image="${IMAGE}" --container-mounts="$(mounts etcd)" etcd --name qwen --data-dir "${ROOT}/etcd" --listen-client-urls "${ETCD_ENDPOINTS}" --advertise-client-urls "${ETCD_ENDPOINTS}" --listen-peer-urls http://127.0.0.1:23901 --initial-advertise-peer-urls http://127.0.0.1:23901 --initial-cluster qwen=http://127.0.0.1:23901 >"${ART}/etcd.log" 2>&1 &
pids+=("$!")
wait_http "${ETCD_ENDPOINTS}/health"
for worker in 0 1 2 3; do
  name=prefill-${worker}
  port=$((30100 + worker))
  printf '%s\t%s\t%s\t2\n' "${name}" "${HEAD_IP}" "${port}" >>"${ROOT}/source/worker-endpoints.tsv"
  srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:2 --cpus-per-task=16 --container-image="${IMAGE}" --container-mounts="$(mounts "${name}")" bash "${ROOT}/scripts/worker.sh" "${framework}" "${name}" "${port}" "${MODEL}" 2 >"${ART}/${name}.log" 2>&1 &
  pids+=("$!")
done
for worker in 0 1; do
  name=decode-${worker}
  port=$((30200 + worker))
  printf '%s\t%s\t%s\t4\n' "${name}" "${TAIL_IP}" "${port}" >>"${ROOT}/source/worker-endpoints.tsv"
  srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-image="${IMAGE}" --container-mounts="$(mounts "${name}")" bash "${ROOT}/scripts/worker.sh" "${framework}" "${name}" "${port}" "${MODEL}" 4 >"${ART}/${name}.log" 2>&1 &
  pids+=("$!")
done
while IFS=$'\t' read -r name host port tp; do wait_http "http://${host}:${port}/health"; printf '%s ready\n' "${name}"; done <"${ROOT}/source/worker-endpoints.tsv"
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 --container-image="${IMAGE}" --container-mounts="$(mounts frontend)" env DYN_TCP_RPC_HOST="${HEAD_IP}" DYN_TCP_RESPONSE_STREAM_HOST="${HEAD_IP}" python3 -m dynamo.frontend --http-port 8000 --http-host 0.0.0.0 >"${ART}/frontend.log" 2>&1 &
pids+=("$!")
wait_http "http://${HEAD_IP}:8000/v1/models"
curl --fail --silent --show-error --max-time 180 -H 'Content-Type: application/json' -d '{"model":"qwen","messages":[{"role":"user","content":"Hello"}],"max_tokens":1,"temperature":0}' "http://${HEAD_IP}:8000/v1/chat/completions" >"${ART}/smoke.json"
jq -e '.choices | length > 0' "${ART}/smoke.json" >/dev/null
curl --fail --silent --show-error "http://${HEAD_IP}:8000/v1/models" >"${ART}/models.json"
jq -e '[.data[].id] | index("qwen") != null' "${ART}/models.json" >/dev/null
echo "READY after ${SECONDS}s"
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=4 --container-image="${CLIENT}" --container-mounts="$(mounts aiperf)" /opt/sglang/venv/bin/aiperf profile --model qwen --tokenizer /scratch/home/ACCOUNT/qwen/framework-comparison/sources/tokenizer/ac9c66cc9b46af7306746a9250f23d47083d689e --url "http://${HEAD_IP}:8000" --endpoint-type chat --endpoint /v1/chat/completions --streaming --concurrency 8 --warmup-duration 20 --benchmark-duration 60 --benchmark-grace-period 15 --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 --output-tokens-mean 200 --output-tokens-stddev 0 --num-dataset-entries 12800 --random-seed 2026 --cache-bust first_turn_prefix --extra-inputs min_tokens:200 --extra-inputs max_tokens:200 --extra-inputs ignore_eos:true --extra-inputs temperature:0.0 --extra-inputs repetition_penalty:1.0 -H 'Accept: text/event-stream' --artifact-dir "${ROOT}/results" >"${ART}/aiperf.log" 2>&1
srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 --container-image="${CLIENT}" --container-mounts="$(mounts aiperf)" /opt/sglang/venv/bin/python "${ROOT}/scripts/parse-firmus-reference-result.py" "${ROOT}/results" --concurrency 8
while IFS=$'\t' read -r name host port tp; do curl --fail --silent --show-error --max-time 10 "http://${host}:${port}/metrics" >"${ART}/${name}-metrics.txt"; done <"${ROOT}/source/worker-endpoints.tsv"
touch "${ROOT}/run-success"
