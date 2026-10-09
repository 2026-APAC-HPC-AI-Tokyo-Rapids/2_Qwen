#!/bin/bash
#SBATCH -J vllm-matched-comparison
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:8
#SBATCH --propagate=NONE
#SBATCH -o runs/vllm-%j.out
#SBATCH -e runs/vllm-%j.err
#
# 2026-10-09 profiler-free vLLM repeat precedes SGLang to reverse the capture order.
# Actions copy of matched-settings-comparison; decoder controls are matched to SGLang. See README.md.
#
# 235B 8+8: 4x Prefill workers at TP=2, EP=1 + 2x Decode workers at TP=4, EP=4 = 16 GPUs.
# Packing: node0 has all 4 prefill workers; node1 has both decode workers. Pass = aiperf
# exits 0. This is the topology walked through in
# 4_1_Qwen3_235B_PD_Application_Notes_Firmus.md.
#
# Copied from reference-scripts/07-pd-235b-8plus8.sbatch.
# This adaptation uses our images and pinned tokenizer, concurrency 8,
# and the reference warmup/benchmark/grace timings (20s/60s/15s).
# Actions changes only run/image/output paths, supplies concurrency from committed
# configuration, and validates raw requests after measurement. Serving flags unchanged.
# Launch, network selection, readiness and cleanup follow the reference.
# Worker commands and gateway are adapted to vLLM/NIXL; the client image stays shared.
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR}
SGLANG=${ROOT}/vllm.sqsh
AIPERF=${ROOT}/sglang.sqsh
MODEL=/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8
R_PORT=8000
READY_SECS=600  # /health cap; typically 2-4 min, fail if still down at 10 min
ISL=4000
OSL=200
N=$(jq -er .concurrency "${ROOT}/source/firmus-framework.json")
ART=${ROOT}/logs
PROFILE=${PROFILE:-false}
[[ "${PROFILE}" == true || "${PROFILE}" == false ]]

NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD=${NODES[0]}
TAIL=${NODES[1]}
ip_of() { getent ahostsv4 "$1" | awk '{print $1; exit}'; }
HEAD_IP=$(ip_of "${HEAD}")
TAIL_IP=$(ip_of "${TAIL}")

mounts() {
  mkdir -p "${ROOT}/home/$1"
  local profiler_mount=
  if [[ "${PROFILE}" == true ]]; then profiler_mount=",${ROOT}/nsys:/opt/nsys:ro"; fi
  echo "/scratch:/scratch,${ROOT}/home/$1:${HOME},${ROOT}:${ROOT},/dev/infiniband:/dev/infiniband${profiler_mount}"
}

wait_http() {
  timeout "${READY_SECS}" curl -sf --retry-connrefused --retry-all-errors --retry-delay 2 --retry 300 --max-time 2 -o /dev/null "$1"
}

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} user=$(whoami)"
echo "head=${HEAD}(${HEAD_IP}) tail=${TAIL}(${TAIL_IP}) model=${MODEL} isl=${ISL} osl=${OSL} N=${N}"
test -r "${SGLANG}" && test -r "${AIPERF}" && test -f "${MODEL}/config.json"
mkdir -p "${ART}"
worker_pids=()
profile_capture_pid=
cleanup_all() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${profile_capture_pid}" ]]; then kill "${profile_capture_pid}" 2>/dev/null || true; wait "${profile_capture_pid}" 2>/dev/null || true; fi
  for pid in "${worker_pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${worker_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup_all EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "${PROFILE}" == true ]]; then
  "${ROOT}/nsys/bin/nsys" --version >"${ROOT}/profile/nsys-version.txt"
  printf 'Capture: identical ten-second CPU-runtime/CUDA/NVTX window after a separate normal reference pass and second-pass warmup. results/ is the normal pass; profile/client/ is diagnostic only.\n' >"${ROOT}/profile/interpretation.txt"
  for worker in 0 1 2 3; do printf 'prefill-%s\t%s\t%s\t2\n' "${worker}" "${HEAD_IP}" "$((30100 + worker))"; done >"${ROOT}/source/worker-endpoints.tsv"
  for worker in 0 1; do printf 'decode-%s\t%s\t%s\t4\n' "${worker}" "${TAIL_IP}" "$((30200 + worker))"; done >>"${ROOT}/source/worker-endpoints.tsv"
fi

cat >"${ROOT}/source/worker.sh" <<'EOF'
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
EOF

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:2 --cpus-per-task=16 --container-image="${SGLANG}" --container-mounts="$(mounts prefill-0)" bash "${ROOT}/source/worker.sh" prefill-0 30100 "${MODEL}" 2 >"${ART}/prefill-0.log" 2>&1 &
worker_pids+=("$!")

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:2 --cpus-per-task=16 --container-image="${SGLANG}" --container-mounts="$(mounts prefill-1)" bash "${ROOT}/source/worker.sh" prefill-1 30101 "${MODEL}" 2 >"${ART}/prefill-1.log" 2>&1 &
worker_pids+=("$!")

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:2 --cpus-per-task=16 --container-image="${SGLANG}" --container-mounts="$(mounts prefill-2)" bash "${ROOT}/source/worker.sh" prefill-2 30102 "${MODEL}" 2 >"${ART}/prefill-2.log" 2>&1 &
worker_pids+=("$!")

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:2 --cpus-per-task=16 --container-image="${SGLANG}" --container-mounts="$(mounts prefill-3)" bash "${ROOT}/source/worker.sh" prefill-3 30103 "${MODEL}" 2 >"${ART}/prefill-3.log" 2>&1 &
worker_pids+=("$!")

srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-image="${SGLANG}" --container-mounts="$(mounts decode-0)" bash "${ROOT}/source/worker.sh" decode-0 30200 "${MODEL}" 4 --enable-expert-parallel --max-cudagraph-capture-size 64 --max-num-seqs 256 --no-enable-flashinfer-autotune >"${ART}/decode-0.log" 2>&1 &
worker_pids+=("$!")

srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-image="${SGLANG}" --container-mounts="$(mounts decode-1)" bash "${ROOT}/source/worker.sh" decode-1 30201 "${MODEL}" 4 --enable-expert-parallel --max-cudagraph-capture-size 64 --max-num-seqs 256 --no-enable-flashinfer-autotune >"${ART}/decode-1.log" 2>&1 &
worker_pids+=("$!")

wait_http "http://${HEAD_IP}:30100/health"; echo "prefill-0 ready"
wait_http "http://${HEAD_IP}:30101/health"; echo "prefill-1 ready"
wait_http "http://${HEAD_IP}:30102/health"; echo "prefill-2 ready"
wait_http "http://${HEAD_IP}:30103/health"; echo "prefill-3 ready"
wait_http "http://${TAIL_IP}:30200/health"; echo "decode-0 ready"
wait_http "http://${TAIL_IP}:30201/health"; echo "decode-1 ready"

srun --overlap -N1 -n1 -w "${HEAD}" --cpus-per-task=2 --container-image="${SGLANG}" --container-mounts="$(mounts router)" /usr/bin/python3 "${ROOT}/vllm-proxy.py" --host 0.0.0.0 --port "${R_PORT}" --prefiller-hosts "${HEAD_IP}" "${HEAD_IP}" "${HEAD_IP}" "${HEAD_IP}" --prefiller-ports 30100 30101 30102 30103 --decoder-hosts "${TAIL_IP}" "${TAIL_IP}" --decoder-ports 30200 30201 >"${ART}/router.log" 2>&1 &
worker_pids+=("$!")

wait_http "http://${HEAD_IP}:${R_PORT}/v1/models"
echo "READY after ${SECONDS}s"


srun --overlap -N1 -n1 -w "${HEAD}" --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" /opt/sglang/venv/bin/aiperf profile --model qwen --tokenizer "/scratch/home/ACCOUNT/qwen/framework-comparison/sources/tokenizer/ac9c66cc9b46af7306746a9250f23d47083d689e" --url "http://127.0.0.1:${R_PORT}" --endpoint-type chat --endpoint /v1/chat/completions --streaming --concurrency "${N}" --warmup-duration 20 --benchmark-duration 60 --benchmark-grace-period 15 --synthetic-input-tokens-mean "${ISL}" --synthetic-input-tokens-stddev 0 --output-tokens-mean "${OSL}" --output-tokens-stddev 0 --num-dataset-entries 12800 --random-seed 2026 --cache-bust first_turn_prefix --extra-inputs "min_tokens:${OSL}" --extra-inputs "max_tokens:${OSL}" --extra-inputs "ignore_eos:true" --extra-inputs "temperature:0.0" --extra-inputs "repetition_penalty:1.0" -H "Accept: text/event-stream" --artifact-dir "${ROOT}/results" >"${ART}/aiperf.log" 2>&1
echo "AIPERF ok"
srun --overlap -N1 -n1 -w "${HEAD}" --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" /opt/sglang/venv/bin/python "${ROOT}/scripts/parse-firmus-reference-result.py" "${ROOT}/results" --concurrency "${N}"
if [[ "${PROFILE}" == true ]]; then
  mkdir -p "${ROOT}/profile/client"
  "${ROOT}/scripts/capture-firmus-profile.sh" "${ROOT}" >"${ART}/profile-capture.log" 2>&1 &
  profile_capture_pid=$!
  srun --overlap -N1 -n1 -w "${HEAD}" --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" /opt/sglang/venv/bin/aiperf profile --model qwen --tokenizer "/scratch/home/ACCOUNT/qwen/framework-comparison/sources/tokenizer/ac9c66cc9b46af7306746a9250f23d47083d689e" --url "http://127.0.0.1:${R_PORT}" --endpoint-type chat --endpoint /v1/chat/completions --streaming --concurrency "${N}" --warmup-duration 20 --benchmark-duration 60 --benchmark-grace-period 15 --synthetic-input-tokens-mean "${ISL}" --synthetic-input-tokens-stddev 0 --output-tokens-mean "${OSL}" --output-tokens-stddev 0 --num-dataset-entries 12800 --random-seed 2026 --cache-bust first_turn_prefix --extra-inputs "min_tokens:${OSL}" --extra-inputs "max_tokens:${OSL}" --extra-inputs "ignore_eos:true" --extra-inputs "temperature:0.0" --extra-inputs "repetition_penalty:1.0" -H "Accept: text/event-stream" --artifact-dir "${ROOT}/profile/client" >"${ROOT}/profile/client/aiperf.log" 2>&1
  wait "${profile_capture_pid}"
  profile_capture_pid=
  deadline=$((SECONDS + 180))
  while [[ "$(find "${ROOT}/profile" -maxdepth 1 -name '*.nsys-rep' -size +0c | wc -l)" -ne 6 ]]; do (( SECONDS < deadline )); sleep 2; done
  srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" /opt/sglang/venv/bin/python "${ROOT}/scripts/validate-firmus-profile.py" "${ROOT}" >"${ART}/profile-validation.log" 2>&1
fi
touch "${ROOT}/run-success"
