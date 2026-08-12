#!/usr/bin/env bash
set -euo pipefail

: "${CHECK_DIR:?}"
: "${MODEL_PATH:?}"
: "${DATASET_PATH:?}"
: "${DATASET_SHA256:?}"
: "${OMPI_COMM_WORLD_RANK:?}"

arm=${1:?arm is required}
trial=${2:?trial is required}
[[ "${arm}" == off || "${arm}" == on ]]
[[ "${trial}" =~ ^[0-9]+$ ]]
node_rank="${OMPI_COMM_WORLD_RANK}"
[[ "${OMPI_COMM_WORLD_SIZE}" -eq 2 ]]

arm_dir="${CHECK_DIR}/trial-${trial}-${arm}"
control_dir="${arm_dir}/control"
node_dir="$(mktemp -d "/dev/shm/qwen-overhead-${arm}.XXXXXXXXXX")"
jit_cache="$(mktemp -d "/tmp/qwen-overhead-jit-${arm}.XXXXXXXXXX")"
singularity_tmp="$(mktemp -d "/dev/shm/qwen-overhead-singularity-${arm}.XXXXXXXXXX")"
export SINGULARITY_TMPDIR="${singularity_tmp}"
metrics_pid=""
engine_pid=""

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${engine_pid}" ]]; then
    kill -- "-${engine_pid}" 2>/dev/null || true
    wait "${engine_pid}" 2>/dev/null || true
  fi
  if [[ -n "${metrics_pid}" ]]; then
    kill -TERM "${metrics_pid}" 2>/dev/null || true
    wait "${metrics_pid}" 2>/dev/null || true
  fi
  rm -rf -- "${node_dir:?}" "${jit_cache:?}" "${singularity_tmp:?}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

finish_metrics() {
  if [[ -z "${metrics_pid}" ]]; then
    return 0
  fi
  local started ended
  started="$(date +%s%N)"
  kill -TERM "${metrics_pid}" 2>/dev/null || true
  set +e
  wait "${metrics_pid}"
  local status=$?
  set -e
  ended="$(date +%s%N)"
  awk -v delta="$((ended - started))" \
    'BEGIN { printf "%.9f\n", delta / 1000000000 }' \
    >"${arm_dir}/producer-finalize-wall-sec-node-${node_rank}"
  metrics_pid=""
  return "${status}"
}

if [[ -d "${HOME}/.cache/sglang-jit" ]]; then
  rsync -a -- "${HOME}/.cache/sglang-jit/" "${jit_cache}/"
fi

master_file="${control_dir}/master-host.txt"
completion_file="${control_dir}/rank-zero-status"
if [[ "${node_rank}" -eq 0 ]]; then
  rm -f -- "${master_file}" "${completion_file}"
  hostname >"${master_file}"
else
  while [[ ! -s "${master_file}" ]]; do sleep 0.1; done
fi
master_host="$(<"${master_file}")"

printf -v run_id 'r-%032x' "$((trial + 2))"
if [[ "${arm}" == on ]]; then
  socket_path="${node_dir}/events.sock"
  ready_file="${node_dir}/events.ready"
  event_summary="${arm_dir}/event-summary-node-${node_rank}.json"
  metrics_cli="${CHECK_DIR}/qwen-experiment-metrics/.venv/bin/qwen-experiment-metrics"
  manifest="${arm_dir}/control/run-manifest.json"
  test -s "${manifest}"
  "${metrics_cli}" serve-events \
    --manifest "${manifest}" --runtime-dir "${arm_dir}" \
    --socket "${socket_path}" --ready-file "${ready_file}" \
    --node-rank "${node_rank}" &
  metrics_pid=$!
  while [[ ! -e "${ready_file}" ]]; do
    kill -0 "${metrics_pid}"
    sleep 0.1
  done
fi

benchmark=(/opt/sglang/venv/bin/python -m sglang.benchmark.offline_throughput
  --model-path /model
  --load-format auto
  --tp-size 8
  --nnodes 2
  --node-rank "${node_rank}"
  --dist-init-addr "${master_host}:25000"
  --dataset-name sharegpt
  --dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json
  --num-prompts 32
  --sharegpt-output-len 256
  --sharegpt-context-len 262144
  --seed 42
  --random-seed 42
  --mem-fraction-static 0.85)

metrics_env=(
  --env QWEN_EXPERIMENT_METRICS_RUN_ID="${run_id}"
  --env QWEN_EXPERIMENT_METRICS_NODE_RANK="${node_rank}"
  --env QWEN_EXPERIMENT_METRICS_SOURCE_DIR=/source-output
  --env QWEN_EXPERIMENT_METRICS_DATASET_MANIFEST_SHA256="${DATASET_SHA256}"
)
metrics_bind=()
if [[ "${arm}" == on ]]; then
  blob_oids="${CHECK_DIR}/control/instrumentation-source-blob-oids.json"
  bench_args_blob_oid="$(jq -er '.bench_args | select(test("^[0-9a-f]{40,64}$"))' "${blob_oids}")"
  server_args_blob_oid="$(jq -er '.server_args | select(test("^[0-9a-f]{40,64}$"))' "${blob_oids}")"
  metrics_env+=(
    --env QWEN_EXPERIMENT_METRICS_SOCKET=/metrics/events.sock
    --env QWEN_EXPERIMENT_METRICS_BENCH_ARGS_BLOB_OID="${bench_args_blob_oid}"
    --env QWEN_EXPERIMENT_METRICS_SERVER_ARGS_BLOB_OID="${server_args_blob_oid}"
  )
  metrics_bind=(--bind "${node_dir}:/metrics")
fi

command=(singularity exec --userns --nv --cleanenv --containall
  --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_DATASETS_OFFLINE=1
  --env SGLANG_DG_CACHE_DIR=/jit-cache/deep-gemm,TORCHINDUCTOR_CACHE_DIR=/jit-cache/torchinductor,TRITON_CACHE_DIR=/jit-cache/triton
  "${metrics_env[@]}"
  "${metrics_bind[@]}"
  --bind "${MODEL_PATH}:/model:ro,${DATASET_PATH}:/dataset/ShareGPT_V3_unfiltered_cleaned_split.json:ro,${jit_cache}:/jit-cache,${jit_cache}:/root/.cache,${arm_dir}/source:/source-output"
  "${CHECK_DIR}/sglang.sif" "${benchmark[@]}")

if [[ "${node_rank}" -eq 0 ]]; then
  set +e
  "${command[@]}"
  status=$?
  set -e
  printf '%s\n' "${status}" >"${completion_file}"
  metrics_status=0
  finish_metrics || metrics_status=$?
  if [[ "${status}" -eq 0 && "${metrics_status}" -ne 0 ]]; then
    status="${metrics_status}"
  fi
  if [[ "${status}" -eq 0 && "${arm}" == on ]]; then
    "${CHECK_DIR}/scripts/v2-overhead-event-summary.py" \
      "${arm_dir}/source/events/node-${node_rank}.jsonl" \
      "${event_summary}" "${run_id}" "${node_rank}"
  fi
  printf '%s\n' "${status}" >"${completion_file}"
  exit "${status}"
fi

setsid "${command[@]}" \
  >"${CHECK_DIR}/logs/trial-${trial}-${arm}.node-rank-1.log" 2>&1 &
engine_pid=$!
while kill -0 "${engine_pid}" 2>/dev/null; do
  if [[ -s "${completion_file}" ]]; then
    status="$(<"${completion_file}")"
    kill -TERM -- "-${engine_pid}" 2>/dev/null || true
    wait "${engine_pid}" 2>/dev/null || true
    engine_pid=""
    metrics_status=0
    finish_metrics || metrics_status=$?
    if [[ "${status}" -eq 0 && "${metrics_status}" -ne 0 ]]; then
      status="${metrics_status}"
    fi
    if [[ "${status}" -eq 0 && "${arm}" == on ]]; then
      "${CHECK_DIR}/scripts/v2-overhead-event-summary.py" \
        "${arm_dir}/source/events/node-${node_rank}.jsonl" \
        "${event_summary}" "${run_id}" "${node_rank}"
    fi
    exit "${status}"
  fi
  sleep 1
done
set +e
wait "${engine_pid}"
status=$?
set -e
engine_pid=""
# The coordinator can tear down the distributed participant just before it
# persists the authoritative rank-zero status.  Preserve genuine participant
# failures, but give the coordinator a bounded window to classify a normal
# coordinated shutdown before returning the child's signal-derived status.
for _ in {1..60}; do
  [[ ! -s "${completion_file}" ]] || break
  sleep 1
done
if [[ -s "${completion_file}" ]]; then
  status="$(<"${completion_file}")"
fi
metrics_status=0
finish_metrics || metrics_status=$?
if [[ "${status}" -eq 0 && "${metrics_status}" -ne 0 ]]; then
  status="${metrics_status}"
fi
if [[ "${status}" -eq 0 && "${arm}" == on ]]; then
  "${CHECK_DIR}/scripts/v2-overhead-event-summary.py" \
    "${arm_dir}/source/events/node-${node_rank}.jsonl" \
    "${event_summary}" "${run_id}" "${node_rank}"
fi
exit "${status}"
