#!/usr/bin/env bash
set -euo pipefail

: "${AIPERF_MODE:?AIPerf mode is required}"
: "${BENCHMARK_DURATION:?benchmark duration is required}"
: "${BENCHMARK_GRACE_PERIOD:?benchmark grace period is required}"
: "${CONCURRENCY:?concurrency is required}"
: "${CONTAINER_IMAGE:?container image is required}"
: "${HOST:?gateway is required}"
: "${RUNTIME_DIR:?runtime directory is required}"
: "${WARMUP_DURATION:?warmup duration is required}"
aiperf_tmp="$(mktemp -d "${TMPDIR:-/tmp}/qwen-aiperf.XXXXXXXXXX")"
child_pid=
cleanup() { status=$?; trap - EXIT INT TERM; if [[ -n "${child_pid}" ]]; then kill "${child_pid}" 2>/dev/null || true; wait "${child_pid}" 2>/dev/null || true; fi; rm -rf -- "${aiperf_tmp:?}"; exit "${status}"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export ENROOT_CACHE_PATH="${aiperf_tmp}/cache" ENROOT_DATA_PATH="${aiperf_tmp}/data" ENROOT_RUNTIME_PATH="${aiperf_tmp}/runtime" ENROOT_TEMP_PATH="${aiperf_tmp}/temp"
install -d -m 0700 "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}"
enroot start -e NVIDIA_VISIBLE_DEVICES=none -e HOME=/tmp -e HF_HOME=/hf-cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e "AIPERF_MODE=${AIPERF_MODE}" -e "BENCHMARK_DURATION=${BENCHMARK_DURATION}" -e "BENCHMARK_GRACE_PERIOD=${BENCHMARK_GRACE_PERIOD}" -e "CONCURRENCY=${CONCURRENCY}" -e "HOST=${HOST}" -e RESULTS_DIR=/results -e "WARMUP_DURATION=${WARMUP_DURATION}" -m "${RUNTIME_DIR}/results:/results" -m "${RUNTIME_DIR}/scripts:/runtime-scripts" -m "${RUNTIME_DIR}/tokenizer-cache:/hf-cache" "${CONTAINER_IMAGE}" /runtime-scripts/run-aiperf.sh &
child_pid=$!
wait "${child_pid}"
child_pid=
