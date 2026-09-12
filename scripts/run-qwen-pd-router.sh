#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_IMAGE:?container image is required}"
: "${HEAD_HOST:?head node is required}"
: "${MODEL_PATH:?model path is required}"
: "${RUNTIME_DIR:?runtime directory is required}"
: "${TAIL_HOST:?tail node is required}"
router_tmp="$(mktemp -d "${TMPDIR:-/tmp}/qwen-router.XXXXXXXXXX")"
child_pid=
cleanup() { status=$?; trap - EXIT INT TERM; if [[ -n "${child_pid}" ]]; then kill "${child_pid}" 2>/dev/null || true; wait "${child_pid}" 2>/dev/null || true; fi; rm -rf -- "${router_tmp:?}"; exit "${status}"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export ENROOT_CACHE_PATH="${router_tmp}/cache" ENROOT_DATA_PATH="${router_tmp}/data" ENROOT_RUNTIME_PATH="${router_tmp}/runtime" ENROOT_TEMP_PATH="${router_tmp}/temp"
install -d -m 0700 "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}"
enroot start -e NVIDIA_VISIBLE_DEVICES=none -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -m "${MODEL_PATH}:/model" "${CONTAINER_IMAGE}" /opt/sglang/venv/bin/python -m sglang_router.launch_router --pd-disaggregation --prefill "http://${HEAD_HOST}:30100" 9000 --prefill "http://${HEAD_HOST}:30101" 9001 --prefill "http://${HEAD_HOST}:30102" 9002 --prefill "http://${HEAD_HOST}:30103" 9003 --decode "http://${TAIL_HOST}:30200" --decode "http://${TAIL_HOST}:30201" --host 0.0.0.0 --port 8000 &
child_pid=$!
wait "${child_pid}"
child_pid=
