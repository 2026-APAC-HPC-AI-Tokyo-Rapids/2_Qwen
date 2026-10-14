#!/usr/bin/env bash
set -euo pipefail

# Stable profiling hook for compute workloads. The selected profiler can be
# introduced here without changing the scheduler or SGLang launch boundary.
: "${1:?a command is required}"
: "${QWEN_PROFILER_INVOCATION_PATH:?profiler invocation path is required}"
[[ "${QWEN_PROFILER_INVOCATION_PATH}" == /* ]]
install -d -m 0755 "$(dirname -- "${QWEN_PROFILER_INVOCATION_PATH}")"
invocation_tmp="${QWEN_PROFILER_INVOCATION_PATH}.tmp-$$"
trap 'rm -f -- "${invocation_tmp}"' EXIT INT TERM
printf '%q ' "$@" >"${invocation_tmp}"
printf '\n' >>"${invocation_tmp}"
mv -- "${invocation_tmp}" "${QWEN_PROFILER_INVOCATION_PATH}"
if [[ -n "${QWEN_PROFILER_NSYS_BIN:-}" ]]; then
  if [[ "${1##*/}" == vllm ]]; then
    report=${QWEN_PROFILER_INVOCATION_PATH%.argv}
    session=qwen-${report##*/}
    # A TP rank's cudaProfilerStop can close collection before other ranks flush.
    # Stop the whole session in this container, and acknowledge only after import.
    "${QWEN_PROFILER_NSYS_BIN}" profile --session-new="${session}" --force-overwrite=true --trace-fork-before-exec=true --capture-range=cudaProfilerApi --capture-range-end=none --kill=none --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none --output="${report}" "$@" &
    profiler_pid=$!
    while [[ ! -e "${report}.stop-request" ]]; do
      if ! kill -0 "${profiler_pid}" 2>/dev/null; then wait "${profiler_pid}"; exit 1; fi
      sleep 0.1
    done
    stop_status=0
    "${QWEN_PROFILER_NSYS_BIN}" stop --session="${session}" >"${report}.stop.log" 2>&1 || stop_status=$?
    printf '%s\n' "${stop_status}" >"${report}.stop-status.tmp"
    mv "${report}.stop-status.tmp" "${report}.stop-status"
    (( stop_status == 0 )) || exit "${stop_status}"
    wait "${profiler_pid}"
    exit 0
  fi
  exec "${QWEN_PROFILER_NSYS_BIN}" profile --force-overwrite=true --trace-fork-before-exec=true --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none --output="${QWEN_PROFILER_INVOCATION_PATH%.argv}" "$@"
fi
exec "$@"
