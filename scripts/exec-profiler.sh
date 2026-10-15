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
  exec "${QWEN_PROFILER_NSYS_BIN}" profile --force-overwrite=true --trace-fork-before-exec=true --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx,osrt --cuda-graph-trace=node --cuda-event-trace=false --sample=none --cpuctxsw=none --output="${QWEN_PROFILER_INVOCATION_PATH%.argv}" "$@"
fi
exec "$@"
