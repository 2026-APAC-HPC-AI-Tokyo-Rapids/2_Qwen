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
exec "$@"
