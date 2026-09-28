#!/usr/bin/env bash
set -euo pipefail

# In-container profiling entrypoint. When PROFILE=true, submit.sbatch runs this
# script as the container command with Nsight Systems bind-mounted read-only at
# /opt/nsys, so the profiler executes inside the container without changing the
# image or the SGLang launch boundary. Collection is bounded by the
# cudaProfilerApi capture range: /start_profile and /stop_profile (activities
# ["CUDA_PROFILER"]) open and close the capture window, so only the requested
# seconds of steady-state execution are recorded.
: "${1:?a command is required}"
: "${QWEN_PROFILER_OUTPUT_DIR:?profiler output directory is required}"
: "${QWEN_PROFILER_LABEL:?profiler label is required}"
nsys_bin="${QWEN_PROFILER_NSYS_BIN:-/opt/nsys/bin/nsys}"

install -d -m 0755 "${QWEN_PROFILER_OUTPUT_DIR}"
invocation="${QWEN_PROFILER_OUTPUT_DIR}/profiler-invocation-${QWEN_PROFILER_LABEL}.argv"
invocation_tmp="${invocation}.tmp-$$"
trap 'rm -f -- "${invocation_tmp}"' EXIT INT TERM
printf '%q ' "$@" >"${invocation_tmp}"
printf '\n' >>"${invocation_tmp}"
mv -- "${invocation_tmp}" "${invocation}"

exec "${nsys_bin}" profile --force-overwrite=true --trace-fork-before-exec=true --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none --output="${QWEN_PROFILER_OUTPUT_DIR}/nsys-${QWEN_PROFILER_LABEL}" "$@"
