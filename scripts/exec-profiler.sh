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
  gpu_metrics=()
  # Counter sampling follows the CUDA capture range, after the normal score.
  # Each Slurm worker owns distinct GPUs; never collect another worker's devices.
  if grep -Eq '^RmProfilingAdminOnly:[[:space:]]*0[[:space:]]*$' /proc/driver/nvidia/params 2>/dev/null; then
    gpu_metrics=(--gpu-metrics-devices=cuda-visible --gpu-metrics-frequency=1000)
    printf 'requested: CUDA-visible GPUs at 1000 Hz during the capture range; verify samples in the report\n' >"${QWEN_PROFILER_INVOCATION_PATH%.argv}.gpu-metrics.txt"
  else
    printf 'skipped: unrestricted GPU performance counter access is not advertised by the driver; CUDA/NVTX trace remains enabled\n' >"${QWEN_PROFILER_INVOCATION_PATH%.argv}.gpu-metrics.txt"
  fi
  exec "${QWEN_PROFILER_NSYS_BIN}" profile --force-overwrite=true --trace-fork-before-exec=true --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none "${gpu_metrics[@]}" --output="${QWEN_PROFILER_INVOCATION_PATH%.argv}" "$@"
fi
exec "$@"
