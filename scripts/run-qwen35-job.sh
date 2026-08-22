#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_IMAGE:?Enroot image is required}"
: "${GPUS_PER_NODE:?GPU count per node is required}"
: "${MODEL_PATH:?model metadata directory is required}"
: "${RUNTIME_DIR:?runtime directory is required}"

export ENROOT_SYSCONF_PATH="${RUNTIME_DIR}/enroot-system"
test -x "${ENROOT_SYSCONF_PATH}/hooks.d/98-nvidia.sh"
cd "${RUNTIME_DIR}"
metrics_cli="${RUNTIME_DIR}/qwen-experiment-metrics/.venv/bin/qwen-experiment-metrics"
manifest="${RUNTIME_DIR}/control/run-manifest.json"
execution_settings="${RUNTIME_DIR}/control/execution-settings.json"
nodes="$(jq -er '.nodes' "${execution_settings}")"
run_purpose="$(jq -er '.run_purpose' "${manifest}")"
mkdir -p logs results source profile/raw

"${metrics_cli}" start-run-lifecycle --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}"
finish_run_lifecycle() {
  status=$?
  trap - EXIT
  set +e
  "${metrics_cli}" finish-run-lifecycle --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" --exit-code "${status}"
  lifecycle_status=$?
  set -e
  if [[ "${lifecycle_status}" -ne 0 ]]; then exit "${lifecycle_status}"; fi
  exit "${status}"
}
trap finish_run_lifecycle EXIT

"${metrics_cli}" capture-runtime-launchers --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}"
"${metrics_cli}" record-environment --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" --scope runtime

{
  printf 'job_id=%s\n' "${PJM_JOBID:-${PBS_JOBID:-unknown}}"
  printf 'started_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'container_image=%s\n' "${CONTAINER_IMAGE}"
  printf 'load_format=dummy\n'
  printf 'enroot_version=%s\n' "$(enroot version)"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
} >logs/environment.log 2>&1

benchmark_command=(mpirun -n "${nodes}" --map-by ppr:1:node --bind-to none "${RUNTIME_DIR}/scripts/run-qwen35-node.sh")
set +e
"${metrics_cli}" run-stage --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" --name benchmark --cwd "${RUNTIME_DIR}" --stderr-mode stdout --stdout-path "${RUNTIME_DIR}/logs/benchmark.log" -- "${benchmark_command[@]}"
benchmark_status=$?
set -e
printf 'exit_code=%d\n' "${benchmark_status}" >>logs/benchmark.log
printf 'finished_at=%s\n' "$(date -u +%FT%TZ)" >>logs/environment.log

if [[ "${benchmark_status}" -eq 0 ]]; then
  test -s source/primary-result.json
  jq -e '.error_count == 0 and ((.generation_wall_sec | tonumber) > 0)' source/primary-result.json >/dev/null
  if [[ "${run_purpose}" == profile ]]; then "${metrics_cli}" record-profile-results --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" --expected-ranks "$(jq -er '.tensor_parallel_size' "${execution_settings}")"; fi
  touch run-success
fi
exit "${benchmark_status}"
