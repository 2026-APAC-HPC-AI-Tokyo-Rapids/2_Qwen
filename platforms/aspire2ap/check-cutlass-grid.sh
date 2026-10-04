#!/usr/bin/env bash
set -euo pipefail
: "${PBS_JOB_ID:?PBS job ID required}"
: "${RUNTIME_DIR:?runtime directory required}"
/opt/pbs/bin/qstat -x -f "${PBS_JOB_ID}" >"${RUNTIME_DIR}/logs/pbs-accounting.txt"
state="$(awk '/^[[:space:]]*job_state =/ {print $3}' "${RUNTIME_DIR}/logs/pbs-accounting.txt")"
if [[ "${state}" != F ]]; then printf 'scheduler_job_pending=%s state=%s\n' "${PBS_JOB_ID}" "${state}"; else
  cat "${RUNTIME_DIR}/logs/pbs-accounting.txt"
  test "$(awk -F= '/^[[:space:]]*Exit_status =/{gsub(/[[:space:]]/, "", $2); print $2}' "${RUNTIME_DIR}/logs/pbs-accounting.txt")" = 0
  test -e "${RUNTIME_DIR}/run-success"
  test -s "${RUNTIME_DIR}/results/moe-routing.json"
  test ! -e "${RUNTIME_DIR}/results/score.json"
fi
