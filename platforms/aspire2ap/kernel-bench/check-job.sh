#!/usr/bin/env bash
set -euo pipefail
: "${PBS_JOB_ID:?}" "${RUNTIME_DIR:?}"
/opt/pbs/bin/qstat -x -f "${PBS_JOB_ID}" >"${RUNTIME_DIR}/logs/pbs-accounting.txt"
state=$(awk '/^[[:space:]]*job_state =/ {print $3}' "${RUNTIME_DIR}/logs/pbs-accounting.txt")
if [[ "${state}" != F ]]; then
  printf 'scheduler_job_pending=%s state=%s\n' "${PBS_JOB_ID}" "${state}"
  exit 75
fi
cat "${RUNTIME_DIR}/logs/pbs-accounting.txt"
if [[ -f "${RUNTIME_DIR}/logs/driver.log" ]]; then cat "${RUNTIME_DIR}/logs/driver.log"; fi
exit_status=$(awk -F= '/^[[:space:]]*Exit_status =/ {gsub(/[[:space:]]/, "", $2); print $2}' "${RUNTIME_DIR}/logs/pbs-accounting.txt")
test "${exit_status}" = 0
test -e "${RUNTIME_DIR}/run-success"
test "$(cat "${RUNTIME_DIR}/results/exit-code")" = 0
test -s "${RUNTIME_DIR}/results/steps.tsv"
test -s "${RUNTIME_DIR}/results/image/bench.jsonl"
# Candidate correctness and eligibility are judged by the unchanged summary.
