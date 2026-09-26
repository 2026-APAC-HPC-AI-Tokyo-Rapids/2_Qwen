#!/usr/bin/env bash
set -euo pipefail
: "${PBS_JOB_ID:?PBS job ID required}"
: "${RUNTIME_DIR:?runtime directory required}"
/opt/pbs/bin/qstat -x -f "${PBS_JOB_ID}" >"${RUNTIME_DIR}/logs/pbs-accounting.txt"
state="$(awk '/^[[:space:]]*job_state =/ {print $3}' "${RUNTIME_DIR}/logs/pbs-accounting.txt")"
# The runner starts a login shell with SHLVL unset; explicit exit executes .bash_logout.
# Finish naturally so a successful query retains its status.
if [[ "${state}" != F ]]; then
  printf 'scheduler_job_pending=%s state=%s\n' "${PBS_JOB_ID}" "${state}"
else
  cat "${RUNTIME_DIR}/logs/pbs-accounting.txt"
  cat "${RUNTIME_DIR}/logs/pbs.log"
  grep -Eq '^[[:space:]]*Exit_status = 0$' "${RUNTIME_DIR}/logs/pbs-accounting.txt"
  test -e "${RUNTIME_DIR}/run-success"
  [[ "$(find "${RUNTIME_DIR}/results" -maxdepth 1 -name 'rank-*.json' | wc -l)" -eq 16 ]]
fi
