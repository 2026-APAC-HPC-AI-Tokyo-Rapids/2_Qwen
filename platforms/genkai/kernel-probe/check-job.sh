#!/usr/bin/env bash
set -euo pipefail
: "${PJM_JOB_ID:?}" "${RUNTIME_DIR:?}"
if timeout 20 pjwait -z "${PJM_JOB_ID}"; then
  pjstat -H "${PJM_JOB_ID}" >"${RUNTIME_DIR}/logs/pjm-history.txt"
  cat "${RUNTIME_DIR}/logs/${PJM_JOB_ID}.pjm.log"
  test -e "${RUNTIME_DIR}/run-success"
  test "$(cat "${RUNTIME_DIR}/results/exit-code")" = 0
  trials=3
  if [[ "$(jq -r '.calibration // false' "${RUNTIME_DIR}/source/provenance.json")" == true ]]; then trials=1; fi
  for ((trial=1; trial<=trials; trial++)); do grep -q '^PASS ' "${RUNTIME_DIR}/results/trial-${trial}.log"; done
  exit 0
else
  status=$?
fi
if [[ "${status}" -eq 124 ]]; then printf 'scheduler_job_pending=%s\n' "${PJM_JOB_ID}"; exit 75; fi
exit "${status}"
