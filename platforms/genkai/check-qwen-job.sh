#!/usr/bin/env bash
set -euo pipefail

: "${PJM_JOB_ID:?PJM job ID is required}"
: "${RUNTIME_DIR:?runtime directory is required}"

if timeout 20 pjwait -z "${PJM_JOB_ID}"; then
  pjwait "${PJM_JOB_ID}" | tee "${RUNTIME_DIR}/logs/pjm-accounting.txt"
  cat "${RUNTIME_DIR}/logs/${PJM_JOB_ID}.pjm.log"
  test -e "${RUNTIME_DIR}/run-success"
  test -s "${RUNTIME_DIR}/results/profile_export_aiperf.json"
  test -s "${RUNTIME_DIR}/results/score.json"
  if [[ -s "${RUNTIME_DIR}/source/genkai-serving.json" && "$(jq -r '.speculation.algorithm' "${RUNTIME_DIR}/source/genkai-serving.json")" == EAGLE3 ]]; then test -s "${RUNTIME_DIR}/results/eagle-acceptance.json"; fi
  exit 0
else
  status=$?
fi
if [[ "${status}" -eq 124 ]]; then printf 'scheduler_job_pending=%s\n' "${PJM_JOB_ID}"; exit 75; fi
exit "${status}"
