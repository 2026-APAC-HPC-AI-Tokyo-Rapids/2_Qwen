#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]]
remote_shell=$1
checker=$2
connection_failure_started=-1

while true; do
  if output=$(timeout --foreground 180 "${remote_shell}" --github-shell "${checker}"); then status=0; else status=$?; fi
  printf '%s\n' "${output}"
  if [[ "${status}" -eq 75 ]] || { [[ "${status}" -eq 0 ]] && [[ "${output}" == scheduler_job_pending=* ]]; }; then
    connection_failure_started=-1
    sleep 30
    continue
  fi
  [[ "${status}" -ne 0 ]] || exit 0
  [[ "${status}" -eq 124 || "${status}" -eq 255 ]] || exit "${status}"
  if [[ "${connection_failure_started}" -lt 0 ]]; then connection_failure_started=${SECONDS}; fi
  connection_failure_seconds=$((SECONDS - connection_failure_started))
  [[ "${connection_failure_seconds}" -lt 300 ]] || exit "${status}"
  printf 'scheduler_monitor_connection_failed status=%s retry_in_seconds=15 failure_seconds=%s\n' "${status}" "${connection_failure_seconds}" >&2
  sleep 15
done
