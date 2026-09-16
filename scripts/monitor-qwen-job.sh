#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]]
remote_shell=$1
checker=$2
connection_failure_started=-1

while true; do
  if timeout --foreground 60 "${remote_shell}" --github-shell "${checker}"; then exit 0; else status=$?; fi
  if [[ "${status}" -eq 75 ]]; then
    connection_failure_started=-1
    sleep 30
    continue
  fi
  [[ "${status}" -eq 124 || "${status}" -eq 255 ]] || exit "${status}"
  if [[ "${connection_failure_started}" -lt 0 ]]; then connection_failure_started=${SECONDS}; fi
  connection_failure_seconds=$((SECONDS - connection_failure_started))
  [[ "${connection_failure_seconds}" -lt 300 ]] || exit "${status}"
  printf 'scheduler_monitor_connection_failed status=%s retry_in_seconds=15 failure_seconds=%s\n' "${status}" "${connection_failure_seconds}" >&2
  sleep 15
done
