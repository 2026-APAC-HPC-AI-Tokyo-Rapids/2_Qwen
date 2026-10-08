#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]]
remote_shell=$1
checker=$2
connection_failure_started=-1
last_observed_state=unknown
transport_stderr=$(mktemp)
trap 'rm -f -- "${transport_stderr}"' EXIT

while true; do
  if output=$(timeout --foreground 180 "${remote_shell}" --github-shell "${checker}" 2>"${transport_stderr}"); then status=0; else status=$?; fi
  cat "${transport_stderr}" >&2
  # The shell adapter can report tar exit 2 after an SSH stream is cut off.
  if [[ "${status}" -eq 2 ]] && grep -Eq '^Connection closed by .* port [0-9]+$' "${transport_stderr}"; then status=255; fi
  printf '%s\n' "${output}"
  if [[ "${status}" -eq 75 ]] || { [[ "${status}" -eq 0 ]] && [[ "${output}" == scheduler_job_pending=* ]]; } || { [[ "${status}" -eq 2 ]] && [[ "${output}" =~ ^scheduler_job_pending=[[:alnum:]._-]+[[:space:]]state=[A-Z]+$ ]]; }; then
    if [[ "${output}" =~ state=([A-Z]+) ]]; then last_observed_state=${BASH_REMATCH[1]}; fi
    connection_failure_started=-1
    sleep 30
    continue
  fi
  [[ "${status}" -ne 0 ]] || exit 0
  [[ "${status}" -eq 124 || "${status}" -eq 255 ]] || exit "${status}"
  if [[ "${connection_failure_started}" -lt 0 ]]; then connection_failure_started=${SECONDS}; fi
  connection_failure_seconds=$((SECONDS - connection_failure_started))
  # Lost SSH observations do not establish scheduler failure; wait for a definitive result.
  printf 'scheduler_monitor_connection_failed status=%s retry_in_seconds=15 failure_seconds=%s last_state=%s limit_seconds=%s\n' "${status}" "${connection_failure_seconds}" "${last_observed_state}" none >&2
  sleep 15
done
