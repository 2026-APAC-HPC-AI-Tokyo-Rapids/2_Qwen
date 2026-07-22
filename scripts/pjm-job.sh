#!/usr/bin/env bash
# Monitor, cancel, and clean up a PJM job with its scheduler log.
set -euo pipefail

job_state() {
  local job_id="$1"
  local output status=0
  output="$(pjstat "${job_id}" 2>&1)" || status=$?

  if [[ "${status}" -ne 0 || "${output}" == *"PJM daemon is not present"* ]]; then
    echo "PJM status query failed for job ${job_id}; retrying: ${output}" >&2
    return 2
  fi
  if grep -q "^${job_id}[[:space:]]" <<< "${output}"; then
    return 0
  fi
  return 1
}

cancel_and_wait() {
  local job_id="$1"
  local state
  local cancel_requested=false
  while :; do
    if job_state "${job_id}"; then
      state=0
    else
      state=$?
    fi
    if [[ "${state}" -eq 1 ]]; then
      return
    fi
    if [[ "${cancel_requested}" == false ]]; then
      echo "Cancelling PJM job ${job_id}" >&2
      pjdel "${job_id}" || true
      cancel_requested=true
    fi
    sleep 1
  done
}

wait_for_log() {
  local job_id="$1"
  local log_file="$2"
  local state
  trap 'cancel_and_wait "${job_id}"; exit 143' INT TERM
  while [[ ! -e "${log_file}" ]]; do
    if job_state "${job_id}"; then
      :
    else
      state=$?
    fi
    if [[ "${state:-0}" -eq 1 ]]; then
      echo "PJM job ${job_id} did not create a scheduler log" >&2
      exit 1
    fi
    sleep 1
  done
  trap - INT TERM
}

monitor() {
  local job_id="$1"
  local log_file="$2"
  local success_marker="$3"
  local state
  tail -n +1 -F "${log_file}" &
  local tail_pid=$!
  stop_log_stream() {
    kill "${tail_pid}" 2>/dev/null || true
    wait "${tail_pid}" 2>/dev/null || true
  }
  trap stop_log_stream EXIT
  trap 'cancel_and_wait "${job_id}"; exit 143' INT TERM
  while :; do
    if job_state "${job_id}"; then
      sleep 10
      continue
    fi
    state=$?
    if [[ "${state}" -eq 1 ]]; then
      break
    fi
    sleep 10
  done
  sleep 1
  stop_log_stream
  trap - EXIT INT TERM
  test -e "${success_marker}"
}

cleanup() {
  local job_id="$1"
  shift
  if [[ -n "${job_id}" ]]; then
    cancel_and_wait "${job_id}"
  fi
  local target
  for target in "$@"; do
    if [[ -z "${target}" || "${target}" == / ]]; then
      echo "Refusing to remove an unsafe path" >&2
      exit 2
    fi
    rm -rf -- "${target}"
  done
}

case "${1:-}" in
  wait-for-log) wait_for_log "${2:?job id is required}" "${3:?log file is required}" ;;
  monitor) monitor "${2:?job id is required}" "${3:?log file is required}" "${4:?success marker is required}" ;;
  cleanup) cleanup "${2:-}" "${@:3}" ;;
  *)
    echo "usage: $0 {wait-for-log JOB_ID LOG_FILE|monitor JOB_ID LOG_FILE SUCCESS_MARKER|cleanup [JOB_ID] PATH...}" >&2
    exit 2
    ;;
esac
