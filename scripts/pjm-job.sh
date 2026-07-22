#!/usr/bin/env bash
# Monitor, cancel, and clean up a PJM job with its scheduler log.
set -euo pipefail

job_is_active() {
  pjstat "$1" | grep -q "^$1[[:space:]]"
}

cancel_and_wait() {
  local job_id="$1"
  if job_is_active "${job_id}"; then
    echo "Cancelling PJM job ${job_id}" >&2
    pjdel "${job_id}" || true
  fi
  while job_is_active "${job_id}"; do
    sleep 1
  done
}

wait_for_log() {
  local job_id="$1"
  local log_file="$2"
  trap 'cancel_and_wait "${job_id}"; exit 143' INT TERM
  while [[ ! -e "${log_file}" ]]; do
    if status="$(pjstat "${job_id}" 2>&1)" && ! grep -q "^${job_id}[[:space:]]" <<<"${status}"; then
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
  tail -n +1 -F "${log_file}" &
  local tail_pid=$!
  stop_log_stream() {
    kill "${tail_pid}" 2>/dev/null || true
    wait "${tail_pid}" 2>/dev/null || true
  }
  trap stop_log_stream EXIT
  trap 'cancel_and_wait "${job_id}"; exit 143' INT TERM
  while ! { status="$(pjstat "${job_id}" 2>&1)" &&
             ! grep -q "^${job_id}[[:space:]]" <<<"${status}"; }; do
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
