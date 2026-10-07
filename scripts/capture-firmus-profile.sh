#!/usr/bin/env bash
# One five-second CUDA capture on all six SGLang PD workers after warmup.
set -euo pipefail
runtime_dir=${1:?runtime directory required}
start_requested=false
stop_requested=false
stop_workers() {
  local prefix=$1 status=0 name host port tp pid
  local pids=()
  while IFS=$'\t' read -r name host port tp; do
    curl --fail --silent --show-error --max-time 180 -X POST "http://${host}:${port}/stop_profile" >"${runtime_dir}/profile/${prefix}-${name}.txt" 2>"${runtime_dir}/profile/${prefix}-${name}.stderr" &
    pids+=("$!")
  done <"${runtime_dir}/source/worker-endpoints.tsv"
  for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
  return "${status}"
}
cleanup_capture() {
  status=$?
  trap - EXIT
  if [[ "${start_requested}" == true && "${stop_requested}" == false ]]; then stop_workers cleanup-stop || true; fi
  exit "${status}"
}
trap cleanup_capture EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
deadline=$((SECONDS + 600))
while ! grep -Fq 'Phase profiling (profiling) started' "${runtime_dir}/logs/aiperf.log"; do (( SECONDS < deadline )); sleep 1; done
sleep 5
start_requested=true
pids=()
count=0
while IFS=$'\t' read -r name host port tp; do
  printf 'start_requested worker=%s utc=%s\n' "${name}" "$(date -u +%FT%T.%NZ)"
  curl --fail --silent --show-error --max-time 60 -X POST "http://${host}:${port}/start_profile" -H 'Content-Type: application/json' -d '{"activities":["CUDA_PROFILER"]}' >"${runtime_dir}/profile/start-${name}.txt" 2>"${runtime_dir}/profile/start-${name}.stderr" &
  pids+=("$!")
  count=$((count + 1))
done <"${runtime_dir}/source/worker-endpoints.tsv"
[[ "${count}" -eq 6 ]] || exit 1
start_status=0
for pid in "${pids[@]}"; do wait "${pid}" || start_status=1; done
[[ "${start_status}" -eq 0 ]] || exit 1
printf 'capture_active utc=%s\n' "$(date -u +%FT%T.%NZ)"
sleep 5
stop_requested=true
printf 'stop_requested utc=%s\n' "$(date -u +%FT%T.%NZ)"
stop_workers stop
printf 'capture_stopped utc=%s\n' "$(date -u +%FT%T.%NZ)"
