#!/usr/bin/env bash
# Capture SGLang through benchmark completion; retain the five-second vLLM window.
set -euo pipefail
runtime_dir=${1:?runtime directory required}
framework=$(jq -er '.framework | select(. == "sglang" or . == "vllm")' "${runtime_dir}/source/firmus-framework.json")
client_log=${2:-${runtime_dir}/logs/aiperf.log}
endpoints="${runtime_dir}/source/worker-endpoints.tsv"
if [[ -s "${runtime_dir}/source/profile-worker-endpoints.tsv" ]]; then endpoints="${runtime_dir}/source/profile-worker-endpoints.tsv"; fi
start_body=
if [[ "${framework}" == sglang ]]; then start_body='{"activities":["CUDA_PROFILER"]}'; fi
start_requested=false
stop_requested=false
stop_workers() {
  local prefix=$1 status=0 name host port tp pid
  local pids=()
  while IFS=$'\t' read -r name host port tp; do
    curl --fail --silent --show-error --max-time 180 -X POST "http://${host}:${port}/stop_profile" >"${runtime_dir}/profile/${prefix}-${name}.txt" 2>"${runtime_dir}/profile/${prefix}-${name}.stderr" &
    pids+=("$!")
  done <"${endpoints}"
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
while ! grep -Fq 'Phase profiling (profiling) started' "${client_log}"; do (( SECONDS < deadline )); sleep 1; done
sleep 5
start_requested=true
pids=()
count=0
while IFS=$'\t' read -r name host port tp; do
  printf 'start_requested worker=%s utc=%s\n' "${name}" "$(date -u +%FT%T.%NZ)"
  curl --fail --silent --show-error --max-time 60 -X POST "http://${host}:${port}/start_profile" -H 'Content-Type: application/json' -d "${start_body}" >"${runtime_dir}/profile/start-${name}.txt" 2>"${runtime_dir}/profile/start-${name}.stderr" &
  pids+=("$!")
  count=$((count + 1))
done <"${endpoints}"
expected_workers=$(awk 'END {print NR}' "${endpoints}")
[[ "${count}" -eq "${expected_workers}" && "${count}" -gt 0 ]] || exit 1
start_status=0
for pid in "${pids[@]}"; do wait "${pid}" || start_status=1; done
[[ "${start_status}" -eq 0 ]] || exit 1
printf 'capture_active utc=%s\n' "$(date -u +%FT%T.%NZ)"
if [[ "${framework}" == sglang ]]; then
  deadline=$((SECONDS + 180))
  while ! grep -Fq 'Phase profiling (profiling) complete |' "${client_log}"; do
    (( SECONDS < deadline )) || exit 1
    sleep 1
  done
else
  sleep 5
fi
stop_requested=true
printf 'stop_requested utc=%s\n' "$(date -u +%FT%T.%NZ)"
stop_workers stop
printf 'capture_stopped utc=%s\n' "$(date -u +%FT%T.%NZ)"
