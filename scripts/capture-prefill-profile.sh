#!/usr/bin/env bash
# One bounded capture on the two ASPIRE prefill workers, after AIPerf warmup.
set -euo pipefail
runtime_dir=${1:?runtime directory required}
start_requested=false
stop_requested=false
cleanup_capture() {
  status=$?
  trap - EXIT
  if [[ "${start_requested}" == true && "${stop_requested}" == false ]]; then
    cleanup_pids=()
    cleanup_index=0
    while IFS=$'\t' read -r role host port _; do
      [[ "${role}" == prefill ]] || continue
      curl --fail --silent --show-error --max-time 180 -X POST "http://${host}:${port}/stop_profile" >"${runtime_dir}/profile/cleanup-stop-prefill-${cleanup_index}.txt" 2>"${runtime_dir}/profile/cleanup-stop-prefill-${cleanup_index}.stderr" &
      cleanup_pids+=("$!")
      cleanup_index=$((cleanup_index + 1))
    done <"${runtime_dir}/source/worker-endpoints.tsv"
    for pid in "${cleanup_pids[@]}"; do wait "${pid}" || true; done
  fi
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
index=0
while IFS=$'\t' read -r role host port _; do
  [[ "${role}" == prefill ]] || continue
  printf 'start_requested worker=%s utc=%s\n' "${index}" "$(date -u +%FT%T.%NZ)"
  curl --fail --silent --show-error --max-time 60 -X POST "http://${host}:${port}/start_profile" -H 'Content-Type: application/json' -d "{\"activities\":[\"CPU\",\"CUDA_PROFILER\"],\"with_stack\":false,\"record_shapes\":false,\"profile_id\":\"prefill-${index}\",\"output_dir\":\"/profiler-output/torch-prefill-${index}\"}" >"${runtime_dir}/profile/start-prefill-${index}.txt" &
  pids+=("$!")
  index=$((index + 1))
done <"${runtime_dir}/source/worker-endpoints.tsv"
[[ "${index}" -eq 2 ]]
for pid in "${pids[@]}"; do wait "${pid}"; done
printf 'capture_active utc=%s\n' "$(date -u +%FT%T.%NZ)"
sleep 5
stop_requested=true
pids=()
index=0
while IFS=$'\t' read -r role host port _; do
  [[ "${role}" == prefill ]] || continue
  printf 'stop_requested worker=%s utc=%s\n' "${index}" "$(date -u +%FT%T.%NZ)"
  curl --fail --silent --show-error --max-time 180 -X POST "http://${host}:${port}/stop_profile" >"${runtime_dir}/profile/stop-prefill-${index}.txt" &
  pids+=("$!")
  index=$((index + 1))
done <"${runtime_dir}/source/worker-endpoints.tsv"
for pid in "${pids[@]}"; do wait "${pid}"; done
printf 'capture_stopped utc=%s\n' "$(date -u +%FT%T.%NZ)"
