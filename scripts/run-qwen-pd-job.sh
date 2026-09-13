#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_IMAGE:?container image is required}"
: "${MODEL_PATH:?model path is required}"
: "${PROFILE:?profile selection is required}"
: "${RUNTIME_DIR:?runtime directory is required}"

config="${RUNTIME_DIR}/source/run-config.json"
mode="$(jq -er '.mode | select(. == "official" or . == "exploratory")' "${config}")"
concurrency="$(jq -er '.concurrency | select(type == "number" and . >= 1 and floor == .)' "${config}")"
warmup="$(jq -er '.exploratory_timing.warmup_duration_seconds' "${config}")"
duration="$(jq -er '.exploratory_timing.benchmark_duration_seconds' "${config}")"
grace="$(jq -er '.exploratory_timing.benchmark_grace_period_seconds' "${config}")"
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ "${#nodes[@]}" -eq 2 ]]
head_host="${nodes[0]}"
tail_host="${nodes[1]}"
head_ip="$(getent ahostsv4 "${head_host}" | awk '$2 == "STREAM" { print $1; exit }')"
tail_ip="$(getent ahostsv4 "${tail_host}" | awk '$2 == "STREAM" { print $1; exit }')"
[[ "${head_ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]
[[ "${tail_ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]
server_step_pid=
router_step_pid=

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${router_step_pid}" ]]; then kill "${router_step_pid}" 2>/dev/null || true; wait "${router_step_pid}" 2>/dev/null || true; fi
  if [[ -n "${server_step_pid}" ]]; then kill "${server_step_pid}" 2>/dev/null || true; wait "${server_step_pid}" 2>/dev/null || true; fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'job_id=%s\nhead_host=%s\nhead_ip=%s\ntail_host=%s\ntail_ip=%s\nmode=%s\nconcurrency=%s\nprofile=%s\nstarted_at=%s\n' "${SLURM_JOB_ID}" "${head_host}" "${head_ip}" "${tail_host}" "${tail_ip}" "${mode}" "${concurrency}" "${PROFILE}" "$(date -u +%FT%TZ)" >"${RUNTIME_DIR}/logs/job-environment.log"
srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 --distribution=block:block "${RUNTIME_DIR}/scripts/run-qwen-pd-node.sh" >"${RUNTIME_DIR}/logs/server-step.log" 2>&1 &
server_step_pid=$!

worker_urls=("http://${head_ip}:30100/health" "http://${head_ip}:30101/health" "http://${head_ip}:30102/health" "http://${head_ip}:30103/health" "http://${tail_ip}:30200/health" "http://${tail_ip}:30201/health")
deadline="$((SECONDS + 1200))"
while true; do
  kill -0 "${server_step_pid}"
  ready=0
  for url in "${worker_urls[@]}"; do if curl --fail --silent --max-time 2 "${url}" >/dev/null; then ready="$((ready + 1))"; fi; done
  if [[ "${ready}" -eq 6 ]]; then break; fi
  (( SECONDS < deadline ))
  sleep 5
done

srun --overlap --nodes=1 --ntasks=1 --nodelist="${head_host}" env HEAD_IP="${head_ip}" TAIL_IP="${tail_ip}" "${RUNTIME_DIR}/scripts/run-qwen-pd-router.sh" >"${RUNTIME_DIR}/logs/router.log" 2>&1 &
router_step_pid=$!
deadline="$((SECONDS + 300))"
models_url="http://${head_ip}:8000/v1/models"
models_pending="${RUNTIME_DIR}/results/models.pending.json"
while true; do
  kill -0 "${router_step_pid}"
  if curl --fail --silent --max-time 2 "${models_url}" >"${models_pending}" && jq -e '.data | map(.id) | index("qwen") != null' "${models_pending}" >/dev/null; then mv "${models_pending}" "${RUNTIME_DIR}/results/models.json"; break; fi
  (( SECONDS < deadline ))
  sleep 2
done

prewarm_request="${RUNTIME_DIR}/logs/nixl-prewarm-request.json"
prewarm_response="${RUNTIME_DIR}/logs/nixl-prewarm-response.json"
prewarm_timing="${RUNTIME_DIR}/logs/nixl-prewarm-timing.txt"
jq -n --arg content "nixl-prewarm-${SLURM_JOB_ID}" '{model:"qwen",messages:[{role:"user",content:$content}],stream:false,min_tokens:2,max_tokens:2,ignore_eos:true,temperature:0.0}' >"${prewarm_request}"
prewarm_started="$(date +%s)"
printf 'started_at=%s\n' "$(date -u +%FT%TZ)" >"${prewarm_timing}"
curl --fail --silent --show-error --max-time 180 -H 'Content-Type: application/json' --data-binary "@${prewarm_request}" "http://${head_ip}:8000/v1/chat/completions" >"${prewarm_response}"
jq -e '.choices | length > 0' "${prewarm_response}" >/dev/null
printf 'finished_at=%s\nelapsed_seconds=%s\n' "$(date -u +%FT%TZ)" "$(( $(date +%s) - prewarm_started ))" >>"${prewarm_timing}"

srun --overlap --nodes=1 --ntasks=1 --nodelist="${head_host}" env AIPERF_MODE="${mode}" BENCHMARK_DURATION="${duration}" BENCHMARK_GRACE_PERIOD="${grace}" CONCURRENCY="${concurrency}" HOST=127.0.0.1:8000 WARMUP_DURATION="${warmup}" "${RUNTIME_DIR}/scripts/run-aiperf-container.sh" >"${RUNTIME_DIR}/logs/aiperf.log" 2>&1
parser_args=()
if [[ "${mode}" == official ]]; then parser_args+=(--require-eligible --command "${RUNTIME_DIR}/results/aiperf-command.txt" --provenance "${RUNTIME_DIR}/results/aiperf-provenance.txt"); fi
"${RUNTIME_DIR}/scripts/parse-aiperf-result.py" "${RUNTIME_DIR}/results/profile_export_aiperf.json" "${RUNTIME_DIR}/results/score.json" --concurrency "${concurrency}" "${parser_args[@]}"
for url in "${worker_urls[@]}"; do metrics_url="${url%/health}/metrics"; filename="$(sed 's%http://%%; s%[:/]%-%g' <<<"${metrics_url}")"; curl --fail --silent --max-time 10 "${metrics_url}" >"${RUNTIME_DIR}/results/${filename}.prom" || true; done
printf 'finished_at=%s\n' "$(date -u +%FT%TZ)" >>"${RUNTIME_DIR}/logs/job-environment.log"
touch "${RUNTIME_DIR}/run-success"
