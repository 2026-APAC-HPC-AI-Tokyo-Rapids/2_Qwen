#!/usr/bin/env bash
set -euo pipefail

: "${AIPERF_MODE:?AIPerf mode is required}"
: "${CONCURRENCY:?concurrency is required}"
: "${HOST:?gateway host and port are required}"
: "${RESULTS_DIR:?results directory is required}"

[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]
[[ "${AIPERF_MODE}" == official || "${AIPERF_MODE}" == exploratory ]]
MODEL=qwen
cd "${RESULTS_DIR}"

common=(--url "http://${HOST}" --endpoint-type chat --endpoint /v1/chat/completions --model "${MODEL}" --tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507 --streaming -H 'Accept: text/event-stream' --concurrency "${CONCURRENCY}" --workers-max "${CONCURRENCY}" --api-key NOT_USED --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 --output-tokens-mean 200 --output-tokens-stddev 0 --extra-inputs min_tokens:200 --extra-inputs max_tokens:200 --extra-inputs ignore_eos:true --extra-inputs temperature:0.0 --extra-inputs repetition_penalty:1.0 --num-dataset-entries 12800 --random-seed 2026 --cache-bust first_turn_prefix --goodput "time_to_first_token:3000 inter_token_latency:20" --ui simple)
if [[ "${AIPERF_MODE}" == official ]]; then timing=(--warmup-duration 60 --benchmark-duration 120 --benchmark-grace-period 30); else : "${WARMUP_DURATION:?exploratory warmup duration is required}"; : "${BENCHMARK_DURATION:?exploratory benchmark duration is required}"; : "${BENCHMARK_GRACE_PERIOD:?exploratory grace period is required}"; timing=(--warmup-duration "${WARMUP_DURATION}" --benchmark-duration "${BENCHMARK_DURATION}" --benchmark-grace-period "${BENCHMARK_GRACE_PERIOD}"); fi
command=(/opt/sglang/venv/bin/aiperf profile "${common[@]:0:19}" "${timing[@]}" "${common[@]:19}")
printf '%q ' "${command[@]}" >aiperf-command.txt
printf '\n' >>aiperf-command.txt
printf 'aiperf_version=%s\nmode=%s\nconcurrency=%s\ngateway=http://%s/v1/chat/completions\n' "$(/opt/sglang/venv/bin/python -c 'from importlib.metadata import version; print(version("aiperf"))')" "${AIPERF_MODE}" "${CONCURRENCY}" "${HOST}" >aiperf-provenance.txt
"${command[@]}"
export_file="$(find artifacts -type f -name profile_export_aiperf.json -print -quit)"
[[ -n "${export_file}" && -s "${export_file}" ]]
cp "${export_file}" profile_export_aiperf.json
