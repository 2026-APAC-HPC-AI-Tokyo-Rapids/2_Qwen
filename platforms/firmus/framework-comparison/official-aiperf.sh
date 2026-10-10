#!/usr/bin/env bash
set -euo pipefail
ROOT=$1
N=$2
mode=${3:-official}
timings=(--warmup-duration 60 --benchmark-duration 120 --benchmark-grace-period 30)
output=${ROOT}/results
if [[ "${mode}" == profile ]]; then
  output=${ROOT}/profile/client
  timings=(--warmup-duration 20 --benchmark-duration 30 --benchmark-grace-period 90)
else
  [[ "${mode}" == official ]]
fi
mkdir -p "${output}"
cd "${output}"
export HF_HOME=${ROOT}/tokenizer-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# AIPerf 0.12.0 synchronously forks N workers and record processors.
# At c256 registration paused for 20s. Allow startup to finish and use
# eight record processors instead of 64 for the four-CPU client.
export AIPERF_SERVICE_REGISTRATION_INTERVAL=1 AIPERF_SERVICE_REGISTRATION_MAX_ATTEMPTS=60 AIPERF_SERVICE_REGISTRATION_TIMEOUT=60
export AIPERF_RECORD_PROCESSOR_SCALE_FACTOR=32
command=(/opt/sglang/venv/bin/aiperf profile --url http://127.0.0.1:8000 --endpoint-type chat --endpoint /v1/chat/completions --model qwen --tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507 --streaming -H 'Accept: text/event-stream' --concurrency "${N}" --workers-max "${N}" --api-key NOT_USED "${timings[@]}" --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 --output-tokens-mean 200 --output-tokens-stddev 0 --extra-inputs min_tokens:200 --extra-inputs max_tokens:200 --extra-inputs ignore_eos:true --extra-inputs temperature:0.0 --extra-inputs repetition_penalty:1.0 --num-dataset-entries 12800 --random-seed 2026 --cache-bust first_turn_prefix --goodput 'time_to_first_token:3000 inter_token_latency:20' --ui simple)
printf '%q ' "${command[@]}" >aiperf-command.txt
printf '\n' >>aiperf-command.txt
printf 'aiperf_version=%s\nmode=%s\nconcurrency=%s\ngateway=http://127.0.0.1:8000/v1/chat/completions\ntokenizer_revision=%s\n' "$(/opt/sglang/venv/bin/python -c 'from importlib.metadata import version; print(version("aiperf"))')" "${mode}" "${N}" "$(cat "${HF_HOME}/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/refs/main")" >aiperf-provenance.txt
printf 'registration_interval=%s\nregistration_max_attempts=%s\nregistration_timeout=%s\n' "${AIPERF_SERVICE_REGISTRATION_INTERVAL}" "${AIPERF_SERVICE_REGISTRATION_MAX_ATTEMPTS}" "${AIPERF_SERVICE_REGISTRATION_TIMEOUT}" >>aiperf-provenance.txt
printf 'record_processor_scale_factor=%s\n' "${AIPERF_RECORD_PROCESSOR_SCALE_FACTOR}" >>aiperf-provenance.txt
/opt/sglang/venv/bin/python "${ROOT}/scripts/run-aiperf.py" "${command[@]}"
export_file=$(find artifacts -mindepth 2 -maxdepth 2 -type f -name profile_export_aiperf.json -print -quit)
test -n "${export_file}"
cp "${export_file}" profile_export_aiperf.json
cp "$(dirname "${export_file}")/profile_export.jsonl" profile_export.jsonl
if [[ "${mode}" == profile ]]; then exit 0; fi
/opt/sglang/venv/bin/python "${ROOT}/scripts/parse-aiperf-result.py" profile_export_aiperf.json score.json --concurrency "${N}" --require-eligible --command aiperf-command.txt --provenance aiperf-provenance.txt
