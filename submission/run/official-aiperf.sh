#!/usr/bin/env bash
# Run the official AIPerf benchmark against the gateway on this node, then score it.
#
# usage: official-aiperf.sh RUN_DIR CONCURRENCY
#   Writes RUN_DIR/results/: aiperf-command.txt, aiperf-provenance.txt, profile_export_aiperf.json,
#   profile_export.jsonl and score.json.
set -euo pipefail

root=$1
concurrency=$2
cd "${root}/results"

# The tokenizer comes from the offline Hugging Face cache built by prepare.sh.
export HF_HOME=${root}/tokenizer-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# AIPerf 0.12.0 starts its workers and record processors one by one; at c256 their service registration paused for 20 s.
# Give registration time to finish, and use eight record processors instead of 64 on the four-CPU client.
export AIPERF_SERVICE_REGISTRATION_INTERVAL=1 AIPERF_SERVICE_REGISTRATION_MAX_ATTEMPTS=60 AIPERF_SERVICE_REGISTRATION_TIMEOUT=60 AIPERF_RECORD_PROCESSOR_SCALE_FACTOR=32

# The official benchmark command; only the concurrency is chosen by the team.
command=(/opt/sglang/venv/bin/aiperf profile --url http://127.0.0.1:8000 --endpoint-type chat --endpoint /v1/chat/completions --model qwen --tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507 --streaming -H 'Accept: text/event-stream' --concurrency "${concurrency}" --workers-max "${concurrency}" --api-key NOT_USED --warmup-duration 60 --benchmark-duration 120 --benchmark-grace-period 30 --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 --output-tokens-mean 200 --output-tokens-stddev 0 --extra-inputs min_tokens:200 --extra-inputs max_tokens:200 --extra-inputs ignore_eos:true --extra-inputs temperature:0.0 --extra-inputs repetition_penalty:1.0 --num-dataset-entries 12800 --random-seed 2026 --cache-bust first_turn_prefix --goodput 'time_to_first_token:3000 inter_token_latency:20' --ui simple)

# Record the command and the client's versions and settings.
printf '%q ' "${command[@]}" >aiperf-command.txt
printf '\n' >>aiperf-command.txt
aiperf_version=$(/opt/sglang/venv/bin/python -c 'from importlib.metadata import version; print(version("aiperf"))')
printf 'aiperf_version=%s\nconcurrency=%s\ngateway=http://127.0.0.1:8000/v1/chat/completions\ntokenizer_revision=%s\n' "${aiperf_version}" "${concurrency}" "$(cat "${HF_HOME}/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/refs/main")" >aiperf-provenance.txt
env | grep '^AIPERF_' | sort >>aiperf-provenance.txt

"${command[@]}"

export_dir=$(dirname "$(find artifacts -name profile_export_aiperf.json)")
cp "${export_dir}/profile_export_aiperf.json" "${export_dir}/profile_export.jsonl" .
/opt/sglang/venv/bin/python "${root}/scripts/parse-aiperf-result.py" profile_export_aiperf.json score.json
