#!/usr/bin/env bash
# Run the Qwen 3.5 benchmark inside an already prepared SGLang container.
set -euo pipefail

output_dir="${1:-/output}"
mkdir -p "${output_dir}"

command=(/opt/sglang/venv/bin/python -m sglang.benchmark.offline_throughput
  --model-path /model
  --load-format auto
  --tp-size 1
  --dataset-name sharegpt
  --dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json
  --num-prompts 32
  --sharegpt-output-len 256
  --sharegpt-context-len 262144
  --seed 42
  --mem-fraction-static 0.85
  --result-filename /output/results.jsonl)

printf 'command=' >"${output_dir}/command.txt"
printf '%q ' "${command[@]}" >>"${output_dir}/command.txt"
printf '\n' >>"${output_dir}/command.txt"

exec "${command[@]}"
