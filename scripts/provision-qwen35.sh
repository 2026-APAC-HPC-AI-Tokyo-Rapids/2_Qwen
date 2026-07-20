#!/usr/bin/env bash
# Validate the Qwen 3.5 checkpoint without loading a model.
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH is required}"

config_path="${MODEL_PATH}/config.json"
test -s "${config_path}"
jq -e '
  (.model_type | strings | startswith("qwen3_5"))
  and ((.architectures // []) | any(.[]?; type == "string" and startswith("Qwen3_5")))
' "${config_path}" >/dev/null

# Compare the local checkpoint with the current upstream revision.  Do not set
# HF_HUB_OFFLINE here: this check intentionally queries the Hugging Face Hub.
uvx --from 'huggingface_hub>=1.0' hf cache verify \
  Qwen/Qwen3.5-35B-A3B-FP8 \
  --local-dir "${MODEL_PATH}" \
  --fail-on-missing-files
