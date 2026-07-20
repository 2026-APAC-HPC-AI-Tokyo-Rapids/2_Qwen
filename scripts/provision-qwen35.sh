#!/usr/bin/env bash
# Validate the Qwen 3.5 checkpoint and SGLang runtime without loading a model.
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${SIF_ARCHIVE_DIR:?SIF_ARCHIVE_DIR is required}"
: "${GITHUB_RUN_NUMBER:?GITHUB_RUN_NUMBER is required}"

sif_path="${SIF_ARCHIVE_DIR}/${GITHUB_RUN_NUMBER}.sif"

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

module purge
module load singularity-ce/4.1.3

# This reports SGLang, PyTorch and dependency information without binding or
# loading the model.  It intentionally runs without --nv on the login node.
# check_env only catches a non-zero nvidia-smi exit, not an absent executable.
# Supply a temporary stub so it reports unavailable topology instead of failing.
check_env_bin="$(mktemp -d /dev/shm/sglang-check-env.XXXXXXXXXX)"
trap 'rm -rf -- "${check_env_bin:?}"' EXIT
printf '#!/bin/sh\nexit 1\n' > "${check_env_bin}/nvidia-smi"
chmod 0755 "${check_env_bin}/nvidia-smi"
singularity exec --containall --cleanenv \
  --bind "${check_env_bin}:/check-env-bin:ro" \
  --env PATH=/check-env-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  "${sif_path}" /opt/sglang/venv/bin/python -m sglang.check_env
