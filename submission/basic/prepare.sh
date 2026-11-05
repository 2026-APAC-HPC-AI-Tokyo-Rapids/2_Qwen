#!/usr/bin/env bash
# Prepare a run directory for the Basic job on the Firmus login node (no GPU needed).
#
# usage: prepare.sh RUN_DIR IMAGE MODEL TOKENIZER_SNAPSHOT CONFIG
#   RUN_DIR             new directory on storage both compute nodes can read
#   IMAGE               BUILD.sqsh from scripts/build-basic-container.sh, with BUILD.container-sidecar.json beside it
#   MODEL               Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 at revision e156cb4efae43fbee1a1ab073f946a1377e6b969
#   TOKENIZER_SNAPSHOT  Qwen/Qwen3-235B-A22B-Instruct-2507 tokenizer files at TOKENIZER_REVISION
#   CONFIG              basic/config/firmus-basic.json (or a copy with a new build's image pins)
#
# Then: cd RUN_DIR && sbatch --export=PATH,HOME,USER,LOGNAME scripts/submit.sbatch
# Nothing in SGLang, its kernels or the gateway is patched; only the draft's config is rewritten (see below).
set -euo pipefail

MODEL_REVISION=e156cb4efae43fbee1a1ab073f946a1377e6b969
TOKENIZER_REVISION=ac9c66cc9b46af7306746a9250f23d47083d689e
TOKENIZER_FILES=(config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json)
DRAFT_REPOSITORY=nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507
DRAFT_REVISION=ff0f0c0999f6c86b21dfd2f5803eebc4ac712224

package=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
root=$(realpath -m "$1")
image=$(realpath "$2")
model=$(realpath "$3")
tokenizer=$(realpath "$4")
config=$(realpath "$5")
sidecar=${image%.sqsh}.container-sidecar.json

mkdir "${root}"
install -d -m 0755 "${root}/source" "${root}/scripts" "${root}/logs" "${root}/results" "${root}/home"

# ---- Image and model ----

ln -s "${image}" "${root}/sglang.sqsh"
cp "${sidecar}" "${root}/source/container-sidecar.json"
jq --arg path "${image}" --slurpfile sidecar "${sidecar}" '.images.sglang = {path:$path,sha256:$sidecar[0].container_artifact.sha256,size_bytes:$sidecar[0].container_artifact.size_bytes}' "${config}" >"${root}/source/firmus-framework.json"
jq -n --arg path "${model}" --arg revision "${MODEL_REVISION}" '{model:{path:$path,repository:"Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",revision:$revision}}' >"${root}/source/model.json"

# ---- Tokenizer: an offline Hugging Face cache for AIPerf ----

cache=${root}/tokenizer-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507
install -d -m 0755 "${cache}/refs" "${cache}/snapshots/${TOKENIZER_REVISION}"
for file in "${TOKENIZER_FILES[@]}"; do cp -L "${tokenizer}/${file}" "${cache}/snapshots/${TOKENIZER_REVISION}/${file}"; done
printf '%s' "${TOKENIZER_REVISION}" >"${cache}/refs/main"

# ---- Draft: download the public BF16 weights, then declare the SGLang architecture and dtype in its config ----

export HF_HUB_DISABLE_XET=1
uvx --from huggingface_hub==0.36.0 hf download "${DRAFT_REPOSITORY}" config.json README.md model.safetensors --revision "${DRAFT_REVISION}" --local-dir "${root}/draft" --max-workers 1
# Only the architecture selector and the dtype declaration change; the weights do not.
python3 "${package}/run/prepare_eagle.py" "${root}/draft" "${root}/source"

# ---- Job scripts ----

cp "${package}/basic/submit.sbatch" "${package}/basic/worker.py" "${package}/run/parse-aiperf-result.py" "${root}/scripts/"
# The official benchmark command is unchanged; AIPerf only runs from the image's separate /opt/aiperf environment.
sed 's|/opt/sglang/venv/bin/|/opt/aiperf/bin/|g' "${package}/run/official-aiperf.sh" >"${root}/scripts/official-aiperf.sh"
# Keep the exact preparation scripts with the run's records.
cp "${package}/basic/prepare.sh" "${package}/run/prepare_eagle.py" "${root}/source/"

printf 'cd %q && sbatch --export=PATH,HOME,USER,LOGNAME scripts/submit.sbatch\n' "${root}"
