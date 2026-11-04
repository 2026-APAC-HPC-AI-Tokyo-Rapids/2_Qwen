#!/usr/bin/env bash
# Prepare a run directory for the Basic job on the Firmus login node (no GPU needed).
#
# usage: prepare.sh RUN_DIR IMAGE MODEL TOKENIZER_SNAPSHOT CONFIG
#   RUN_DIR             new directory on storage both compute nodes can read; must not exist
#   IMAGE               BUILD.sqsh from scripts/build-basic-container.sh, with BUILD.container-sidecar.json beside it
#   MODEL               Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 at MODEL_REVISION, all 24 shards
#   TOKENIZER_SNAPSHOT  Qwen/Qwen3-235B-A22B-Instruct-2507 tokenizer files at TOKENIZER_REVISION
#   CONFIG              configs/firmus-basic.json (or a copy with a new build's image pins)
#
# Then: cd RUN_DIR && sbatch --export=PATH,HOME,USER,LOGNAME scripts/submit.sbatch
# Nothing in SGLang, its kernels or the gateway is patched; only the draft's config is rewritten (see below).
set -euo pipefail

SGLANG_COMMIT=297e54fd105e974cdf401e5b4d20253a22c16f90
MODEL_REVISION=e156cb4efae43fbee1a1ab073f946a1377e6b969
TOKENIZER_REVISION=ac9c66cc9b46af7306746a9250f23d47083d689e
TOKENIZER_FILES=(config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json)
DRAFT_REPOSITORY=nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507
DRAFT_REVISION=ff0f0c0999f6c86b21dfd2f5803eebc4ac712224
DRAFT_WEIGHT_SHA256=bf226a966868fe413a22e62e471aa1130bd94f17fd2c40d322b1c8b4def1ae86

[[ $# -eq 5 ]] || { echo 'Usage: prepare.sh RUN_DIR IMAGE MODEL TOKENIZER CONFIG' >&2; exit 64; }
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=$(realpath -m "$1")
image=$(realpath "$2")
model=$(realpath "$3")
tokenizer=$(realpath "$4")
config=$(realpath "$5")
sidecar=${image%.sqsh}.container-sidecar.json

# ---- Checks before anything is created ----

# The image is an unpatched Basic build of SGLANG_COMMIT, and the very artifact the configuration pins.
jq -e --slurpfile config "${config}" --arg sglang "${SGLANG_COMMIT}" '.tier == "basic" and .applied_patches == [] and .source_commits.sglang == $sglang and .source_commits.qwen == $config[0].container_source_commit and .container_artifact.sha256 == $config[0].container_sha256 and .container_artifact.size_bytes == $config[0].container_size_bytes' "${sidecar}" >/dev/null
[[ "$(stat -c '%s' "${image}")" == "$(jq -er .container_artifact.size_bytes "${sidecar}")" ]]
[[ "$(sha256sum "${image}" | cut -d' ' -f1)" == "$(jq -er .container_artifact.sha256 "${sidecar}")" ]]
# The official FP8 checkpoint at the pinned revision, every shard present.
[[ "$(head -n1 "${model}/.cache/huggingface/download/config.json.metadata")" == "${MODEL_REVISION}" ]]
jq -e '.model_type == "qwen3_moe" and .quantization_config.quant_method == "fp8"' "${model}/config.json" >/dev/null
[[ "$(jq -r '[.weight_map[]] | unique | length' "${model}/model.safetensors.index.json")" == 24 ]]
while read -r shard; do test -s "${model}/${shard}"; done < <(jq -r '[.weight_map[]] | unique[]' "${model}/model.safetensors.index.json")

# ---- Run directory; removed again if preparation fails or is interrupted ----

mkdir "${root}"
draft_pid=
cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "${draft_pid}" ]]; then kill -TERM -- "-${draft_pid}" 2>/dev/null || true; wait "${draft_pid}" 2>/dev/null || true; fi
  if (( status != 0 )); then rm -rf -- "${root}"; fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
install -d -m 0755 "${root}/source" "${root}/scripts" "${root}/logs" "${root}/results" "${root}/profile" "${root}/home"
ln -s "${image}" "${root}/sglang.sqsh"
cp "${sidecar}" "${root}/source/container-sidecar.json"
jq --arg path "${image}" --slurpfile sidecar "${sidecar}" '.images.sglang = {path:$path,sha256:$sidecar[0].container_artifact.sha256,size_bytes:$sidecar[0].container_artifact.size_bytes}' "${config}" >"${root}/source/firmus-framework.json"
jq -n --arg path "${model}" --arg revision "${MODEL_REVISION}" '{model:{path:$path,repository:"Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",revision:$revision}}' >"${root}/source/model.json"

# ---- Tokenizer: an offline Hugging Face cache for AIPerf ----

cache=${root}/tokenizer-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507
install -d -m 0755 "${cache}/refs" "${cache}/snapshots/${TOKENIZER_REVISION}"
for file in "${TOKENIZER_FILES[@]}"; do cp -L "${tokenizer}/${file}" "${cache}/snapshots/${TOKENIZER_REVISION}/${file}"; done
printf '%s' "${TOKENIZER_REVISION}" >"${cache}/refs/main"

# ---- Draft: download in its own process group (so an interrupt stops it), then check the BF16 weights ----

export HF_HUB_DISABLE_XET=1
setsid uvx --from huggingface_hub==0.36.0 hf download "${DRAFT_REPOSITORY}" config.json README.md model.safetensors --revision "${DRAFT_REVISION}" --local-dir "${root}/draft" --max-workers 1 &
draft_pid=$!
wait "${draft_pid}"
draft_pid=
# Only the architecture selector and the dtype declaration in the draft's config change; the weights do not.
python3 "${repo}/submission/run/prepare_eagle.py" "${root}/draft" "${root}/source"
jq -e --arg sha "${DRAFT_WEIGHT_SHA256}" '.weight_sha256 == $sha and .precision == "BF16"' "${root}/source/draft-provenance.json" >/dev/null

# ---- Job scripts ----

cp "${repo}/submission/basic/submit.sbatch" "${repo}/scripts/basic-worker.py" "${repo}/scripts/check-basic-server-info.py" "${repo}/scripts/collect-eagle-acceptance.py" "${repo}/submission/run/run-aiperf.py" "${repo}/submission/run/parse-aiperf-result.py" "${root}/scripts/"
# The official benchmark command is unchanged; AIPerf only runs from the image's separate /opt/aiperf environment.
sed 's|/opt/sglang/venv/bin/|/opt/aiperf/bin/|g' "${repo}/submission/run/official-aiperf.sh" >"${root}/scripts/official-aiperf.sh"
# Keep the exact preparation scripts with the run's records.
cp "${repo}/submission/basic/prepare.sh" "${repo}/submission/run/prepare_eagle.py" "${root}/source/"

printf 'cd %q && sbatch --export=PATH,HOME,USER,LOGNAME scripts/submit.sbatch\n' "${root}"
