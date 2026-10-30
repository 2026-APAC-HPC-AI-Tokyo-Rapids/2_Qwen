#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 5 ]] || { echo 'Usage: prepare.sh RUN_DIR IMAGE MODEL TOKENIZER CONFIG' >&2; exit 64; }
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=$(realpath -m "$1")
image=$(realpath "$2")
model=$(realpath "$3")
tokenizer=$(realpath "$4")
config=$(realpath "$5")
sidecar=${image%.sqsh}.container-sidecar.json
jq -e --slurpfile config "${config}" '.tier == "basic" and .applied_patches == [] and .source_commits.sglang == "297e54fd105e974cdf401e5b4d20253a22c16f90" and .source_commits.qwen == $config[0].container_source_commit and .container_artifact.sha256 == $config[0].container_sha256 and .container_artifact.size_bytes == $config[0].container_size_bytes' "${sidecar}" >/dev/null
[[ "$(stat -c '%s' "${image}")" == "$(jq -er .container_artifact.size_bytes "${sidecar}")" ]]
[[ "$(sha256sum "${image}" | cut -d' ' -f1)" == "$(jq -er .container_artifact.sha256 "${sidecar}")" ]]
[[ "$(head -n1 "${model}/.cache/huggingface/download/config.json.metadata")" == e156cb4efae43fbee1a1ab073f946a1377e6b969 ]]
jq -e '.model_type == "qwen3_moe" and .quantization_config.quant_method == "fp8"' "${model}/config.json" >/dev/null
[[ "$(jq -r '[.weight_map[]] | unique | length' "${model}/model.safetensors.index.json")" == 24 ]]
while read -r shard; do test -s "${model}/${shard}"; done < <(jq -r '[.weight_map[]] | unique[]' "${model}/model.safetensors.index.json")
mkdir "${root}"
trap 'status=$?; if (( status != 0 )); then rm -rf -- "${root}"; fi' EXIT
install -d -m 0755 "${root}/source" "${root}/scripts" "${root}/logs" "${root}/results" "${root}/profile" "${root}/home"
ln -s "${image}" "${root}/sglang.sqsh"
cp "${sidecar}" "${root}/source/container-sidecar.json"
jq --arg path "${image}" --slurpfile sidecar "${sidecar}" '.images.sglang = {path:$path,sha256:$sidecar[0].container_artifact.sha256,size_bytes:$sidecar[0].container_artifact.size_bytes}' "${config}" >"${root}/source/firmus-framework.json"
jq -n --arg path "${model}" '{model:{path:$path,repository:"Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",revision:"e156cb4efae43fbee1a1ab073f946a1377e6b969"}}' >"${root}/source/model.json"
cache=${root}/tokenizer-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507
install -d -m 0755 "${cache}/refs" "${cache}/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e"
for file in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do cp -L "${tokenizer}/${file}" "${cache}/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e/${file}"; done
printf '%s' ac9c66cc9b46af7306746a9250f23d47083d689e >"${cache}/refs/main"
export HF_HUB_DISABLE_XET=1
uvx --from huggingface_hub==0.36.0 hf download nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507 config.json README.md model.safetensors --revision ff0f0c0999f6c86b21dfd2f5803eebc4ac712224 --local-dir "${root}/draft" --max-workers 1
python3 "${repo}/submission/run/prepare_eagle.py" "${root}/draft" "${root}/source"
jq -e '.weight_sha256 == "bf226a966868fe413a22e62e471aa1130bd94f17fd2c40d322b1c8b4def1ae86" and .precision == "BF16"' "${root}/source/draft-provenance.json" >/dev/null
cp "${repo}/submission/basic/submit.sbatch" "${repo}/scripts/basic-worker.py" "${repo}/scripts/check-basic-server-info.py" "${repo}/scripts/collect-eagle-acceptance.py" "${repo}/submission/run/run-aiperf.py" "${repo}/submission/run/parse-aiperf-result.py" "${root}/scripts/"
# Only the executable prefix changes; the official benchmark argv is retained.
sed 's|/opt/sglang/venv/bin/|/opt/aiperf/bin/|g' "${repo}/submission/run/official-aiperf.sh" >"${root}/scripts/official-aiperf.sh"
cp "${repo}/submission/basic/prepare.sh" "${repo}/submission/run/prepare_eagle.py" "${root}/source/"
printf 'cd %q && sbatch scripts/submit.sbatch\n' "${root}"
