#!/usr/bin/env bash
set -euo pipefail

for tool in unsquashfs git jq python3 sha256sum uvx; do command -v "${tool}" >/dev/null; done
[[ $# -eq 4 ]] || { echo 'Usage: prepare.sh RUN_DIR IMAGE MODEL TOKENIZER_SNAPSHOT' >&2; exit 64; }
package="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir=$(realpath -m "$1")
image=$(realpath "$2")
model=$(realpath "$3")
tokenizer=$(realpath "$4")
mkdir "${runtime_dir}"
install -d -m 0755 "${runtime_dir}/logs" "${runtime_dir}/results" "${runtime_dir}/scripts" "${runtime_dir}/source" "${runtime_dir}/home"
config=${package}/config/serving.json
install -m 0644 "${config}" "${runtime_dir}/source/firmus-framework.json"
[[ "$(stat -c '%s' "${image}")" == "$(jq -er .images.sglang.size_bytes "${config}")" ]]
sha256sum "${image}" >"${runtime_dir}/source/sglang-image.sha256"
[[ "$(awk '{print $1}' "${runtime_dir}/source/sglang-image.sha256")" == "$(jq -er .images.sglang.sha256 "${config}")" ]]
ln -s "${image}" "${runtime_dir}/sglang.sqsh"

# Same checkpoint and shard checks as scripts/provision-firmus-framework.sh.
[[ -s "${model}/config.json" && -s "${model}/model.safetensors.index.json" ]]
[[ "$(head -n 1 "${model}/.cache/huggingface/download/config.json.metadata")" == e156cb4efae43fbee1a1ab073f946a1377e6b969 ]]
jq -e '.model_type == "qwen3_moe" and .quantization_config.quant_method == "fp8"' "${model}/config.json" >/dev/null
[[ "$(jq -r '[.weight_map[]] | unique | length' "${model}/model.safetensors.index.json")" == 24 ]]
while read -r shard; do test -s "${model}/${shard}"; done < <(jq -r '[.weight_map[]] | unique[]' "${model}/model.safetensors.index.json")
jq -n --arg model_path "${model}" '{model:{repository:"Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",revision:"e156cb4efae43fbee1a1ab073f946a1377e6b969",path:$model_path,precision:"fp8",shards:24},tokenizer:{repository:"Qwen/Qwen3-235B-A22B-Instruct-2507",revision:"ac9c66cc9b46af7306746a9250f23d47083d689e"}}' >"${runtime_dir}/source/model.json"

test -s "${tokenizer}/tokenizer.json"
sha256sum "${tokenizer}"/* >"${runtime_dir}/source/tokenizer.sha256"
tokenizer_cache=${runtime_dir}/tokenizer-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507
install -d -m 0755 "${tokenizer_cache}/refs" "${tokenizer_cache}/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e"
for file in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do cp -L "${tokenizer}/${file}" "${tokenizer_cache}/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e/${file}"; done
printf '%s' ac9c66cc9b46af7306746a9250f23d47083d689e >"${tokenizer_cache}/refs/main"

export HF_HUB_DISABLE_XET=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1
uvx --from 'huggingface_hub==0.36.0' hf download "$(jq -er '.speculation.draft_repository' "${config}")" config.json README.md model.safetensors --revision "$(jq -er '.speculation.draft_revision' "${config}")" --local-dir "${runtime_dir}/draft" --max-workers 1
python3 "${package}/run/prepare_eagle.py" "${runtime_dir}/draft" "${runtime_dir}/source"
jq -e '.revision == "ff0f0c0999f6c86b21dfd2f5803eebc4ac712224" and .weight_sha256 == "bf226a966868fe413a22e62e471aa1130bd94f17fd2c40d322b1c8b4def1ae86" and .weight_bytes == 1145943008 and .precision == "BF16"' "${runtime_dir}/source/draft-provenance.json" >/dev/null

# Extract only the two runtime packages; patch paths are relative to these roots.
stage_overlay() {
  local name=$1 image_path=$2 package_path=$3 patch
  local root="${runtime_dir}/overlay/${name}" extract="${runtime_dir}/overlay/extract-${name}"
  local patches=("${package}/patches/overlay/${name}"/[0-9][0-9]-*.patch)
  install -d -m 0755 "${runtime_dir}/overlay"
  unsquashfs -no-progress -no-xattrs -processors 1 -d "${extract}" "${runtime_dir}/sglang.sqsh" "${image_path}"
  chmod -R u+w "${extract}"
  install -d -m 0755 "$(dirname "${root}/${package_path}")" "${runtime_dir}/source/${name}-overlay"
  mv "${extract}/${image_path}" "${root}/${package_path}"
  rm -rf -- "${extract}"
  for patch in "${patches[@]}"; do
    (cd "${root}" && git apply --check "${patch}" && git apply "${patch}")
    cp "${patch}" "${runtime_dir}/source/${name}-overlay/"
  done
  for patch in "${patches[@]}"; do jq -n --arg path "${patch#"${package}/"}" --arg sha "$(sha256sum "${patch}" | awk '{print $1}')" '{path:$path,sha256:$sha}'; done | jq -s --arg image "$(jq -er .images.sglang.sha256 "${config}")" --arg mount "/${image_path}" '{base_image_sha256:$image,mount:$mount,patches:.}' >"${runtime_dir}/source/${name}-overlay.json"
}
stage_overlay sglang opt/sglang/source/python/sglang python/sglang
stage_overlay deep-gemm opt/sglang/venv/lib/python3.12/site-packages/deep_gemm deep_gemm

install -m 0644 "${package}/config/prefill-expert-location.json" "${runtime_dir}/source/prefill-expert-location.json"
install -m 0755 "${package}/run/submit.sbatch" "${package}/run/official-aiperf.sh" "${runtime_dir}/scripts/"
install -m 0644 "${package}/run/run-aiperf.py" "${package}/run/parse-aiperf-result.py" "${runtime_dir}/scripts/"
cp "${package}/run/prepare.sh" "${package}/run/prepare_eagle.py" "${runtime_dir}/scripts/submit.sbatch" "${runtime_dir}/scripts/official-aiperf.sh" "${runtime_dir}/scripts/run-aiperf.py" "${runtime_dir}/scripts/parse-aiperf-result.py" "${runtime_dir}/source/"
printf 'Prepared %s\n' "${runtime_dir}"
printf 'cd %q && sbatch scripts/submit.sbatch\n' "${runtime_dir}"
