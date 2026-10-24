#!/usr/bin/env bash
# Network-enabled login-node preparation; source/probe runs offline on four GPUs.
set -euo pipefail
: "${RUNTIME_DIR:?}" "${GITHUB_WORKSPACE:?}" "${GITHUB_SHA:?}"
[[ -z "${CANDIDATE_PATCH:-}" ]]
config="${GITHUB_WORKSPACE}/configs/firmus-framework.json"
image="/home/ACCOUNT/share/sglang-containers/$(basename "$(jq -er .images.sglang.path "${config}")")"
model=/fast/PROJECT/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
[[ "$(stat -c '%s' "${image}")" == "$(jq -er .images.sglang.size_bytes "${config}")" ]]
[[ "$(jq -er .container_artifact.sha256 "${image%.sqsh}.container-sidecar.json")" == "$(jq -er .images.sglang.sha256 "${config}")" ]]
[[ "$(head -n 1 "${model}/.cache/huggingface/download/config.json.metadata")" == e156cb4efae43fbee1a1ab073f946a1377e6b969 ]]
jq -e '.model_type == "qwen3_moe" and .quantization_config.quant_method == "fp8" and .quantization_config.fmt == "e4m3"' "${model}/config.json" >/dev/null
[[ "$(jq -r '[.weight_map[]] | unique | length' "${model}/model.safetensors.index.json")" == 24 ]]
while read -r shard; do test -s "${model}/${shard}"; done < <(jq -r '[.weight_map[]] | unique[]' "${model}/model.safetensors.index.json")
jq -e '.speculation.algorithm == "EAGLE3" and .speculation.steps == 3 and .speculation.draft_tokens == 4 and .speculation.draft_repository == "nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507" and .speculation.draft_revision == "ff0f0c0999f6c86b21dfd2f5803eebc4ac712224"' "${config}" >/dev/null

install -d "${RUNTIME_DIR}/source" "${RUNTIME_DIR}/logs" "${RUNTIME_DIR}/results" "${RUNTIME_DIR}/enroot-library" "${RUNTIME_DIR}/enroot-system"
# Same pinned downloads and config adaptation as the Firmus/Genkai serving routes.
export HF_HUB_DISABLE_XET=1 HF_HOME="${RUNTIME_DIR}/tokenizer-cache" UV_CACHE_DIR="${RUNTIME_DIR}/uv-cache"
export UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1
uvx --from 'huggingface_hub==0.36.0' hf download "$(jq -er .speculation.draft_repository "${config}")" config.json README.md model.safetensors --revision "$(jq -er .speculation.draft_revision "${config}")" --local-dir "${RUNTIME_DIR}/draft" --max-workers 1
/usr/bin/python3.9 "${GITHUB_WORKSPACE}/research/simulation/prepare_eagle.py" "${RUNTIME_DIR}/draft" "${RUNTIME_DIR}/source"
jq -e '.revision == "ff0f0c0999f6c86b21dfd2f5803eebc4ac712224" and .weight_sha256 == "bf226a966868fe413a22e62e471aa1130bd94f17fd2c40d322b1c8b4def1ae86" and .weight_bytes == 1145943008 and .precision == "BF16" and .runtime_config_sha256 == "85dc5a49b8644d070c396b5b993a4ec9d0ec19aaf41dfa5025500c0f5578329c"' "${RUNTIME_DIR}/source/draft-provenance.json" >/dev/null
tokenizer_revision=ac9c66cc9b46af7306746a9250f23d47083d689e
uvx --from 'huggingface_hub==0.36.0' hf download Qwen/Qwen3-235B-A22B-Instruct-2507 config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json --max-workers 1 --revision "${tokenizer_revision}"
tokenizer_cache="${HF_HOME}/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507"
for file in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do test -s "${tokenizer_cache}/snapshots/${tokenizer_revision}/${file}"; done
install -d "${tokenizer_cache}/refs"
printf '%s' "${tokenizer_revision}" >"${tokenizer_cache}/refs/main"

# Reuse only the Enroot setup, without extracting/applying Firmus overlays.
cp -a --no-preserve=ownership,xattr /home/ACCOUNT/ku60000563/.local/lib/enroot/. "${RUNTIME_DIR}/enroot-library/"
cp -a --no-preserve=ownership,xattr /home/ACCOUNT/ku60000563/.local/etc/enroot/. "${RUNTIME_DIR}/enroot-system/"
sed -i 's/local timeout=100 pid=-1 id=0/local timeout=5000 pid=-1 id=0/' "${RUNTIME_DIR}/enroot-library/runtime.sh"
sed -i 's%--ldconfig=@$(command -v ldconfig.real || command -v ldconfig)%--ldconfig=/sbin/ldconfig%' "${RUNTIME_DIR}/enroot-system/hooks.d/98-nvidia.sh"
if [[ -e "${RUNTIME_DIR}/enroot-system/hooks.d/10-cgroups.sh" ]]; then chmod -x "${RUNTIME_DIR}/enroot-system/hooks.d/10-cgroups.sh"; fi
ln -s "${image}" "${RUNTIME_DIR}/sglang.sqsh"
install -m 0755 "${GITHUB_WORKSPACE}/kernels/probes/run-eagle-accept.sh" "${RUNTIME_DIR}/source/probe"
install -m 0644 "${GITHUB_WORKSPACE}/kernels/probes/eagle-accept.py" "${RUNTIME_DIR}/source/eagle-accept.py"
install -m 0644 "${GITHUB_WORKSPACE}/scripts/collect-eagle-acceptance.py" "${RUNTIME_DIR}/source/eagle_metrics.py"
install -m 0644 "${GITHUB_WORKSPACE}/research/simulation/prepare_eagle.py" "${config}" "${RUNTIME_DIR}/source/"
jq -n --arg commit "${GITHUB_SHA}" --arg model "${model}" --arg tokenizer_revision "${tokenizer_revision}" --slurpfile config "${config}" '{schema:"genkai-eagle-accept/v1",calibration:true,commit:$commit,image:$config[0].images.sglang,model:$model,model_revision:"e156cb4efae43fbee1a1ab073f946a1377e6b969",tokenizer_revision:$tokenizer_revision,speculation:$config[0].speculation,scope:"single TP4 server, no PD, no DP attention, no runtime overlays, CUDA graphs disabled, acceptance only",timing:{startup_estimate_s:450,aiperf_setup_estimate_s:45,benchmark_duration_s:20,grace_period_s:30,collection_and_shutdown_estimate_s:15,total_estimate_s:560,process_limit_s:570},reference_runs:[35105770792,35108498091,40071892077]}' >"${RUNTIME_DIR}/source/provenance.json"
