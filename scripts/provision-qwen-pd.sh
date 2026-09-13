#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_ARCHIVE_DIR:?container archive directory is required}"
: "${CONTAINER_SHA256:?container SHA-256 is required}"
: "${CONTAINER_SIZE:?container size is required}"
: "${ENROOT_LIBRARY_PATH:?Enroot library path is required}"
: "${ENROOT_SYSCONF_PATH:?Enroot system configuration path is required}"
: "${GITHUB_RUN_ATTEMPT:?workflow run attempt is required}"
: "${GITHUB_RUN_ID:?workflow run ID is required}"
: "${GITHUB_SHA:?workflow commit is required}"
: "${GITHUB_WORKSPACE:?workflow workspace is required}"
: "${MODEL_PATH:?model path is required}"
: "${PROFILE:?profile selection is required}"
: "${PROVISION_OUTPUT:?GitHub output file is required}"
: "${RUNNER_TEMP:?runner temporary directory is required}"
: "${RUNTIME_ROOT:?runtime root is required}"

model_repository=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
model_revision=e156cb4efae43fbee1a1ab073f946a1377e6b969
tokenizer_repository=Qwen/Qwen3-235B-A22B-Instruct-2507
tokenizer_revision=ac9c66cc9b46af7306746a9250f23d47083d689e
archive_image="${CONTAINER_ARCHIVE_DIR}/${GITHUB_RUN_ID}.sqsh"
archive_sidecar="${CONTAINER_ARCHIVE_DIR}/${GITHUB_RUN_ID}.container-sidecar.json"
[[ "${CONTAINER_SHA256}" =~ ^[0-9a-f]{64}$ && "${CONTAINER_SIZE}" =~ ^[1-9][0-9]*$ ]]
[[ "$(jq -er '.container_artifact.size_bytes' "${archive_sidecar}")" == "${CONTAINER_SIZE}" ]]
[[ "$(jq -er '.container_artifact.sha256' "${archive_sidecar}")" == "${CONTAINER_SHA256}" ]]
[[ "$(stat -c '%s' "${archive_image}")" == "${CONTAINER_SIZE}" ]]
[[ -s "${MODEL_PATH}/config.json" && -s "${MODEL_PATH}/model.safetensors.index.json" ]]
[[ "$(find "${MODEL_PATH}" -maxdepth 1 -type f -name 'model-*-of-00024.safetensors' | wc -l)" -eq 24 ]]
[[ "$(head -n 1 "${MODEL_PATH}/.cache/huggingface/download/config.json.metadata")" == "${model_revision}" ]]
jq -e '.model_type == "qwen3_moe" and .architectures == ["Qwen3MoeForCausalLM"] and .quantization_config.quant_method == "fp8" and .quantization_config.fmt == "e4m3"' "${MODEL_PATH}/config.json" >/dev/null

install -d -m 2775 "${RUNTIME_ROOT}"
runtime_dir="$(mktemp -d "${RUNTIME_ROOT}/pd-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.XXXXXXXXXX")"
printf 'runtime_dir=%s\n' "${runtime_dir}" >>"${PROVISION_OUTPUT}"
install -d -m 0755 "${runtime_dir}/enroot-library" "${runtime_dir}/enroot-system" "${runtime_dir}/logs" "${runtime_dir}/profile" "${runtime_dir}/results" "${runtime_dir}/scripts" "${runtime_dir}/source" "${runtime_dir}/tokenizer-cache"
ln -s "${archive_image}" "${runtime_dir}/sglang.sqsh"
cp "${archive_sidecar}" "${runtime_dir}/source/container-sidecar.json"
cp -a --no-preserve=ownership,xattr "${ENROOT_SYSCONF_PATH}/." "${runtime_dir}/enroot-system/"
cp -a --no-preserve=ownership,xattr "${ENROOT_LIBRARY_PATH}/." "${runtime_dir}/enroot-library/"
runtime_library="${runtime_dir}/enroot-library/runtime.sh"
[[ "$(grep -Fxc '    local timeout=100 pid=-1 id=0' "${runtime_library}")" -eq 1 ]]
sed -i 's/local timeout=100 pid=-1 id=0/local timeout=5000 pid=-1 id=0/' "${runtime_library}"
nvidia_hook="${runtime_dir}/enroot-system/hooks.d/98-nvidia.sh"
if [[ -x "${nvidia_hook}" ]]; then sed -i 's%--ldconfig=@$(command -v ldconfig.real || command -v ldconfig)%--ldconfig=/sbin/ldconfig%' "${nvidia_hook}"; fi

install -m 0755 "${GITHUB_WORKSPACE}/submission/run/submit.sbatch" "${runtime_dir}/scripts/submit.sbatch"
install -m 0644 "${GITHUB_WORKSPACE}/configs/firmus-pd.json" "${runtime_dir}/source/run-config.json"
export HF_HOME="${runtime_dir}/tokenizer-cache"
export UV_CACHE_DIR="${RUNNER_TEMP}/uv-cache"
uvx --from 'huggingface_hub==0.36.0' hf download "${tokenizer_repository}" config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json --revision "${tokenizer_revision}" >/dev/null
tokenizer_cache_repository="${HF_HOME}/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507"
tokenizer_snapshot="${tokenizer_cache_repository}/snapshots/${tokenizer_revision}"
for tokenizer_file in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do test -s "${tokenizer_snapshot}/${tokenizer_file}"; done
install -d -m 0755 "${tokenizer_cache_repository}/refs"
printf '%s' "${tokenizer_revision}" >"${tokenizer_cache_repository}/refs/main"
jq -n --arg repository "${model_repository}" --arg revision "${model_revision}" --arg path "${MODEL_PATH}" --arg tokenizer_repository "${tokenizer_repository}" --arg tokenizer_revision "${tokenizer_revision}" '{schema_version:"qwen3-pd-model/v1",model:{repository:$repository,revision:$revision,path:$path,precision:"fp8",shards:24},tokenizer:{repository:$tokenizer_repository,revision:$tokenizer_revision}}' >"${runtime_dir}/source/model.json"
jq -n --arg commit "${GITHUB_SHA}" --argjson profile "${PROFILE}" --arg container_sha256 "${CONTAINER_SHA256}" --argjson container_size "${CONTAINER_SIZE}" '{schema_version:"qwen3-pd-provenance/v1",commit:$commit,profile:$profile,container:{sha256:$container_sha256,size_bytes:$container_size},aiperf:{version:"0.12.0",commit:"be53bf2953d30e46c500e6a80fc1f8b6f84bc718"},sglang_router:{version:"0.3.2"}}' >"${runtime_dir}/source/provenance.json"
