#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${GITHUB_SHA:?}"
: "${GITHUB_OUTPUT:?}"
: "${RUNTIME_ROOT:?}"
: "${PROFILE:?}"
: "${REQUIRE_BUILD:?}"
[[ "${PROFILE}" == true || "${PROFILE}" == false ]]
[[ "${REQUIRE_BUILD}" == false ]] || { echo 'This branch reuses pinned baseline images; rebuild and update their hashes before requesting a source experiment.' >&2; exit 64; }

config="${GITHUB_WORKSPACE}/configs/firmus-framework.json"
framework="$(jq -er '.framework | select(. == "sglang" or . == "vllm")' "${config}")"
jq -e --argjson profile "${PROFILE}" '((.workload == "reference-07" and .framework == "vllm") or (.workload == "official" and ($profile == false or .framework == "vllm"))) and (.concurrency | type == "number" and . >= 1 and floor == .)' "${config}" >/dev/null
install -d -m 2775 "${RUNTIME_ROOT}"
runtime_dir="$(mktemp -d "${RUNTIME_ROOT}/pd-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.XXXXXXXXXX")"
printf 'runtime_dir=%s\n' "${runtime_dir}" >>"${GITHUB_OUTPUT}"
install -d -m 0755 "${runtime_dir}/logs" "${runtime_dir}/profile" "${runtime_dir}/results" "${runtime_dir}/scripts" "${runtime_dir}/source" "${runtime_dir}/home"
install -m 0644 "${config}" "${runtime_dir}/source/firmus-framework.json"
for image_framework in sglang "${framework}"; do
  image="$(jq -er --arg framework "${image_framework}" '.images[$framework].path' "${config}")"
  expected_sha="$(jq -er --arg framework "${image_framework}" '.images[$framework].sha256' "${config}")"
  expected_size="$(jq -er --arg framework "${image_framework}" '.images[$framework].size_bytes' "${config}")"
  provenance="$(jq -er --arg framework "${image_framework}" '.images[$framework].provenance' "${config}")"
  [[ -L "${runtime_dir}/${image_framework}.sqsh" ]] && continue
  [[ "${expected_sha}" =~ ^[0-9a-f]{64}$ && "$(stat -c '%s' "${image}")" == "${expected_size}" ]]
  echo "Checking ${image_framework} image SHA-256"
  while sleep 15; do echo "container_sha256_check_in_progress=${image_framework}"; done &
  progress_pid=$!
  trap 'kill "${progress_pid}" 2>/dev/null || true' EXIT
  sha256sum "${image}" >"${runtime_dir}/source/${image_framework}-image.sha256"
  # A login fork failure may already have stopped the advisory progress loop.
  kill "${progress_pid}" 2>/dev/null || true
  wait "${progress_pid}" 2>/dev/null || true
  trap - EXIT
  [[ "$(awk '{print $1}' "${runtime_dir}/source/${image_framework}-image.sha256")" == "${expected_sha}" ]]
  ln -s "${image}" "${runtime_dir}/${image_framework}.sqsh"
  cp "${provenance}" "${runtime_dir}/source/${image_framework}-container.json"
  if [[ "${framework}" == sglang ]]; then cp "${provenance}" "${runtime_dir}/source/container-sidecar.json"; fi
done
model=/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8
[[ -s "${model}/config.json" && -s "${model}/model.safetensors.index.json" ]]
[[ "$(head -n 1 "${model}/.cache/huggingface/download/config.json.metadata")" == e156cb4efae43fbee1a1ab073f946a1377e6b969 ]]
jq -e '.model_type == "qwen3_moe" and .quantization_config.quant_method == "fp8"' "${model}/config.json" >/dev/null
while read -r shard; do test -s "${model}/${shard}"; done < <(jq -r '.weight_map | values | unique[]' "${model}/model.safetensors.index.json")
tokenizer=/scratch/home/ACCOUNT/qwen/framework-comparison/sources/tokenizer/ac9c66cc9b46af7306746a9250f23d47083d689e
test -s "${tokenizer}/tokenizer.json"
sha256sum "${tokenizer}"/* >"${runtime_dir}/source/tokenizer.sha256"
if [[ "$(jq -r .workload "${config}")" == official ]]; then
  tokenizer_cache=${runtime_dir}/tokenizer-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507
  install -d -m 0755 "${tokenizer_cache}/refs" "${tokenizer_cache}/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e"
  for file in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do cp -L "${tokenizer}/${file}" "${tokenizer_cache}/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e/${file}"; done
  printf '%s' ac9c66cc9b46af7306746a9250f23d47083d689e >"${tokenizer_cache}/refs/main"
fi
if [[ "$(jq -r '.speculation.algorithm // empty' "${config}")" == EAGLE3 ]]; then
  jq -e '.speculation.algorithm == "EAGLE3" and ((.speculation.steps == 3 and .speculation.draft_tokens == 4) or (.framework == "vllm" and .speculation.steps == 2 and .speculation.draft_tokens == 3))' "${config}" >/dev/null
  export HF_HUB_DISABLE_XET=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1
  uvx --from 'huggingface_hub==0.36.0' hf download "$(jq -er '.speculation.draft_repository' "${config}")" config.json README.md model.safetensors --revision "$(jq -er '.speculation.draft_revision' "${config}")" --local-dir "${runtime_dir}/draft" --max-workers 1
  python3 "${GITHUB_WORKSPACE}/research/simulation/prepare_eagle.py" "${runtime_dir}/draft" "${runtime_dir}/source"
  jq -e '.revision == "ff0f0c0999f6c86b21dfd2f5803eebc4ac712224" and .weight_sha256 == "bf226a966868fe413a22e62e471aa1130bd94f17fd2c40d322b1c8b4def1ae86" and .weight_bytes == 1145943008 and .precision == "BF16"' "${runtime_dir}/source/draft-provenance.json" >/dev/null
  cp "${GITHUB_WORKSPACE}/research/simulation/prepare_eagle.py" "${runtime_dir}/source/"
fi
install -m 0755 "${GITHUB_WORKSPACE}/platforms/firmus/framework-comparison/${framework}.sbatch" "${runtime_dir}/scripts/submit.sbatch"
install -m 0755 "${GITHUB_WORKSPACE}/platforms/firmus/framework-comparison/official-aiperf.sh" "${runtime_dir}/scripts/official-aiperf.sh"
install -m 0644 "${GITHUB_WORKSPACE}/platforms/firmus/framework-comparison/vllm-proxy.py" "${runtime_dir}/vllm-proxy.py"
install -m 0644 "${GITHUB_WORKSPACE}/scripts/parse-aiperf-result.py" "${GITHUB_WORKSPACE}/scripts/parse-firmus-reference-result.py" "${GITHUB_WORKSPACE}/scripts/summarize-firmus-run.py" "${runtime_dir}/scripts/"
if [[ "${PROFILE}" == true ]]; then
  install -m 0755 "${GITHUB_WORKSPACE}/scripts/exec-profiler.sh" "${GITHUB_WORKSPACE}/scripts/capture-firmus-profile.sh" "${runtime_dir}/scripts/"
  install -m 0644 "${GITHUB_WORKSPACE}/scripts/validate-firmus-profile.py" "${runtime_dir}/scripts/"
  command -v ar >/dev/null
  package="${runtime_dir}/nsight.deb"
  curl --fail --location --show-error --max-time 600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-systems-2025.3.2_2025.3.2.474-1_amd64.deb -o "${package}"
  printf 'c7cfe27e2250eb91e1a67e7feb5f2c490c7f598e3b3a3d047aff000bc49f9d6b  %s\n' "${package}" | sha256sum --check >"${runtime_dir}/source/nsys-package.sha256"
  install -d -m 0755 "${runtime_dir}/nsys-extract"
  ar p "${package}" data.tar.gz | tar -xz -C "${runtime_dir}/nsys-extract" --wildcards './opt/nvidia/nsight-systems/2025.3.2/bin/*' './opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/*' './opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/*' './opt/nvidia/nsight-systems/2025.3.2/EULA.txt'
  mv "${runtime_dir}/nsys-extract/opt/nvidia/nsight-systems/2025.3.2" "${runtime_dir}/nsys"
  rm -rf -- "${package}" "${runtime_dir}/nsys-extract"
  test -x "${runtime_dir}/nsys/bin/nsys"
  cp "${runtime_dir}/scripts/exec-profiler.sh" "${runtime_dir}/scripts/capture-firmus-profile.sh" "${runtime_dir}/scripts/validate-firmus-profile.py" "${runtime_dir}/source/"
fi
cp "${runtime_dir}/scripts/submit.sbatch" "${runtime_dir}/scripts/official-aiperf.sh" "${runtime_dir}/vllm-proxy.py" "${runtime_dir}/source/"
if [[ "${framework}" == vllm ]]; then
  for worker in prefill-0 prefill-1 prefill-2 prefill-3 prefill-4 prefill-5 decode-0; do
    source_cache="/scratch/home/ACCOUNT/qwen/framework-comparison/matched-settings-comparison/home/${worker}/.tensorrt_llm"
    if [[ -d "${source_cache}/cache" ]]; then
      install -d -m 0755 "${runtime_dir}/home/${worker}"
      cp -a --no-preserve=ownership "${source_cache}" "${runtime_dir}/home/${worker}/"
    fi
  done
  (cd "${runtime_dir}" && find home -type f -print0 | sort -z | xargs -0 -r sha256sum) >"${runtime_dir}/source/jit-cache.sha256"
fi
jq -n --arg model_path "${model}" '{schema_version:"qwen3-pd-model/v1",model:{repository:"Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",revision:"e156cb4efae43fbee1a1ab073f946a1377e6b969",path:$model_path,precision:"fp8",shards:24},tokenizer:{repository:"Qwen/Qwen3-235B-A22B-Instruct-2507",revision:"ac9c66cc9b46af7306746a9250f23d47083d689e"}}' >"${runtime_dir}/source/model.json"
jq -n --arg commit "${GITHUB_SHA}" --arg framework "${framework}" --arg workload "$(jq -r .workload "${config}")" --arg run_id "${GITHUB_RUN_ID}" --arg attempt "${GITHUB_RUN_ATTEMPT}" --argjson profile "${PROFILE}" --argjson images "$(jq .images "${config}")" '{schema_version:"qwen3-pd-provenance/v1",cluster:"firmus",commit:$commit,framework:$framework,workflow_run_id:$run_id,workflow_run_attempt:$attempt,profile:$profile,workload:$workload,official_submission:false,container:($images[$framework] | {sha256,size_bytes}),client_container:($images.sglang | {sha256,size_bytes}),aiperf:{version:"0.12.0",commit:"be53bf2953d30e46c500e6a80fc1f8b6f84bc718"},cache_source:(if $framework == "vllm" then "matched-settings-comparison/home" else null end)}' >"${runtime_dir}/source/provenance.json"
