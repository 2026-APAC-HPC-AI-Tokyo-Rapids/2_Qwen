#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${GITHUB_SHA:?}"
: "${GITHUB_OUTPUT:?}"
: "${RUNTIME_ROOT:?}"
: "${PROFILE:?}"
: "${QWEN_COMMIT_MESSAGE:=}"
[[ "${PROFILE}" == false ]] || { echo 'This reference comparison does not implement GPU profiling.' >&2; exit 64; }
[[ "${QWEN_COMMIT_MESSAGE}" != '[require-build]'* ]] || { echo 'This branch reuses pinned baseline images; rebuild and update their hashes before requesting a source experiment.' >&2; exit 64; }

config="${GITHUB_WORKSPACE}/configs/firmus-framework.json"
framework="$(jq -er '.framework | select(. == "sglang" or . == "vllm")' "${config}")"
jq -e '.workload == "reference-07" and (.concurrency | type == "number" and . >= 1 and floor == .)' "${config}" >/dev/null
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
  kill "${progress_pid}"
  wait "${progress_pid}" 2>/dev/null || true
  trap - EXIT
  [[ "$(awk '{print $1}' "${runtime_dir}/source/${image_framework}-image.sha256")" == "${expected_sha}" ]]
  ln -s "${image}" "${runtime_dir}/${image_framework}.sqsh"
  cp "${provenance}" "${runtime_dir}/source/${image_framework}-container.json"
done
model=/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8
[[ -s "${model}/config.json" && -s "${model}/model.safetensors.index.json" ]]
[[ "$(head -n 1 "${model}/.cache/huggingface/download/config.json.metadata")" == e156cb4efae43fbee1a1ab073f946a1377e6b969 ]]
jq -e '.model_type == "qwen3_moe" and .quantization_config.quant_method == "fp8"' "${model}/config.json" >/dev/null
tokenizer=/scratch/home/apacsc14/qwen/framework-comparison/sources/tokenizer/ac9c66cc9b46af7306746a9250f23d47083d689e
test -s "${tokenizer}/tokenizer.json"
sha256sum "${tokenizer}"/* >"${runtime_dir}/source/tokenizer.sha256"
install -m 0755 "${GITHUB_WORKSPACE}/platforms/firmus/framework-comparison/${framework}.sbatch" "${runtime_dir}/scripts/submit.sbatch"
install -m 0644 "${GITHUB_WORKSPACE}/platforms/firmus/framework-comparison/vllm-proxy.py" "${runtime_dir}/vllm-proxy.py"
install -m 0644 "${GITHUB_WORKSPACE}/scripts/parse-aiperf-result.py" "${GITHUB_WORKSPACE}/scripts/parse-firmus-reference-result.py" "${runtime_dir}/scripts/"
cp "${runtime_dir}/scripts/submit.sbatch" "${runtime_dir}/vllm-proxy.py" "${runtime_dir}/source/"
if [[ "${framework}" == vllm ]]; then
  for worker in prefill-0 prefill-1 prefill-2 prefill-3 decode-0 decode-1; do
    source_cache="/scratch/home/apacsc14/qwen/framework-comparison/matched-settings-comparison/home/${worker}/.tensorrt_llm"
    if [[ -d "${source_cache}/cache" ]]; then
      install -d -m 0755 "${runtime_dir}/home/${worker}"
      cp -a --no-preserve=ownership "${source_cache}" "${runtime_dir}/home/${worker}/"
    fi
  done
  (cd "${runtime_dir}" && find home -type f -print0 | sort -z | xargs -0 -r sha256sum) >"${runtime_dir}/source/jit-cache.sha256"
fi
jq -n --arg commit "${GITHUB_SHA}" --arg framework "${framework}" --arg run_id "${GITHUB_RUN_ID}" --arg attempt "${GITHUB_RUN_ATTEMPT}" '{schema_version:"qwen3-pd-provenance/v1",cluster:"firmus",commit:$commit,framework:$framework,workflow_run_id:$run_id,workflow_run_attempt:$attempt,profile:false,workload:"reference-07",official_submission:false,model:{repository:"Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",revision:"e156cb4efae43fbee1a1ab073f946a1377e6b969"},tokenizer:{revision:"ac9c66cc9b46af7306746a9250f23d47083d689e"},aiperf:{version:"0.12.0"},cache_source:"matched-settings-comparison/home"}' >"${runtime_dir}/source/provenance.json"
