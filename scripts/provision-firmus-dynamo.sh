#!/usr/bin/env bash
set -euo pipefail
: "${GITHUB_WORKSPACE:?}"
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${GITHUB_SHA:?}"
: "${GITHUB_OUTPUT:?}"
: "${RUNTIME_ROOT:?}"
: "${CONTAINER_SHA256:?}"
: "${CONTAINER_SIZE:?}"
[[ "${PROFILE}" == false ]]
[[ "${CONTAINER_SHA256}" =~ ^[0-9a-f]{64}$ && "${CONTAINER_SIZE}" =~ ^[1-9][0-9]*$ ]]
config=${GITHUB_WORKSPACE}/configs/firmus-dynamo.json
framework=$(jq -er '.framework | select(. == "vllm" or . == "sglang")' "${config}")
jq -e '.workload == "reference-07" and .concurrency == 8 and .dynamo.version == "1.5.1"' "${config}" >/dev/null
install -d -m 2775 "${RUNTIME_ROOT}"
runtime_dir=$(mktemp -d "${RUNTIME_ROOT}/dynamo-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.XXXXXXXXXX")
printf 'runtime_dir=%s\n' "${runtime_dir}" >>"${GITHUB_OUTPUT}"
install -d -m 0755 "${runtime_dir}/logs" "${runtime_dir}/profile" "${runtime_dir}/results" "${runtime_dir}/scripts" "${runtime_dir}/source" "${runtime_dir}/home"
cp "${config}" "${runtime_dir}/source/firmus-framework.json"
runtime_image=/scratch/home/ACCOUNT/qwen/dynamo-containers/${GITHUB_RUN_ID}.sqsh
runtime_sidecar=/scratch/home/ACCOUNT/qwen/dynamo-containers/${GITHUB_RUN_ID}.container-sidecar.json
jq -e --arg backend "${framework}" --arg sha "${CONTAINER_SHA256}" --argjson dynamo "$(jq .dynamo "${config}")" '.backend == $backend and .dynamo == $dynamo and .container_artifact.sha256 == $sha' "${runtime_sidecar}" >/dev/null
for image_name in dynamo sglang; do
  if [[ "${image_name}" == dynamo ]]; then
    image=${runtime_image}; expected_sha=${CONTAINER_SHA256}; expected_size=${CONTAINER_SIZE}; sidecar=${runtime_sidecar}
  else
    image=$(jq -er .client_image.path "${config}"); expected_sha=$(jq -er .client_image.sha256 "${config}"); expected_size=$(jq -er .client_image.size_bytes "${config}"); sidecar=$(jq -er .client_image.provenance "${config}")
  fi
  [[ "$(stat -c '%s' "${image}")" == "${expected_size}" ]]
  printf 'Checking %s image SHA-256\n' "${image_name}"
  while sleep 15; do printf 'container_sha256_check_in_progress=%s\n' "${image_name}"; done &
  progress_pid=$!
  trap 'kill "${progress_pid}" 2>/dev/null || true' EXIT
  sha256sum "${image}" >"${runtime_dir}/source/${image_name}-image.sha256"
  kill "${progress_pid}"
  wait "${progress_pid}" 2>/dev/null || true
  trap - EXIT
  [[ "$(awk '{print $1}' "${runtime_dir}/source/${image_name}-image.sha256")" == "${expected_sha}" ]]
  ln -s "${image}" "${runtime_dir}/${image_name}.sqsh"
  cp "${sidecar}" "${runtime_dir}/source/${image_name}-container.json"
done
model=/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8
[[ "$(head -n 1 "${model}/.cache/huggingface/download/config.json.metadata")" == e156cb4efae43fbee1a1ab073f946a1377e6b969 ]]
jq -e '.model_type == "qwen3_moe" and .quantization_config.quant_method == "fp8"' "${model}/config.json" >/dev/null
while read -r shard; do test -s "${model}/${shard}"; done < <(jq -r '.weight_map | values | unique[]' "${model}/model.safetensors.index.json")
tokenizer=/scratch/home/ACCOUNT/qwen/framework-comparison/sources/tokenizer/ac9c66cc9b46af7306746a9250f23d47083d689e
test -s "${tokenizer}/tokenizer.json"
sha256sum "${tokenizer}"/* >"${runtime_dir}/source/tokenizer.sha256"
install -m 0755 "${GITHUB_WORKSPACE}/platforms/firmus/dynamo/run.sbatch" "${runtime_dir}/scripts/submit.sbatch"
install -m 0755 "${GITHUB_WORKSPACE}/platforms/firmus/dynamo/worker.sh" "${runtime_dir}/scripts/worker.sh"
install -m 0644 "${GITHUB_WORKSPACE}/scripts/parse-aiperf-result.py" "${GITHUB_WORKSPACE}/scripts/parse-firmus-reference-result.py" "${GITHUB_WORKSPACE}/scripts/summarize-firmus-run.py" "${runtime_dir}/scripts/"
cp "${runtime_dir}/scripts/submit.sbatch" "${runtime_dir}/scripts/worker.sh" "${runtime_dir}/source/"
jq -n --arg commit "${GITHUB_SHA}" --arg framework "${framework}" --arg run_id "${GITHUB_RUN_ID}" --arg attempt "${GITHUB_RUN_ATTEMPT}" --arg sha "${CONTAINER_SHA256}" --argjson size "${CONTAINER_SIZE}" --argjson config "$(cat "${config}")" '{schema_version:"qwen3-pd-provenance/v1",cluster:"firmus",commit:$commit,framework:$framework,dynamo:$config.dynamo,workflow_run_id:$run_id,workflow_run_attempt:$attempt,profile:false,workload:"reference-07",official_submission:false,container:{sha256:$sha,size_bytes:$size},client_container:($config.client_image | {sha256,size_bytes}),aiperf:{version:"0.12.0",commit:"be53bf2953d30e46c500e6a80fc1f8b6f84bc718"},cache_source:null}' >"${runtime_dir}/source/provenance.json"
cp "${model}/config.json" "${model}/model.safetensors.index.json" "${runtime_dir}/source/"
