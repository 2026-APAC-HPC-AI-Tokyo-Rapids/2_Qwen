#!/usr/bin/env bash
# Stage a finite kernel diagnostic, without models, tokenizers or serving workers.
set -euo pipefail
: "${GITHUB_WORKSPACE:?}"
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${GITHUB_SHA:?}"
: "${GITHUB_OUTPUT:?}"
: "${RUNTIME_ROOT:?}"
[[ "${PROFILE:?}" == false && "${REQUIRE_BUILD:?}" == false ]]
config="${GITHUB_WORKSPACE}/configs/firmus-vllm-megamoe-gate.json"
jq -e '.diagnostic == "vllm-sm90-megamoe" and .world_size == 4' "${config}" >/dev/null
image=$(jq -er .image.path "${config}")
expected_sha=$(jq -er .image.sha256 "${config}")
expected_size=$(jq -er .image.size_bytes "${config}")
sidecar=$(jq -er .image.provenance "${config}")
[[ "$(stat -c '%s' "${image}")" == "${expected_size}" ]]
[[ "$(sha256sum "${image}" | cut -d' ' -f1)" == "${expected_sha}" ]]
jq -e --arg sha "${expected_sha}" --argjson size "${expected_size}" --arg commit "$(jq -er .image.build_commit "${config}")" '.container_artifact.sha256 == $sha and .container_artifact.size_bytes == $size and .source_commits.qwen == $commit' "${sidecar}" >/dev/null
install -d -m 2775 "${RUNTIME_ROOT}"
runtime_dir=$(mktemp -d "${RUNTIME_ROOT}/megamoe-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.XXXXXXXXXX")
printf 'runtime_dir=%s\n' "${runtime_dir}" >>"${GITHUB_OUTPUT}"
install -d -m 0755 "${runtime_dir}/logs" "${runtime_dir}/profile" "${runtime_dir}/results" "${runtime_dir}/source" "${runtime_dir}/scripts/megamoe" "${runtime_dir}/home"
ln -s "${image}" "${runtime_dir}/vllm.sqsh"
install -m 0644 "${config}" "${runtime_dir}/source/megamoe-gate-config.json"
install -m 0644 "${sidecar}" "${runtime_dir}/source/vllm-container.json"
cp -a "${GITHUB_WORKSPACE}/research/firmus-vllm-megamoe/." "${runtime_dir}/scripts/megamoe/"
install -m 0755 "${GITHUB_WORKSPACE}/platforms/firmus/megamoe-gate.sbatch" "${runtime_dir}/scripts/submit.sbatch"
(cd "${runtime_dir}" && find scripts -type f -print0 | sort -z | xargs -0 sha256sum) >"${runtime_dir}/source/diagnostic-sources.sha256"
jq -n --arg commit "${GITHUB_SHA}" --arg run_id "${GITHUB_RUN_ID}" --arg attempt "${GITHUB_RUN_ATTEMPT}" --slurpfile config "${config}" '{schema_version:"firmus-kernel-gate-provenance/v1",cluster:"firmus",framework:"vllm",diagnostic:"vllm-sm90-megamoe",measurement_mode:"kernel-gate",workload:"finite-kernel",profile:false,official_submission:false,commit:$commit,workflow_run_id:$run_id,workflow_run_attempt:$attempt,world_size:4,container:$config[0].image}' >"${runtime_dir}/source/provenance.json"
