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
[[ "${REQUIRE_BUILD}" == true || "${REQUIRE_BUILD}" == false ]]
if [[ "${REQUIRE_BUILD}" == true ]]; then
  [[ "${CONTAINER_SHA256:-}" =~ ^[0-9a-f]{64}$ && "${CONTAINER_SIZE:-}" =~ ^[1-9][0-9]*$ ]] || { echo 'A source experiment requires the successful CPU build SHA-256 and size.' >&2; exit 64; }
fi

config="${GITHUB_WORKSPACE}/${1:-configs/firmus-framework.json}"
framework="$(jq -er '.framework | select(. == "sglang" or . == "vllm")' "${config}")"
jq -e --argjson profile "${PROFILE}" '((.workload == "reference-07" and .framework == "vllm") or (.workload == "official" and (.framework == "sglang" or $profile == false))) and (.concurrency | type == "number" and . >= 1 and floor == .)' "${config}" >/dev/null
# A profile-only recovery must never be mistaken for a new official score.
jq -e --argjson profile "${PROFILE}" '(.profile_only // false) as $only | ($only | type == "boolean") and ($only == false or ($profile and .framework == "sglang" and .workload == "official"))' "${config}" >/dev/null
# Routing capture is a profile-only diagnostic; reject mistakes before image hashing or staging.
jq -e '(.prefill.routing_capture_samples // 0) | type == "number" and floor == . and . >= 0 and . <= 8' "${config}" >/dev/null
routing_capture_samples=$(jq -r '.prefill.routing_capture_samples // 0' "${config}")
if (( routing_capture_samples > 0 )); then
  [[ "${PROFILE}" == true && "${framework}" == sglang ]]
  jq -e '.workload == "official" and .prefill.workers == 3 and .prefill.tp == 4 and .prefill.dp == 4 and .prefill.ep == 4 and .prefill.ep_num_redundant_experts == 12 and .prefill.moe_a2a_backend == "megamoe" and .prefill.moe_runner_backend == "deep_gemm" and .prefill.expert_location != "trivial"' "${config}" >/dev/null
  test -s "${GITHUB_WORKSPACE}/$(jq -er .prefill.expert_location "${config}")"
  test -s "${GITHUB_WORKSPACE}/patches/sglang/prefill-routing-capture.patch"
  test -s "${GITHUB_WORKSPACE}/scripts/summarize-prefill-routing.py"
fi
install -d -m 2775 "${RUNTIME_ROOT}"
runtime_dir="$(mktemp -d "${RUNTIME_ROOT}/pd-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.XXXXXXXXXX")"
# This process owns only the mktemp directory above. A failed or interrupted
# preparation must not leak it, including interruption during image hashing.
provision_complete=false
cleanup_provision() {
  local status=$?
  trap - EXIT
  if [[ "${provision_complete}" != true ]]; then
    chmod -R u+w "${runtime_dir}" 2>/dev/null || true
    rm -rf -- "${runtime_dir}"
  fi
  exit "${status}"
}
trap cleanup_provision EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'runtime_dir=%s\n' "${runtime_dir}" >>"${GITHUB_OUTPUT}"
install -d -m 0755 "${runtime_dir}/logs" "${runtime_dir}/profile" "${runtime_dir}/results" "${runtime_dir}/scripts" "${runtime_dir}/source" "${runtime_dir}/home"
install -m 0644 "${config}" "${runtime_dir}/source/firmus-framework.json"
config="${runtime_dir}/source/firmus-framework.json"
if (( routing_capture_samples > 0 )); then
  jq --arg patch "patches/sglang/prefill-routing-capture.patch" '.sglang_overlay_patches |= (map(select(. != $patch)) + [$patch])' "${config}" >"${config}.tmp"
  mv "${config}.tmp" "${config}"
fi
if [[ "${framework}" == sglang && -n "${CONTAINER_SHA256:-}" ]]; then
  [[ "${CONTAINER_SHA256}" =~ ^[0-9a-f]{64}$ && "${CONTAINER_SIZE:-}" =~ ^[1-9][0-9]*$ ]]
  archive=/scratch/home/ACCOUNT/qwen/containers
  if [[ "${REQUIRE_BUILD}" == false ]]; then
    if [[ "$(jq -r '.prefill.pp // 1' "${config}")" == 4 ]]; then
      source_run=$(jq -er .container_build_run_id "${config}")
      source_commit=$(jq -er .container_source_commit "${config}")
      [[ "${CONTAINER_SIZE}" == "$(jq -er .container_size_bytes "${config}")" ]]
      jq -e --arg commit "${source_commit}" --arg sha "${CONTAINER_SHA256}" '.source_commits.qwen == $commit and .container_artifact.sha256 == $sha' "${archive}/${source_run}.container-sidecar.json" "${archive}/${GITHUB_RUN_ID}.container-sidecar.json" >/dev/null
    else
      [[ "${CONTAINER_SHA256}" == "$(jq -er .images.sglang.sha256 "${config}")" ]]
    fi
  fi
  # A fresh build replaces a pinned image (PP4 container_* fields) in the runtime config, so the overlay
  # records and the PP4 CPU gate describe the image actually served.
  jq --arg path "${archive}/${GITHUB_RUN_ID}.sqsh" --arg sha "${CONTAINER_SHA256}" --argjson size "${CONTAINER_SIZE}" --arg provenance "${archive}/${GITHUB_RUN_ID}.container-sidecar.json" --argjson require_build "${REQUIRE_BUILD}" --argjson run "${GITHUB_RUN_ID}" --arg commit "${GITHUB_SHA}" '.images.sglang = {path:$path,sha256:$sha,size_bytes:$size,provenance:$provenance} | if $require_build and has("container_sha256") then .container_build_run_id = $run | .container_sha256 = $sha | .container_size_bytes = $size | .container_source_commit = $commit else . end' "${config}" >"${config}.tmp"
  mv "${config}.tmp" "${config}"
  jq '.images.sglang' "${config}"
  jq -e --arg commit "${GITHUB_SHA}" --arg sha "${CONTAINER_SHA256}" --argjson require_build "${REQUIRE_BUILD}" '.container_artifact.sha256 == $sha and ($require_build == false or .source_commits.qwen == $commit)' "${archive}/${GITHUB_RUN_ID}.container-sidecar.json" >/dev/null
fi
if [[ "$(jq -r '.prefill.pp // 1' "${config}")" == 4 ]]; then
  python3 "${GITHUB_WORKSPACE}/scripts/prepare-firmus-pp4-settings.py" "${config}" "${GITHUB_WORKSPACE}" "${runtime_dir}"
  gpu_clock=$(jq -r '.prefill.gpu_sm_clock // empty' "${config}")
  if [[ -n "${gpu_clock}" ]]; then
    install -m 0644 "${GITHUB_WORKSPACE}/${gpu_clock}" "${runtime_dir}/source/firmus-gpu-sm-clock.json"
    install -m 0644 "${GITHUB_WORKSPACE}/scripts/order-firmus-pp4-gpus.py" "${runtime_dir}/scripts/"
  fi
fi
selection_record=$(jq -r '.reuse_kernel_selection // empty' "${config}")
if [[ -n "${selection_record}" ]]; then
  [[ "${framework}" == sglang && "${REQUIRE_BUILD}" == false ]]
  python3 "${GITHUB_WORKSPACE}/scripts/reuse-firmus-kernel-selection.py" --record "${GITHUB_WORKSPACE}/${selection_record}" --config "${config}" --repo "${GITHUB_WORKSPACE}" --runtime "${runtime_dir}"
  install -m 0644 "${GITHUB_WORKSPACE}/scripts/reuse-firmus-kernel-selection.py" "${runtime_dir}/source/"
fi
for image_framework in sglang "${framework}"; do
  image="$(jq -er --arg framework "${image_framework}" '.images[$framework].path' "${config}")"
  expected_sha="$(jq -er --arg framework "${image_framework}" '.images[$framework].sha256' "${config}")"
  expected_size="$(jq -er --arg framework "${image_framework}" '.images[$framework].size_bytes' "${config}")"
  provenance="$(jq -er --arg framework "${image_framework}" '.images[$framework].provenance' "${config}")"
  [[ -L "${runtime_dir}/${image_framework}.sqsh" ]] && continue
  [[ "${expected_sha}" =~ ^[0-9a-f]{64}$ && "$(stat -c '%s' "${image}")" == "${expected_size}" ]]
  echo "Checking ${image_framework} image SHA-256"
  # Keep hashing foreground: an advisory timer adds a kill/wait race to cleanup.
  sha256sum "${image}" >"${runtime_dir}/source/${image_framework}-image.sha256"
  [[ "$(awk '{print $1}' "${runtime_dir}/source/${image_framework}-image.sha256")" == "${expected_sha}" ]]
  ln -s "${image}" "${runtime_dir}/${image_framework}.sqsh"
  cp "${provenance}" "${runtime_dir}/source/${image_framework}-container.json"
  if [[ "${framework}" == sglang ]]; then cp "${provenance}" "${runtime_dir}/source/container-sidecar.json"; fi
done
if [[ "${framework}" == sglang ]]; then
  bash "${GITHUB_WORKSPACE}/scripts/check-firmus-image.sh" "${runtime_dir}/sglang.sqsh" "${runtime_dir}/source" "${GITHUB_WORKSPACE}"
fi
if [[ "${framework}" == sglang ]]; then bash "${GITHUB_WORKSPACE}/scripts/stage-qwen-overlays.sh" "${runtime_dir}" "${config}" "${GITHUB_WORKSPACE}"; fi
if [[ "$(jq -r '.prefill.pp // 1' "${config}")" == 4 ]]; then
  uv run --no-project --python 3.12 --with torch==2.11.0+cpu --with numpy==2.2.6 --extra-index-url https://download.pytorch.org/whl/cpu python "${GITHUB_WORKSPACE}/scripts/check-firmus-pp4-cpu.py" "${runtime_dir}" "${config}" "${GITHUB_WORKSPACE}"
fi
if (( routing_capture_samples > 0 )); then
  test -s "${runtime_dir}/overlay/sglang/python/sglang/srt/layers/moe/prefill_routing_capture.py"
  install -m 0644 "${GITHUB_WORKSPACE}/scripts/summarize-prefill-routing.py" "${runtime_dir}/scripts/"
  cp "${runtime_dir}/scripts/summarize-prefill-routing.py" "${runtime_dir}/source/"
fi
model=/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8
[[ -s "${model}/config.json" && -s "${model}/model.safetensors.index.json" ]]
[[ "$(head -n 1 "${model}/.cache/huggingface/download/config.json.metadata")" == e156cb4efae43fbee1a1ab073f946a1377e6b969 ]]
jq -e '.model_type == "qwen3_moe" and .quantization_config.quant_method == "fp8"' "${model}/config.json" >/dev/null
[[ "$(jq -r '[.weight_map[]] | unique | length' "${model}/model.safetensors.index.json")" == 24 ]]
while read -r shard; do test -s "${model}/${shard}"; done < <(jq -r '[.weight_map[]] | unique[]' "${model}/model.safetensors.index.json")
tokenizer=/scratch/home/ACCOUNT/qwen/framework-comparison/sources/tokenizer/ac9c66cc9b46af7306746a9250f23d47083d689e
test -s "${tokenizer}/tokenizer.json"
sha256sum "${tokenizer}"/* >"${runtime_dir}/source/tokenizer.sha256"
if [[ "$(jq -r .workload "${config}")" == official ]]; then
  tokenizer_cache=${runtime_dir}/tokenizer-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507
  install -d -m 0755 "${tokenizer_cache}/refs" "${tokenizer_cache}/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e"
  for file in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do cp -L "${tokenizer}/${file}" "${tokenizer_cache}/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e/${file}"; done
  printf '%s' ac9c66cc9b46af7306746a9250f23d47083d689e >"${tokenizer_cache}/refs/main"
fi
if [[ "${framework}" == sglang ]]; then
  jq -e '.speculation.algorithm == "EAGLE3" and (.speculation.steps | type) == "number" and .speculation.steps >= 1 and .speculation.draft_tokens == .speculation.steps + 1 and (.speculation.draft_quantization == null or .speculation.draft_quantization == "fp8")' "${config}" >/dev/null
  export HF_HUB_DISABLE_XET=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1
  uvx --from 'huggingface_hub==0.36.0' hf download "$(jq -er '.speculation.draft_repository' "${config}")" config.json README.md model.safetensors --revision "$(jq -er '.speculation.draft_revision' "${config}")" --local-dir "${runtime_dir}/draft" --max-workers 1
  python3 "${GITHUB_WORKSPACE}/research/simulation/prepare_eagle.py" "${runtime_dir}/draft" "${runtime_dir}/source"
  prefill_location="$(jq -er .prefill.expert_location "${config}")"
  if [[ "${prefill_location}" != trivial ]]; then install -m 0644 "${GITHUB_WORKSPACE}/${prefill_location}" "${runtime_dir}/source/prefill-expert-location.json"; fi
  jq -e '.revision == "ff0f0c0999f6c86b21dfd2f5803eebc4ac712224" and .weight_sha256 == "bf226a966868fe413a22e62e471aa1130bd94f17fd2c40d322b1c8b4def1ae86" and .weight_bytes == 1145943008 and .precision == "BF16"' "${runtime_dir}/source/draft-provenance.json" >/dev/null
  cp "${GITHUB_WORKSPACE}/research/simulation/prepare_eagle.py" "${runtime_dir}/source/"
fi
install -m 0755 "${GITHUB_WORKSPACE}/platforms/firmus/framework-comparison/${framework}.sbatch" "${runtime_dir}/scripts/submit.sbatch"
install -m 0755 "${GITHUB_WORKSPACE}/platforms/firmus/framework-comparison/official-aiperf.sh" "${runtime_dir}/scripts/official-aiperf.sh"
install -m 0644 "${GITHUB_WORKSPACE}/scripts/run-aiperf.py" "${runtime_dir}/scripts/run-aiperf.py"
install -m 0755 "${GITHUB_WORKSPACE}/scripts/collect-eagle-acceptance.py" "${runtime_dir}/scripts/"
install -m 0644 "${GITHUB_WORKSPACE}/kernels/fa3-verify/prefill-candidates.json" "${runtime_dir}/source/"
install -m 0644 "${GITHUB_WORKSPACE}/scripts/select-firmus-prefill-fa3.py" "${GITHUB_WORKSPACE}/scripts/select-firmus-prefill-kernels.sh" "${GITHUB_WORKSPACE}/scripts/select-firmus-prefill-megamoe.py" "${runtime_dir}/scripts/"
install -m 0644 "${GITHUB_WORKSPACE}/scripts/check-firmus-pp4-copy-engine.py" "${runtime_dir}/scripts/"
# Independent Decode greedy gate; source changes require a fresh image build.
install -m 0644 "${GITHUB_WORKSPACE}/scripts/select-firmus-decode-greedy.py" "${runtime_dir}/scripts/"
cp "${runtime_dir}/scripts/select-firmus-decode-greedy.py" "${runtime_dir}/source/"
# Round 41 upper-bound probe, separately timed before Prefill model startup.
install -m 0644 "${GITHUB_WORKSPACE}/scripts/probe-firmus-megamoe-rescale.py" "${GITHUB_WORKSPACE}/scripts/run-firmus-megamoe-rescale-probe.sh" "${GITHUB_WORKSPACE}/scripts/launch-firmus-megamoe-rescale-probe.sh" "${runtime_dir}/scripts/"
cp "${runtime_dir}/scripts/probe-firmus-megamoe-rescale.py" "${runtime_dir}/scripts/run-firmus-megamoe-rescale-probe.sh" "${runtime_dir}/scripts/launch-firmus-megamoe-rescale-probe.sh" "${runtime_dir}/source/"
install -m 0644 "${GITHUB_WORKSPACE}/kernels/probes/megamoe-local-ep1-routing.npz" "${runtime_dir}/source/"
# Round-17 P and optional P3 gate staging, independent of other optional gates.
install -m 0644 "${GITHUB_WORKSPACE}/scripts/select-firmus-prefill-fa3-prepack.py" "${runtime_dir}/scripts/"
cp "${runtime_dir}/scripts/select-firmus-prefill-fa3-prepack.py" "${runtime_dir}/source/"

install -m 0644 "${GITHUB_WORKSPACE}/scripts/select-firmus-prefill-fa3-pair.py" "${runtime_dir}/scripts/"
cp "${runtime_dir}/scripts/select-firmus-prefill-fa3-pair.py" "${runtime_dir}/source/"
install -m 0644 "${GITHUB_WORKSPACE}/scripts/check-firmus-megamoe.py" "${GITHUB_WORKSPACE}/scripts/check-firmus-decode-kernels.py" "${GITHUB_WORKSPACE}/scripts/firmus_megamoe_replicas.py" "${GITHUB_WORKSPACE}/scripts/select-firmus-decode-kernels.sh" "${GITHUB_WORKSPACE}/scripts/select-firmus-decode-dense.py" "${GITHUB_WORKSPACE}/scripts/select-firmus-decode-fa3.py" "${GITHUB_WORKSPACE}/scripts/select-firmus-decode-moe-input.py" "${GITHUB_WORKSPACE}/scripts/bench-firmus-sglang-kernels.py" "${GITHUB_WORKSPACE}/scripts/bench-firmus-megamoe.py" "${runtime_dir}/scripts/"
cp "${runtime_dir}/scripts/check-firmus-megamoe.py" "${runtime_dir}/scripts/check-firmus-decode-kernels.py" "${runtime_dir}/scripts/firmus_megamoe_replicas.py" "${runtime_dir}/scripts/select-firmus-decode-kernels.sh" "${runtime_dir}/scripts/select-firmus-decode-dense.py" "${runtime_dir}/scripts/select-firmus-decode-fa3.py" "${runtime_dir}/scripts/select-firmus-decode-moe-input.py" "${runtime_dir}/scripts/bench-firmus-sglang-kernels.py" "${runtime_dir}/scripts/bench-firmus-megamoe.py" "${runtime_dir}/source/"
install -m 0644 "${GITHUB_WORKSPACE}/platforms/firmus/framework-comparison/vllm-proxy.py" "${runtime_dir}/vllm-proxy.py"
install -m 0644 "${GITHUB_WORKSPACE}/scripts/parse-aiperf-result.py" "${GITHUB_WORKSPACE}/scripts/parse-firmus-reference-result.py" "${GITHUB_WORKSPACE}/scripts/summarize-firmus-run.py" "${runtime_dir}/scripts/"
if [[ "${PROFILE}" == true ]]; then
  install -m 0755 "${GITHUB_WORKSPACE}/scripts/exec-profiler.sh" "${GITHUB_WORKSPACE}/scripts/capture-firmus-profile.sh" "${runtime_dir}/scripts/"
  install -m 0644 "${GITHUB_WORKSPACE}/scripts/analyze_prefill_profile.py" "${GITHUB_WORKSPACE}/scripts/validate-firmus-profile.py" "${runtime_dir}/scripts/"
  command -v ar >/dev/null
  package="${runtime_dir}/nsight.deb"
  curl --fail --location --show-error --max-time 600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-systems-2025.3.2_2025.3.2.474-1_amd64.deb -o "${package}"
  printf 'c7cfe27e2250eb91e1a67e7feb5f2c490c7f598e3b3a3d047aff000bc49f9d6b  %s\n' "${package}" | sha256sum --check >"${runtime_dir}/source/nsys-package.sha256"
  install -d -m 0755 "${runtime_dir}/nsys-extract"
  ar p "${package}" data.tar.gz | tar -xz -C "${runtime_dir}/nsys-extract" --wildcards './opt/nvidia/nsight-systems/2025.3.2/bin/*' './opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/*' './opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/*' './opt/nvidia/nsight-systems/2025.3.2/EULA.txt'
  mv "${runtime_dir}/nsys-extract/opt/nvidia/nsight-systems/2025.3.2" "${runtime_dir}/nsys"
  rm -rf -- "${package}" "${runtime_dir}/nsys-extract"
  test -x "${runtime_dir}/nsys/bin/nsys"
  cp "${runtime_dir}/scripts/exec-profiler.sh" "${runtime_dir}/scripts/capture-firmus-profile.sh" "${runtime_dir}/scripts/validate-firmus-profile.py" "${runtime_dir}/scripts/analyze_prefill_profile.py" "${runtime_dir}/source/"
fi
cp "${runtime_dir}/scripts/submit.sbatch" "${runtime_dir}/scripts/official-aiperf.sh" "${runtime_dir}/vllm-proxy.py" "${runtime_dir}/source/"
cp "${runtime_dir}/scripts/run-aiperf.py" "${runtime_dir}/source/"
if [[ "${framework}" == vllm ]]; then
  for worker in prefill-0 prefill-1 prefill-2 prefill-3 decode-0 decode-1; do
    source_cache="/scratch/home/ACCOUNT/qwen/framework-comparison/matched-settings-comparison/home/${worker}/.tensorrt_llm"
    if [[ -d "${source_cache}/cache" ]]; then
      install -d -m 0755 "${runtime_dir}/home/${worker}"
      cp -a --no-preserve=ownership "${source_cache}" "${runtime_dir}/home/${worker}/"
    fi
  done
  (cd "${runtime_dir}" && find home -type f -print0 | sort -z | xargs -0 -r sha256sum) >"${runtime_dir}/source/jit-cache.sha256"
fi
jq -n --arg model_path "${model}" '{schema_version:"qwen3-pd-model/v1",model:{repository:"Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",revision:"e156cb4efae43fbee1a1ab073f946a1377e6b969",path:$model_path,precision:"fp8",shards:24},tokenizer:{repository:"Qwen/Qwen3-235B-A22B-Instruct-2507",revision:"ac9c66cc9b46af7306746a9250f23d47083d689e"}}' >"${runtime_dir}/source/model.json"
jq -n --arg commit "${GITHUB_SHA}" --arg framework "${framework}" --arg workload "$(jq -r .workload "${config}")" --arg run_id "${GITHUB_RUN_ID}" --arg attempt "${GITHUB_RUN_ATTEMPT}" --argjson profile "${PROFILE}" --argjson profile_only "$(jq -r '.profile_only // false' "${config}")" --argjson images "$(jq .images "${config}")" '{schema_version:"qwen3-pd-provenance/v1",cluster:"firmus",commit:$commit,framework:$framework,workflow_run_id:$run_id,workflow_run_attempt:$attempt,profile:$profile,profile_only:$profile_only,normal_benchmark_before_profile:($profile and ($profile_only | not) and $framework == "sglang" and $workload == "official"),workload:$workload,official_submission:false,container:($images[$framework] | {sha256,size_bytes}),client_container:($images.sglang | {sha256,size_bytes}),aiperf:{version:"0.12.0",commit:"be53bf2953d30e46c500e6a80fc1f8b6f84bc718"},cache_source:(if $framework == "vllm" then "matched-settings-comparison/home" else null end)}' >"${runtime_dir}/source/provenance.json"
provision_complete=true
