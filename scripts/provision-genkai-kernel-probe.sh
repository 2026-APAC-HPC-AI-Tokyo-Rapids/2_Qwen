#!/usr/bin/env bash
# Compile the isolated FC1 probe on the login node, without a GPU/container job.
set -euo pipefail
: "${GITHUB_WORKSPACE:?}" "${GITHUB_RUN_ID:?}" "${GITHUB_RUN_ATTEMPT:?}" "${GITHUB_SHA:?}" "${GITHUB_OUTPUT:?}" "${RUNTIME_ROOT:?}"
candidate_patch=${CANDIDATE_PATCH:-}
if [[ -n "${candidate_patch}" ]]; then
  [[ "${candidate_patch}" =~ ^patches/[a-zA-Z0-9_./-]+\.patch$ && "${candidate_patch}" != *..* && -f "${GITHUB_WORKSPACE}/${candidate_patch}" ]]
fi
config="${GITHUB_WORKSPACE}/configs/firmus-framework.json"
image="/home/ACCOUNT/share/sglang-containers/$(basename "$(jq -er .images.sglang.path "${config}")")"
sidecar="${image%.sqsh}.container-sidecar.json"
[[ "$(stat -c '%s' "${image}")" == "$(jq -er .images.sglang.size_bytes "${config}")" ]]
[[ "$(jq -er .container_artifact.sha256 "${sidecar}")" == "$(jq -er .images.sglang.sha256 "${config}")" ]]
install -d -m 2775 "${RUNTIME_ROOT}"
runtime_dir=$(mktemp -d "${RUNTIME_ROOT}/kernel-probe-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.XXXXXXXX")
printf 'runtime_dir=%s\n' "${runtime_dir}" >>"${GITHUB_OUTPUT}"
install -d "${runtime_dir}/logs" "${runtime_dir}/source" "${runtime_dir}/results" "${runtime_dir}/scripts"
package=opt/sglang/venv/lib/python3.12/site-packages/deep_gemm
unsquashfs -no-progress -no-xattrs -processors 1 -d "${runtime_dir}/image" "${image}" "${package}/include" opt/apac/sm90-megamoe-build.json >"${runtime_dir}/logs/extract.log" 2>&1
mv "${runtime_dir}/image/${package}" "${runtime_dir}/source/deep_gemm"
mv "${runtime_dir}/image/opt/apac/sm90-megamoe-build.json" "${runtime_dir}/source/"
rm -rf -- "${runtime_dir}/image"
chmod -R u+w "${runtime_dir}/source"
header=deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh
[[ "$(sha256sum "${runtime_dir}/source/${header}" | awk '{print $1}')" == "$(jq -er .header_sha256 "${runtime_dir}/source/sm90-megamoe-build.json")" ]]
cd "${runtime_dir}/source"
mapfile -t overlays < <(jq -r '.deep_gemm_overlay_patches[]' "${config}")
for patch in "${overlays[@]}"; do
  git apply --check "${GITHUB_WORKSPACE}/${patch}"
  git apply "${GITHUB_WORKSPACE}/${patch}"
  install -D -m 0644 "${GITHUB_WORKSPACE}/${patch}" "applied/${patch}"
done
cp "${header}" baseline.cuh
if [[ -n "${candidate_patch}" ]]; then
  [[ "$(git apply --numstat "${GITHUB_WORKSPACE}/${candidate_patch}" | cut -f3)" == "${header}" ]]
  git apply --check "${GITHUB_WORKSPACE}/${candidate_patch}"
  git apply "${GITHUB_WORKSPACE}/${candidate_patch}"
  cp "${GITHUB_WORKSPACE}/${candidate_patch}" candidate.patch
fi
cp "${header}" candidate.cuh
cp baseline.cuh "${header}"
install -D -m 0644 "${GITHUB_WORKSPACE}/scripts/build-prefill-fc1-probe.py" scripts/build-prefill-fc1-probe.py
install -D -m 0644 "${GITHUB_WORKSPACE}/kernels/probes/prefill_fc1.cu.in" kernels/probes/prefill_fc1.cu.in
python3 scripts/build-prefill-fc1-probe.py baseline.cuh candidate.cuh probe.cu >"${runtime_dir}/logs/generate.log"
module load cuda/12.9.1
nvcc --version > toolchain.txt
/opt/rh/gcc-toolset-13/root/usr/bin/g++ --version >> toolchain.txt
nvcc --std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -O3 --ptxas-options=-v,-warn-spills -ccbin /opt/rh/gcc-toolset-13/root/usr/bin/g++ -gencode arch=compute_90a,code=sm_90a -Ideep_gemm/include -L"$(dirname "$(command -v nvcc)")/../lib64/stubs" probe.cu -lcuda -o probe >"${runtime_dir}/logs/build.log" 2>&1
cuobjdump --dump-sass probe > probe.sass
sha256sum baseline.cuh candidate.cuh probe.cu probe > SHA256SUMS
cp "${sidecar}" container-sidecar.json
cp "${config}" firmus-framework.json
cp "${GITHUB_WORKSPACE}/scripts/provision-genkai-kernel-probe.sh" build.sh
install -m 0755 "${GITHUB_WORKSPACE}/platforms/genkai/kernel-probe/run.pjm" "${runtime_dir}/scripts/run.pjm"
jq -n --arg commit "${GITHUB_SHA}" --arg run "${GITHUB_RUN_ID}" --arg attempt "${GITHUB_RUN_ATTEMPT}" --arg candidate "${candidate_patch}" --arg image "${image}" '{schema_version:"qwen-kernel-probe/v1",cluster:"genkai",kernel:"prefill-fc1",gpu_count:1,commit:$commit,run_id:$run,attempt:$attempt,candidate_patch:$candidate,image:$image,image_check:"sidecar identity, size and extracted header hash",scope:"isolated FC1; no routing, NVLink, FC2, SwiGLU or serving"}' > provenance.json
