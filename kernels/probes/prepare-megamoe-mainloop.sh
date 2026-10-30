#!/usr/bin/env bash
# Prepare the standalone FC1 mainloop model (megamoe-mainloop.cu) for the shared one-GPU runner.
# Only the CUTLASS/CuTe/DeepGEMM headers come from the pinned serving image; no candidate patch.
set -euo pipefail
: "${RUNTIME_DIR:?}" "${GITHUB_WORKSPACE:?}"
[[ -z "${CANDIDATE_PATCH:-}" ]]
config="${GITHUB_WORKSPACE}/configs/firmus-framework.json"
image="/home/ACCOUNT/share/sglang-containers/$(basename "$(jq -er .images.sglang.path "${config}")")"
sidecar="${image%.sqsh}.container-sidecar.json"
[[ "$(stat -c '%s' "${image}")" == "$(jq -er .images.sglang.size_bytes "${config}")" ]]
[[ "$(jq -er .container_artifact.sha256 "${sidecar}")" == "$(jq -er .images.sglang.sha256 "${config}")" ]]
package=opt/sglang/venv/lib/python3.12/site-packages/deep_gemm
unsquashfs -no-progress -no-xattrs -processors 1 -d "${RUNTIME_DIR}/image" "${image}" "${package}/include" >"${RUNTIME_DIR}/logs/extract.log" 2>&1
chmod -R u+w "${RUNTIME_DIR}/image"
mv "${RUNTIME_DIR}/image/${package}/include" "${RUNTIME_DIR}/source/include"
rm -rf -- "${RUNTIME_DIR}/image"
cd "${RUNTIME_DIR}/source"
cp "${GITHUB_WORKSPACE}/kernels/probes/megamoe-mainloop.cu" .
source "${RUNTIME_DIR}/scripts/build-env.sh"
nvcc --version >toolchain.txt
nvcc --std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -O3 --ptxas-options=-v,-warn-spills -ccbin "${CXX}" -gencode arch=compute_90a,code=sm_90a -Iinclude -L"$(dirname "$(command -v nvcc)")/../lib64/stubs" megamoe-mainloop.cu -lcuda -o probe >"${RUNTIME_DIR}/logs/build.log" 2>&1 || { tail -100 "${RUNTIME_DIR}/logs/build.log"; exit 1; }
cat toolchain.txt "${RUNTIME_DIR}/logs/build.log"
jq -n --arg commit "${GITHUB_SHA}" --arg image "${image}" '{schema_version:"qwen-kernel-probe/v1",cluster:"genkai",kernel:"prefill-megamoe-fc1-mainloop-model",gpu_count:1,commit:$commit,image:$image,scope:"standalone FC1 K loop of the EP1 2WG four-stage tile (and its FC2 length); split-accumulator chunk hand-off, warpgroup stagger, ldmatrix A with RS halves, rescale removal; no routing, epilogue or serving"}' >provenance.json
