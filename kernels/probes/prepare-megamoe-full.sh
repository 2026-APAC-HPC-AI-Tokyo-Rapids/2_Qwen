#!/usr/bin/env bash
# Full EP4 microbench, using the shared Genkai runner with gpu_count=4.
set -euo pipefail
: "${RUNTIME_DIR:?}" "${GITHUB_WORKSPACE:?}"
candidate=${CANDIDATE_PATCH:-}
CANDIDATE_PATCH= bash "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
cd "${RUNTIME_DIR}/source"
header=deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh
[[ "$(sha256sum "${header}" | awk '{print $1}')" == "$(jq -er .header_sha256 sm90-megamoe-build.json)" ]]
mapfile -t overlays < <(jq -r '.deep_gemm_overlay_patches[]' "${GITHUB_WORKSPACE}/configs/firmus-framework.json")
for patch in "${overlays[@]}"; do git apply --check "${GITHUB_WORKSPACE}/${patch}"; git apply "${GITHUB_WORKSPACE}/${patch}"; done
mkdir baseline
mv deep_gemm baseline/
if [[ -n "${candidate}" ]]; then
  [[ "${candidate}" =~ ^patches/[a-zA-Z0-9_./-]+\.patch$ && "${candidate}" != *..* && -f "${GITHUB_WORKSPACE}/${candidate}" ]]
  cp -a baseline candidate
  (cd candidate; git apply --check "${GITHUB_WORKSPACE}/${candidate}"; git apply "${GITHUB_WORKSPACE}/${candidate}")
fi
cp -a baseline timeline
(cd timeline; git apply --check "${GITHUB_WORKSPACE}/patches/deep-gemm-timeline/sm90-mega-moe-prefill-2wg.patch"; git apply "${GITHUB_WORKSPACE}/patches/deep-gemm-timeline/sm90-mega-moe-prefill-2wg.patch")
for script in bench-firmus-megamoe.py firmus_megamoe_replicas.py summarize-megamoe-kernel-bench.py; do cp "${GITHUB_WORKSPACE}/scripts/${script}" .; done
cp "${GITHUB_WORKSPACE}/kernels/probes/megamoe-full.py" bench.py
jq -n --arg commit "${GITHUB_SHA}" --arg candidate "${candidate}" '{commit:$commit,gpu_count:4,candidate_patch:$candidate,scope:"full EP4 Prefill kernel, 140 physical experts, measured routing frequencies; normal versus instrumented output equality and timeline; no serving score"}' > provenance.json
