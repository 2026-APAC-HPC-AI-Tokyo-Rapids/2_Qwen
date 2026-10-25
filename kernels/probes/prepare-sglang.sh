#!/usr/bin/env bash
# Build on the login node; the shared one-GPU runner executes the comparison.
set -euo pipefail
: "${RUNTIME_DIR:?}" "${GITHUB_WORKSPACE:?}"
[[ -z "${CANDIDATE_PATCH:-}" ]]
config="${GITHUB_WORKSPACE}/configs/firmus-framework.json"
if [[ ! -e "${RUNTIME_DIR}/sglang.sqsh" ]]; then source "${GITHUB_WORKSPACE}/platforms/genkai/kernel-probe/prepare-runtime.sh"; fi
image="${RUNTIME_DIR}/sglang.sqsh"
package=opt/sglang/venv/lib/python3.12/site-packages/deep_gemm
unsquashfs -no-progress -no-xattrs -processors 1 -d "${RUNTIME_DIR}/image" "${image}" "${package}" opt/apac/sm90-megamoe-build.json opt/sglang/source/python/sglang >"${RUNTIME_DIR}/logs/extract.log" 2>&1
chmod -R u+w "${RUNTIME_DIR}/image"
mv "${RUNTIME_DIR}/image/${package}" "${RUNTIME_DIR}/source/deep_gemm"
mv "${RUNTIME_DIR}/image/opt/apac/sm90-megamoe-build.json" "${RUNTIME_DIR}/source/"
mv "${RUNTIME_DIR}/image/opt/sglang/source/python" "${RUNTIME_DIR}/source/python"
rm -rf -- "${RUNTIME_DIR}/image"
cd "${RUNTIME_DIR}/source"
mapfile -t overlays < <(jq -r '.sglang_overlay_patches[]' "${config}")
for patch in "${overlays[@]}"; do git apply --check "${GITHUB_WORKSPACE}/${patch}"; git apply "${GITHUB_WORKSPACE}/${patch}"; done
install -m 0755 "${GITHUB_WORKSPACE}/kernels/probes/run-sglang.sh" probe
