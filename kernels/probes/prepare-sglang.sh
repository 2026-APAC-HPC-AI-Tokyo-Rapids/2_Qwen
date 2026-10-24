#!/usr/bin/env bash
# Build on the login node; the shared one-GPU runner executes the comparison.
set -euo pipefail
: "${RUNTIME_DIR:?}" "${GITHUB_WORKSPACE:?}"
[[ -z "${CANDIDATE_PATCH:-}" ]]
config="${GITHUB_WORKSPACE}/configs/firmus-framework.json"
image="/home/ACCOUNT/share/sglang-containers/$(basename "$(jq -er .images.sglang.path "${config}")")"
[[ "$(stat -c '%s' "${image}")" == "$(jq -er .images.sglang.size_bytes "${config}")" ]]
[[ "$(jq -er .container_artifact.sha256 "${image%.sqsh}.container-sidecar.json")" == "$(jq -er .images.sglang.sha256 "${config}")" ]]
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
install -d "${RUNTIME_DIR}/enroot-library" "${RUNTIME_DIR}/enroot-system"
cp -a --no-preserve=ownership,xattr /home/ACCOUNT/ku60000563/.local/lib/enroot/. "${RUNTIME_DIR}/enroot-library/"
cp -a --no-preserve=ownership,xattr /home/ACCOUNT/ku60000563/.local/etc/enroot/. "${RUNTIME_DIR}/enroot-system/"
sed -i 's/local timeout=100 pid=-1 id=0/local timeout=5000 pid=-1 id=0/' "${RUNTIME_DIR}/enroot-library/runtime.sh"
sed -i 's%--ldconfig=@$(command -v ldconfig.real || command -v ldconfig)%--ldconfig=/sbin/ldconfig%' "${RUNTIME_DIR}/enroot-system/hooks.d/98-nvidia.sh"
if [[ -e "${RUNTIME_DIR}/enroot-system/hooks.d/10-cgroups.sh" ]]; then chmod -x "${RUNTIME_DIR}/enroot-system/hooks.d/10-cgroups.sh"; fi
ln -s "${image}" "${RUNTIME_DIR}/sglang.sqsh"
install -m 0755 "${GITHUB_WORKSPACE}/kernels/probes/run-sglang.sh" probe
