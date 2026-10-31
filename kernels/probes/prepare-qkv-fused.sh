#!/usr/bin/env bash
# One H100, exact pinned PP4 image and overlays. No candidate_patch or serving.
set -euo pipefail
: "${RUNTIME_DIR:?}" "${GITHUB_WORKSPACE:?}"
[[ -z "${CANDIDATE_PATCH:-}" ]]
config="${GITHUB_WORKSPACE}/configs/firmus-prefill-pp4.json"
image="/home/ACCOUNT/share/sglang-containers/$(basename "$(jq -er .images.sglang.path "${config}")")"
[[ "$(stat -c '%s' "${image}")" == "$(jq -er .images.sglang.size_bytes "${config}")" ]]
[[ "$(jq -er .container_artifact.sha256 "${image%.sqsh}.container-sidecar.json")" == "$(jq -er .images.sglang.sha256 "${config}")" ]]
[[ -x /usr/local/cuda-13.4/bin/nvcc ]]
install -d "${RUNTIME_DIR}/enroot-library" "${RUNTIME_DIR}/enroot-system"
cp -a --no-preserve=ownership,xattr /home/ACCOUNT/ku60000563/.local/lib/enroot/. "${RUNTIME_DIR}/enroot-library/"
cp -a --no-preserve=ownership,xattr /home/ACCOUNT/ku60000563/.local/etc/enroot/. "${RUNTIME_DIR}/enroot-system/"
sed -i 's/local timeout=100 pid=-1 id=0/local timeout=5000 pid=-1 id=0/' "${RUNTIME_DIR}/enroot-library/runtime.sh"
sed -i 's%--ldconfig=@$(command -v ldconfig.real || command -v ldconfig)%--ldconfig=/sbin/ldconfig%' "${RUNTIME_DIR}/enroot-system/hooks.d/98-nvidia.sh"
if [[ -e "${RUNTIME_DIR}/enroot-system/hooks.d/10-cgroups.sh" ]]; then chmod -x "${RUNTIME_DIR}/enroot-system/hooks.d/10-cgroups.sh"; fi
ln -s "${image}" "${RUNTIME_DIR}/sglang.sqsh"
package=opt/sglang/venv/lib/python3.12/site-packages/deep_gemm
unsquashfs -no-progress -no-xattrs -processors 1 -d "${RUNTIME_DIR}/image" "${image}" "${package}" opt/sglang/source/python/sglang >"${RUNTIME_DIR}/logs/extract.log" 2>&1
chmod -R u+w "${RUNTIME_DIR}/image"
mv "${RUNTIME_DIR}/image/${package}" "${RUNTIME_DIR}/source/deep_gemm"
mv "${RUNTIME_DIR}/image/opt/sglang/source/python" "${RUNTIME_DIR}/source/python"
rm -rf -- "${RUNTIME_DIR}/image"
cd "${RUNTIME_DIR}/source"
mapfile -t overlays < <(jq -r '.sglang_overlay_patches[]' "${config}")
for patch in "${overlays[@]}"; do git apply --check "${GITHUB_WORKSPACE}/${patch}"; git apply "${GITHUB_WORKSPACE}/${patch}"; done
mapfile -t overlays < <(jq -r '.deep_gemm_overlay_patches[]' "${config}")
for patch in "${overlays[@]}"; do git apply --check "${GITHUB_WORKSPACE}/${patch}"; git apply "${GITHUB_WORKSPACE}/${patch}"; done
install -m 0644 "${GITHUB_WORKSPACE}/kernels/probes/qkv-fused.py" bench.py
install -m 0644 "${GITHUB_WORKSPACE}/kernels/fa3-verify/candidates.json" fa3-verify-candidates.json
cat >probe <<'PROBE'
#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?}"
export ENROOT_LIBRARY_PATH="${RUNTIME_DIR}/enroot-library" ENROOT_SYSCONF_PATH="${RUNTIME_DIR}/enroot-system" ENROOT_MOUNT_HOME=no
node_tmp=$(mktemp -d "${TMPDIR:-/tmp}/qkv-fused.XXXXXXXX")
trap 'rm -rf -- "${node_tmp:?}"' EXIT
export ENROOT_CACHE_PATH="${node_tmp}/cache" ENROOT_DATA_PATH="${node_tmp}/data" ENROOT_RUNTIME_PATH="${node_tmp}/runtime" ENROOT_TEMP_PATH="${node_tmp}/temp"
install -d "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}" "${RUNTIME_DIR}/cache" "${RUNTIME_DIR}/home"
enroot start -m /usr/local/cuda-13.4:/usr/local/cuda-13.4 -m "${RUNTIME_DIR}:${RUNTIME_DIR}" -m "${RUNTIME_DIR}/home:${HOME}" -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_DISABLE_REQUIRE=1 -e "PYTHONPATH=${RUNTIME_DIR}/source/python:${RUNTIME_DIR}/source" -e "TVM_FFI_CACHE_DIR=${RUNTIME_DIR}/cache/tvm" -e "DG_JIT_CACHE_DIR=${RUNTIME_DIR}/cache/deepgemm" -e "TRITON_CACHE_DIR=${RUNTIME_DIR}/cache/triton" -e CUDA_HOME=/usr/local/cuda-13.4 -e CUDACXX=/usr/local/cuda-13.4/bin/nvcc -e "PATH=/usr/local/cuda-13.4/bin:/opt/sglang/venv/bin:/usr/bin:/bin" -e "LD_LIBRARY_PATH=/usr/local/cuda-13.4/lib64" -e DG_JIT_PTXAS_VERBOSE=1 -e DG_PRINT_CONFIGS=1 "${RUNTIME_DIR}/sglang.sqsh" /opt/sglang/venv/bin/python "${RUNTIME_DIR}/source/bench.py"
PROBE
chmod 0755 probe
jq -n --arg commit "${GITHUB_SHA}" --arg image "${image}" --arg sha256 "$(jq -r .images.sglang.sha256 "${config}")" '{schema_version:"qwen-kernel-probe/v1",calibration:true,commit:$commit,image:$image,image_sha256:$sha256,kernel:"PP4 QKV direct-FP32 epilogue",gpu_count:1,toolchain:"CUDA 13.4 sm_90a, control and fused",scope:"all outputs, FP64 reference, paired interleaved timing, 300-call endurance, no serving"}' >provenance.json
