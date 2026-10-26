#!/usr/bin/env bash
set -euo pipefail
runtime_dir=${1:?runtime directory required}
config=${2:?serving config required}
workspace=${3:?source workspace required}

# Source patches to packages that are compiled or imported at run time (the
# SGLang JIT kernels and Python, the DeepGEMM JIT headers) can run on an
# existing image without a container build: apply them, in order, to a copy of
# the image's package that sglang.sbatch mounts over the original path. Paths
# in the patches are relative to the overlay root (python/sglang/..., or
# deep_gemm/... as under site-packages).
stage_overlay() {
  local name=$1 key=$2 image_path=$3 package_path=$4 patch
  local patches=()
  mapfile -t patches < <(jq -r --arg key "${key}" '.[$key] // [] | .[]' "${config}")
  if (( ${#patches[@]} == 0 )); then return 0; fi
  local root="${runtime_dir}/overlay/${name}" extract="${runtime_dir}/overlay/extract-${name}"
  install -d -m 0755 "${runtime_dir}/overlay"
  unsquashfs -no-progress -no-xattrs -processors 1 -d "${extract}" "${runtime_dir}/sglang.sqsh" "${image_path}"
  chmod -R u+w "${extract}"
  install -d -m 0755 "$(dirname "${root}/${package_path}")" "${runtime_dir}/source/${name}-overlay"
  mv "${extract}/${image_path}" "${root}/${package_path}"
  rm -rf -- "${extract}"
  for patch in "${patches[@]}"; do
    (cd "${root}" && git apply --check "${workspace}/${patch}" && git apply "${workspace}/${patch}")
    cp "${workspace}/${patch}" "${runtime_dir}/source/${name}-overlay/"
  done
  for patch in "${patches[@]}"; do jq -n --arg path "${patch}" --arg sha "$(sha256sum "${workspace}/${patch}" | awk '{print $1}')" '{path:$path,sha256:$sha}'; done | jq -s --arg image "$(jq -r '.container_sha256 // .images.sglang.sha256' "${config}")" --arg mount "/${image_path}" '{base_image_sha256:$image,mount:$mount,patches:.}' >"${runtime_dir}/source/${name}-overlay.json"
}
stage_overlay sglang sglang_overlay_patches opt/sglang/source/python/sglang python/sglang
stage_overlay deep-gemm deep_gemm_overlay_patches opt/sglang/venv/lib/python3.12/site-packages/deep_gemm deep_gemm
