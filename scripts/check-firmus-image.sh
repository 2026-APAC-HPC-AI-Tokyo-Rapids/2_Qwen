#!/usr/bin/env bash
# Inspect only image metadata on the login CPU, including SquashFS 4.4 hosts.
set -euo pipefail
image=${1:?image required}
destination=${2:?metadata destination required}
workspace=${3:?source workspace required}
extracted=$(mktemp -d "${destination}/.image-metadata.XXXXXXXXXX")
trap 'chmod -R u+w -- "${extracted}" && rm -rf -- "${extracted}"' EXIT
unsquashfs -no-progress -no-xattrs -processors 1 -d "${extracted}/root" "${image}" opt/apac/sm90-megamoe-build.json opt/apac/manifest.json
cp "${extracted}/root/opt/apac/sm90-megamoe-build.json" "${destination}/sm90-megamoe-build.json"
cp "${extracted}/root/opt/apac/manifest.json" "${destination}/image-manifest.json"
native_patch_sha=$(sha256sum "${workspace}/patches/deep-gemm/sm90-late-topk-fp32.patch" | awk '{print $1}')
jq -e --arg sha "${native_patch_sha}" '.version == "0.1.5" and .header_sha256 == "5479807104bf1aeed4cc3ecc8ce57d2a835d68457f9cf4962ef3f15cc176e8d2" and .patch_sha256 == $sha and .combine_accumulator == "FP32" and (.compiled_geometries | length == 3)' "${destination}/sm90-megamoe-build.json" >/dev/null
while read -r patch; do
  patch_sha=$(sha256sum "${workspace}/${patch}" | awk '{print $1}')
  jq -e --arg path "${patch}" --arg sha "${patch_sha}" 'any(.applied_patches[]; .path == $path and .sha256 == $sha)' "${destination}/image-manifest.json" >/dev/null
done < <(jq -er '.applied_patches[].path' "${destination}/image-manifest.json")
jq -e '[.applied_patches[].path] | index("patches/sglang/firmus-staged-eagle-kv.patch") != null and index("patches/sglang/qwen3-sm90-megamoe.patch") != null' "${destination}/image-manifest.json" >/dev/null
echo 'PASS: Firmus image metadata, SM90 build and applied patch SHA-256'
