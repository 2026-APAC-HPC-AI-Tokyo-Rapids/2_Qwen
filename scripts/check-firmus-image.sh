#!/usr/bin/env bash
# Inspect only image metadata on the login CPU, including SquashFS 4.4 hosts.
set -euo pipefail
image=${1:?image required}
destination=${2:?metadata destination required}
workspace=${3:?source workspace required}
extracted=$(mktemp -d "${destination}/.image-metadata.XXXXXXXXXX")
trap 'chmod -R u+w "${extracted}" && rm -rf -- "${extracted}"' EXIT
unsquashfs -no-progress -no-xattrs -processors 1 -d "${extracted}/root" "${image}" opt/apac/sm90-megamoe-build.json opt/apac/sglang-router-build.json opt/apac/manifest.json opt/sglang/venv/lib/python3.12/site-packages/deep_gemm/_C.so
cp "${extracted}/root/opt/apac/sm90-megamoe-build.json" "${destination}/sm90-megamoe-build.json"
cp "${extracted}/root/opt/apac/manifest.json" "${destination}/image-manifest.json"
cp "${extracted}/root/opt/apac/sglang-router-build.json" "${destination}/sglang-router-build.json"
router_patch_sha=$(sha256sum "${workspace}/patches/sglang-router/prefill-load-lifetime.patch" | awk '{print $1}')
router_nodelay_sha=$(sha256sum "${workspace}/patches/sglang-router/no-delay-acceptor.patch" | awk '{print $1}')
router_lock_sha=$(sha256sum "${workspace}/patches/sglang-router/Cargo.lock" | awk '{print $1}')
jq -e --arg patch "${router_patch_sha}" --arg nodelay "${router_nodelay_sha}" --arg lock "${router_lock_sha}" '.version == "0.3.2" and .source_sha256 == "bdbea1d54cce879fb83d2885d01ee3c006f242d0adbbac806f2fb2b91a5fe73d" and .patch_sha256 == $patch and .nodelay_patch_sha256 == $nodelay and .cargo_lock_sha256 == $lock and .prefill_load == "dispatch_through_body_completion"' "${destination}/sglang-router-build.json" >/dev/null
deep_gemm_patches=$(while read -r name; do jq -cn --arg name "${name}" --arg sha "$(sha256sum "${workspace}/patches/deep-gemm/${name}" | awk '{print $1}')" '{($name):$sha}'; done <"${workspace}/patches/deep-gemm/series" | jq -s 'add')
jq -e --argjson patches "${deep_gemm_patches}" --arg header "$(<"${workspace}/patches/deep-gemm/header.sha256")" '.version == "0.1.5" and .header_sha256 == $header and .patches == $patches and (.patches | keys_unsorted) == ($patches | keys_unsorted) and .combine_accumulator == "FP32" and (.compiled_geometries | length == 2) and any(.compiled_geometries[]; .role == "prefill" and .reuse_accum == true and .registers == 112 and .arrival_counter == true) and any(.compiled_geometries[]; .role == "decode" and .swap_ab == true and .stages == 8)' "${destination}/sm90-megamoe-build.json" >/dev/null
# The rebuilt host extension carries the 144-token SwapAB limit; the pristine wheel's _C.so does not, so compare the real file, not only the record.
host_extension_sha=$(sha256sum "${extracted}/root/opt/sglang/venv/lib/python3.12/site-packages/deep_gemm/_C.so" | awk '{print $1}')
host_patch_sha=$(sha256sum "${workspace}/patches/deep-gemm-host/sm90-swap-ab-144.patch" | awk '{print $1}')
jq -e --arg patch "${host_patch_sha}" --arg extension "${host_extension_sha}" '.host_build as $host | $host.source_commit == "1c8138a59b3df5dac50b76b730834ed57982fdfb" and $host.abi == "tvm-ffi" and $host.host_patch_sha256 == $patch and $host.extension_sha256 == $extension and $host.previous_extension_sha256 != $extension and $host.retained_header_sha256 == .original_header_sha256' "${destination}/sm90-megamoe-build.json" >/dev/null
while read -r patch; do
  patch_sha=$(sha256sum "${workspace}/${patch}" | awk '{print $1}')
  jq -e --arg path "${patch}" --arg sha "${patch_sha}" 'any(.applied_patches[]; .path == $path and .sha256 == $sha)' "${destination}/image-manifest.json" >/dev/null
done < <(jq -er '.applied_patches[].path' "${destination}/image-manifest.json")
required=(patches/sglang/firmus-staged-eagle-kv.patch patches/sglang/qwen3-sm90-megamoe.patch patches/sglang/firmus-dp-rank-query.patch patches/sglang/prefill-ready-handoff.patch patches/sglang/deepgemm-activation-layout.patch patches/sglang/rope-fp8-q.patch patches/sglang/qknorm-rope-fp8.patch patches/sglang/prefill-request-aligned-chunks.patch patches/sglang/qknorm-rope-fp8-kv.patch patches/sglang/qwen-topk-padding.patch patches/sglang/qknorm-rope-fp8-token-warps.patch patches/sglang/add-rmsnorm-group-quant.patch patches/sglang/decode-async-admission.patch patches/sglang/dp-total-requests-rotate-ties.patch patches/sglang/megamoe-idle-prefill-geometry.patch patches/sglang-router/prefill-load-lifetime.patch patches/sglang-router/no-delay-acceptor.patch patches/deep-gemm-host/sm90-swap-ab-144.patch)
jq -e --argjson required "$(printf '%s\n' "${required[@]}" | jq -R . | jq -s .)" '[.applied_patches[].path] as $paths | all($required[]; . as $path | ($paths | index($path)) != null)' "${destination}/image-manifest.json" >/dev/null
echo 'PASS: Firmus image metadata, SM90 build and applied patch SHA-256'
