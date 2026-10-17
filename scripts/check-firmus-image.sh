#!/usr/bin/env bash
# Inspect only image metadata on the login CPU, including SquashFS 4.4 hosts.
set -euo pipefail
image=${1:?image required}
destination=${2:?metadata destination required}
workspace=${3:?source workspace required}
extracted=$(mktemp -d "${destination}/.image-metadata.XXXXXXXXXX")
trap 'chmod -R u+w "${extracted}" && rm -rf -- "${extracted}"' EXIT
unsquashfs -no-progress -no-xattrs -processors 1 -d "${extracted}/root" "${image}" opt/apac/sm90-megamoe-build.json opt/apac/sglang-router-build.json opt/apac/manifest.json
cp "${extracted}/root/opt/apac/sm90-megamoe-build.json" "${destination}/sm90-megamoe-build.json"
cp "${extracted}/root/opt/apac/manifest.json" "${destination}/image-manifest.json"
cp "${extracted}/root/opt/apac/sglang-router-build.json" "${destination}/sglang-router-build.json"
router_patch_sha=$(sha256sum "${workspace}/patches/sglang-router/prefill-load-lifetime.patch" | awk '{print $1}')
router_lock_sha=$(sha256sum "${workspace}/patches/sglang-router/Cargo.lock" | awk '{print $1}')
jq -e --arg patch "${router_patch_sha}" --arg lock "${router_lock_sha}" '.version == "0.3.2" and .source_sha256 == "bdbea1d54cce879fb83d2885d01ee3c006f242d0adbbac806f2fb2b91a5fe73d" and .patch_sha256 == $patch and .cargo_lock_sha256 == $lock and .prefill_load == "dispatch_through_body_completion"' "${destination}/sglang-router-build.json" >/dev/null
native_patch_sha=$(sha256sum "${workspace}/patches/deep-gemm/sm90-late-topk-fp32.patch" | awk '{print $1}')
pipeline_patch_sha=$(sha256sum "${workspace}/patches/deep-gemm/sm90-fc2-pipeline.patch" | awk '{print $1}')
prefill_patch_sha=$(sha256sum "${workspace}/patches/deep-gemm/sm90-prefill-rescale.patch" | awk '{print $1}')
decode_tiles_patch_sha=$(sha256sum "${workspace}/patches/deep-gemm/sm90-decode-exact-tiles.patch" | awk '{print $1}')
prefill_phase_patch_sha=$(sha256sum "${workspace}/patches/deep-gemm/sm90-prefill-phase.patch" | awk '{print $1}')
swap_ab_hoist_patch_sha=$(sha256sum "${workspace}/patches/deep-gemm/sm90-decode-swap-ab-hoist.patch" | awk '{print $1}')
l1_quant_patch_sha=$(sha256sum "${workspace}/patches/deep-gemm/sm90-decode-swap-ab-l1-quant.patch" | awk '{print $1}')
dispatch_overlap_patch_sha=$(sha256sum "${workspace}/patches/deep-gemm/sm90-dispatch-pull-overlap.patch" | awk '{print $1}')
jq -e --arg hoist "${swap_ab_hoist_patch_sha}" --arg l1_quant "${l1_quant_patch_sha}" --arg dispatch "${dispatch_overlap_patch_sha}" --arg phase "${prefill_phase_patch_sha}" --arg tiles "${decode_tiles_patch_sha}" --arg sha "${native_patch_sha}" --arg pipeline "${pipeline_patch_sha}" --arg prefill "${prefill_patch_sha}" '.version == "0.1.5" and .header_sha256 == "135a43befec61e04c27a51be3ca0e4c8b928669529a830acde134042379e1a6d" and .patch_sha256 == $sha and .fc2_pipeline_patch_sha256 == $pipeline and .prefill_rescale_patch_sha256 == $prefill and .decode_tiles_patch_sha256 == $tiles and .prefill_phase_patch_sha256 == $phase and .swap_ab_hoist_patch_sha256 == $hoist and .l1_quant_patch_sha256 == $l1_quant and .dispatch_overlap_patch_sha256 == $dispatch and .combine_accumulator == "FP32" and (.compiled_geometries | length == 2) and any(.compiled_geometries[]; .role == "prefill" and .reuse_accum == true and .registers == 112 and .arrival_counter == true) and any(.compiled_geometries[]; .role == "decode" and .swap_ab == true and .stages == 8)' "${destination}/sm90-megamoe-build.json" >/dev/null
while read -r patch; do
  patch_sha=$(sha256sum "${workspace}/${patch}" | awk '{print $1}')
  jq -e --arg path "${patch}" --arg sha "${patch_sha}" 'any(.applied_patches[]; .path == $path and .sha256 == $sha)' "${destination}/image-manifest.json" >/dev/null
done < <(jq -er '.applied_patches[].path' "${destination}/image-manifest.json")
jq -e '[.applied_patches[].path] | index("patches/sglang/firmus-staged-eagle-kv.patch") != null and index("patches/sglang/qwen3-sm90-megamoe.patch") != null and index("patches/sglang/firmus-dp-rank-query.patch") != null and index("patches/sglang/prefill-ready-handoff.patch") != null and index("patches/sglang/deepgemm-activation-layout.patch") != null and index("patches/sglang/rope-fp8-q.patch") != null and index("patches/sglang/qknorm-rope-fp8.patch") != null and index("patches/sglang/qknorm-rope-fp8-kv.patch") != null and index("patches/sglang/qwen-topk-padding.patch") != null' "${destination}/image-manifest.json" >/dev/null
echo 'PASS: Firmus image metadata, SM90 build and applied patch SHA-256'
