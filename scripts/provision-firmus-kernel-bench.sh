#!/usr/bin/env bash
# Stage a 1-node SM90 MegaMoE kernel bench on the Firmus login node: the pinned
# serving image (no rebuild), its DeepGEMM package, and header overlays built
# from patches/deep-gemm-bench/*.patch (candidates, sorted) and the bench-only
# timeline patch. Patch failures stop here, before any Slurm submission.
set -euo pipefail

: "${GITHUB_WORKSPACE:?}"
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${GITHUB_SHA:?}"
: "${GITHUB_OUTPUT:?}"
: "${RUNTIME_ROOT:?}"

config="${GITHUB_WORKSPACE}/configs/firmus-framework.json"
image="$(jq -er .images.sglang.path "${config}")"
expected_sha="$(jq -er .images.sglang.sha256 "${config}")"
[[ "$(stat -c '%s' "${image}")" == "$(jq -er .images.sglang.size_bytes "${config}")" ]]
install -d -m 2775 "${RUNTIME_ROOT}"
runtime_dir="$(mktemp -d "${RUNTIME_ROOT}/kernel-bench-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.XXXXXXXXXX")"
printf 'runtime_dir=%s\n' "${runtime_dir}" >>"${GITHUB_OUTPUT}"
install -d -m 0755 "${runtime_dir}/logs" "${runtime_dir}/results" "${runtime_dir}/scripts" "${runtime_dir}/source" "${runtime_dir}/scratch" "${runtime_dir}/overlay" "${runtime_dir}/home"
echo "Checking sglang image SHA-256"
while sleep 15; do echo "container_sha256_check_in_progress=sglang"; done &
progress_pid=$!
trap 'kill "${progress_pid}" 2>/dev/null || true' EXIT
sha256sum "${image}" >"${runtime_dir}/source/sglang-image.sha256"
kill "${progress_pid}" 2>/dev/null || true
wait "${progress_pid}" 2>/dev/null || true
trap - EXIT
[[ "$(awk '{print $1}' "${runtime_dir}/source/sglang-image.sha256")" == "${expected_sha}" ]]
ln -s "${image}" "${runtime_dir}/sglang.sqsh"

# The image's DeepGEMM already carries patches/deep-gemm/*.patch (header SHA in its build record).
package=opt/sglang/venv/lib/python3.12/site-packages/deep_gemm
unsquashfs -no-progress -no-xattrs -processors 1 -d "${runtime_dir}/image" "${image}" "${package}" opt/apac/sm90-megamoe-build.json
chmod -R u+w "${runtime_dir}/image"
cp "${runtime_dir}/image/opt/apac/sm90-megamoe-build.json" "${runtime_dir}/source/"
[[ "$(sha256sum "${runtime_dir}/image/${package}/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh" | awk '{print $1}')" == "$(jq -er .header_sha256 "${runtime_dir}/source/sm90-megamoe-build.json")" ]]

shopt -s nullglob
candidates=("${GITHUB_WORKSPACE}"/patches/deep-gemm-bench/*.patch)
shopt -u nullglob
timeline_patch="${GITHUB_WORKSPACE}/patches/deep-gemm-timeline/sm90-mega-moe-timeline.patch"
# Overlay roots mirror site-packages, where the image applied its DeepGEMM patches.
build_overlay() {
  local name=$1
  shift
  install -d -m 0755 "${runtime_dir}/overlay/${name}"
  cp -a "${runtime_dir}/image/${package}" "${runtime_dir}/overlay/${name}/deep_gemm"
  for patch in "$@"; do
    (cd "${runtime_dir}/overlay/${name}" && git apply --check "${patch}" && git apply "${patch}")
  done
}
if (( ${#candidates[@]} )); then build_overlay candidate "${candidates[@]}"; fi
build_overlay timeline "${candidates[@]}" "${timeline_patch}"
install -d -m 0755 "${runtime_dir}/source/patches"
cp "${candidates[@]}" "${timeline_patch}" "${runtime_dir}/source/patches/"

install -m 0755 "${GITHUB_WORKSPACE}/platforms/firmus/kernel-bench/run.sbatch" "${runtime_dir}/scripts/run.sbatch"
install -m 0644 "${GITHUB_WORKSPACE}/scripts/bench-firmus-megamoe.py" "${GITHUB_WORKSPACE}/scripts/check-firmus-megamoe.py" "${GITHUB_WORKSPACE}/scripts/check-firmus-decode-kernels.py" "${runtime_dir}/scripts/"
cp "${runtime_dir}/scripts/run.sbatch" "${runtime_dir}/scripts/bench-firmus-megamoe.py" "${runtime_dir}/scripts/check-firmus-megamoe.py" "${runtime_dir}/scripts/check-firmus-decode-kernels.py" "${runtime_dir}/source/"
patch_records="$(for patch in "${candidates[@]}"; do jq -n --arg path "${patch#"${GITHUB_WORKSPACE}/"}" --arg sha "$(sha256sum "${patch}" | awk '{print $1}')" '{path:$path,sha256:$sha}'; done | jq -s .)"
jq -n --arg commit "${GITHUB_SHA}" --arg run_id "${GITHUB_RUN_ID}" --arg attempt "${GITHUB_RUN_ATTEMPT}" --argjson image "$(jq .images.sglang "${config}")" --argjson candidates "${patch_records}" --arg timeline_sha "$(sha256sum "${timeline_patch}" | awk '{print $1}')" '{schema_version:"qwen3-megamoe-kernel-bench/v1",cluster:"firmus",commit:$commit,workflow_run_id:$run_id,workflow_run_attempt:$attempt,image:($image | {path,sha256,size_bytes}),candidate_patches:$candidates,timeline_patch_sha256:$timeline_sha}' >"${runtime_dir}/source/provenance.json"
