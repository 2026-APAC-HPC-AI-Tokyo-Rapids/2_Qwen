#!/usr/bin/env bash
set -euo pipefail

bucket=$1
input_key=$2
output_prefix=$3
github_run_id=$4
github_run_attempt=$5
github_sha=$6
cache_repository=$7
build_root=/opt/qwen-build
input_archive="${build_root}/input.tar.zst"
logs_archive="${build_root}/build-logs.tar.zst"

[[ "${bucket}" =~ ^[a-z0-9.-]+$ && "${input_key}" == inputs/* && "${output_prefix}" == containers/* ]]
[[ "${github_run_id}" =~ ^[1-9][0-9]*$ && "${github_run_attempt}" =~ ^[1-9][0-9]*$ && "${github_sha}" =~ ^[0-9a-f]{40,64}$ ]]
[[ "${cache_repository}" =~ ^[0-9]+[.]dkr[.]ecr[.][a-z0-9-]+[.]amazonaws[.]com/[a-z0-9._/-]+$ ]]

cleanup() {
  status=$?
  trap - EXIT
  if [[ -d "${build_root}/work/logs" ]]; then tar -I 'zstd -3 -T0' -cf "${logs_archive}" -C "${build_root}/work" logs; aws s3 cp "${logs_archive}" "s3://${bucket}/logs/${github_run_id}/${github_run_attempt}/build-logs.tar.zst" --only-show-errors || true; fi
  exit "${status}"
}
trap cleanup EXIT

rm -rf -- "${build_root:?}"
install -d -m 0755 "${build_root}/input" "${build_root}/output" "${build_root}/work"
aws s3 cp "s3://${bucket}/${input_key}" "${input_archive}" --only-show-errors
tar -I zstd -xf "${input_archive}" -C "${build_root}/input"

export BUILD_DIR="${build_root}/work"
export BUILD_STORAGE_ROOT=/var/lib/containers/storage
export CONTAINER_ARCHIVE_DIR="${build_root}/output"
export GITHUB_RUN_ATTEMPT="${github_run_attempt}"
export GITHUB_RUN_ID="${github_run_id}"
export GITHUB_SHA="${github_sha}"
export GITHUB_WORKSPACE="${build_root}/input/qwen"
export GIT_MIRROR_DIR="${build_root}/input/git-mirror"
export BUILD_CACHE_REPOSITORY="${cache_repository}"
git config --global --add safe.directory "${GIT_MIRROR_DIR}/5_SGLang_Src.git"
cache_registry="${cache_repository%%/*}"
cache_region="$(cut -d. -f4 <<<"${cache_registry}")"
aws ecr get-login-password --region "${cache_region}" | buildah login --username AWS --password-stdin "${cache_registry}"
"${GITHUB_WORKSPACE}/scripts/build-sglang-container.sh"

image="${CONTAINER_ARCHIVE_DIR}/${github_run_id}.sqsh"
sidecar="${CONTAINER_ARCHIVE_DIR}/${github_run_id}.container-sidecar.json"
image_sha256="$(jq -er '.container_artifact.sha256' "${sidecar}")"
image_size="$(jq -er '.container_artifact.size_bytes' "${sidecar}")"
[[ "$(sha256sum "${image}" | awk '{print $1}')" == "${image_sha256}" && "$(stat -c '%s' "${image}")" == "${image_size}" ]]
aws s3 cp "${image}" "s3://${bucket}/${output_prefix}/sglang.sqsh" --metadata "sha256=${image_sha256}" --only-show-errors
aws s3 cp "${sidecar}" "s3://${bucket}/${output_prefix}/container-sidecar.json" --only-show-errors
