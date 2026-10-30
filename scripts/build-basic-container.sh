#!/usr/bin/env bash
set -euo pipefail

export BUILDAH_ISOLATION=chroot
export STORAGE_DRIVER=overlay

: "${BUILD_DIR:?build directory is required}"
: "${BUILD_CACHE_REPOSITORY:?Buildah cache repository is required}"
: "${CONTAINER_ARCHIVE_DIR:?container archive directory is required}"
: "${GITHUB_RUN_ATTEMPT:?workflow run attempt is required}"
: "${GITHUB_RUN_ID:?workflow run ID is required}"
: "${GITHUB_SHA:?workflow commit is required}"
: "${GITHUB_WORKSPACE:?workflow workspace is required}"
[[ "${GITHUB_RUN_ID}" =~ ^[1-9][0-9]*$ && "${GITHUB_RUN_ATTEMPT}" =~ ^[1-9][0-9]*$ && "${GITHUB_SHA}" =~ ^[0-9a-f]{40,64}$ ]]
[[ "$(id -u)" -eq 0 ]]

storage_root="${BUILD_STORAGE_ROOT:-/var/lib/containers/storage}"
run_root="${BUILD_DIR}/runroot"
context="${BUILD_DIR}/context"
oci_build="${BUILD_DIR}/oci-build"
oci_runtime="${BUILD_DIR}/oci-runtime"
image="localhost/apac-qwen-basic:${GITHUB_SHA}"
squashfs_work="${BUILD_DIR}/basic.sqsh"
archive="${CONTAINER_ARCHIVE_DIR}/${GITHUB_RUN_ID}.sqsh"
sidecar_archive="${CONTAINER_ARCHIVE_DIR}/${GITHUB_RUN_ID}.container-sidecar.json"
archive_temporary="${CONTAINER_ARCHIVE_DIR}/.${GITHUB_RUN_ID}.sqsh.${GITHUB_RUN_ATTEMPT}.tmp"
sidecar_temporary="${CONTAINER_ARCHIVE_DIR}/.${GITHUB_RUN_ID}.container-sidecar.json.${GITHUB_RUN_ATTEMPT}.tmp"
test_container=
mounted_container=

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${mounted_container}" ]]; then buildah --root "${storage_root}" --runroot "${run_root}" unmount "${mounted_container}" >/dev/null 2>&1 || true; fi
  if [[ -n "${test_container}" ]]; then buildah --root "${storage_root}" --runroot "${run_root}" rm "${test_container}" >/dev/null 2>&1 || true; fi
  rm -f -- "${archive_temporary}" "${sidecar_temporary}" "${squashfs_work}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

install -d -m 2775 "${BUILD_DIR}/logs" "${context}/containers" "${CONTAINER_ARCHIVE_DIR}" "${storage_root}"
storage_fstype="$(stat -f -c '%T' "${storage_root}")"
case "${storage_fstype}" in ext2/ext3|xfs) :;; *) printf 'Unsupported Buildah storage filesystem %s: %s\n' "${storage_fstype}" "${storage_root}" >&2; exit 1;; esac
cp "${GITHUB_WORKSPACE}/containers/basic.Dockerfile" "${GITHUB_WORKSPACE}/containers/check-basic-image.py" "${context}/containers/"
base_reference="$(sed -n 's/^FROM //p' "${context}/containers/basic.Dockerfile")"
base_manifest_sha256="$(skopeo inspect --raw "docker://${base_reference}" | tee "${BUILD_DIR}/logs/base-oci-manifest.json" | sha256sum | awk '{print $1}')"
qwen_object_format="$(git -C "${GITHUB_WORKSPACE}" rev-parse --show-object-format)"
dockerfile_blob_oid="$(git -C "${GITHUB_WORKSPACE}" hash-object containers/basic.Dockerfile)"
check_blob_oid="$(git -C "${GITHUB_WORKSPACE}" hash-object containers/check-basic-image.py)"
jq -n --arg qwen_object_format "${qwen_object_format}" --arg qwen_commit "${GITHUB_SHA}" --arg dockerfile_blob_oid "${dockerfile_blob_oid}" --arg check_blob_oid "${check_blob_oid}" --arg base_reference "${base_reference}" --arg base_manifest_sha256 "${base_manifest_sha256}" '{schema_version:"oci-embedded-manifest/v1",source_commits:[{repository_id:"apac2026-qwen",commit_role:"container_build_definition",object_hash_algorithm:$qwen_object_format,oid:$qwen_commit}],recipe:{path:"containers/basic.Dockerfile",blob_oid:$dockerfile_blob_oid,build_steps:[{path:"containers/check-basic-image.py",blob_oid:$check_blob_oid}]},tier:"basic",applied_patches:[],base_oci:{reference:$base_reference,manifest_sha256:$base_manifest_sha256}}' >"${context}/apac-manifest.json"

buildah --root "${storage_root}" --runroot "${run_root}" bud --isolation chroot --network host --layers --cache-from "${BUILD_CACHE_REPOSITORY}" --cache-to "${BUILD_CACHE_REPOSITORY}" --cache-ttl 336h --file "${context}/containers/basic.Dockerfile" --tag "${image}" "${context}" 2>&1 | tee "${BUILD_DIR}/logs/buildah.log"
test_container="$(buildah --root "${storage_root}" --runroot "${run_root}" from "${image}")"
buildah --root "${storage_root}" --runroot "${run_root}" run --isolation chroot "${test_container}" /opt/sglang/bin/python -B /opt/apac/check-basic-image.py 2>&1 | tee "${BUILD_DIR}/logs/buildah-test.log"
buildah --root "${storage_root}" --runroot "${run_root}" run --isolation chroot "${test_container}" /opt/aiperf/bin/python -c 'from importlib.metadata import version; assert version("aiperf") == "0.12.0"; assert version("sglang-router") == "0.3.2"'
buildah --root "${storage_root}" --runroot "${run_root}" push "${image}" "oci:${oci_build}:basic" 2>&1 | tee "${BUILD_DIR}/logs/buildah-push.log"
skopeo copy "oci:${oci_build}:basic" "oci:${oci_runtime}:basic" 2>&1 | tee "${BUILD_DIR}/logs/skopeo-copy.log"
skopeo inspect --raw "oci:${oci_runtime}:basic" >"${BUILD_DIR}/logs/oci-manifest.json"
umoci stat --json --image "${oci_runtime}:basic" | tee "${BUILD_DIR}/logs/umoci-stat.json"
jq -e '.history | length > 0' "${BUILD_DIR}/logs/umoci-stat.json" >/dev/null

rootfs="$(buildah --root "${storage_root}" --runroot "${run_root}" mount "${test_container}")"
mounted_container="${test_container}"
cmp "${rootfs}/opt/apac/manifest.json" "${context}/apac-manifest.json"
mksquashfs "${rootfs}" "${squashfs_work}" -noappend -all-root -no-xattrs -no-duplicates -no-fragments -comp gzip -Xcompression-level 2 -processors 8 -mem 8G 2>&1 | tee "${BUILD_DIR}/logs/mksquashfs.log"
buildah --root "${storage_root}" --runroot "${run_root}" unmount "${test_container}"
mounted_container=
buildah --root "${storage_root}" --runroot "${run_root}" rm "${test_container}"
test_container=

container_sha256="$(sha256sum "${squashfs_work}" | awk '{print $1}')"
container_size="$(stat -c '%s' "${squashfs_work}")"
oci_manifest_sha256="$(sha256sum "${BUILD_DIR}/logs/oci-manifest.json" | awk '{print $1}')"
jq -n --argjson size_bytes "${container_size}" --arg sha256 "${container_sha256}" --arg oci_manifest_sha256 "${oci_manifest_sha256}" --arg qwen_commit "${GITHUB_SHA}" --arg base_reference "${base_reference}" --arg buildah_version "$(buildah --version)" --arg skopeo_version "$(skopeo --version)" --arg umoci_version "$(umoci --version | head -n 1)" '{schema_version:"container-sidecar/v1",container_artifact:{artifact_relative_path:"basic.sqsh",size_bytes:$size_bytes,sha256:$sha256},oci_manifest:{sha256:$oci_manifest_sha256},tier:"basic",applied_patches:[],source_commits:{qwen:$qwen_commit,sglang:"297e54fd105e974cdf401e5b4d20253a22c16f90"},base_oci:{reference:$base_reference},toolchain:{buildah:$buildah_version,skopeo:$skopeo_version,umoci:$umoci_version}}' >"${sidecar_temporary}"
cp --reflink=auto --sparse=always "${squashfs_work}" "${archive_temporary}"
[[ "$(stat -c '%s' "${archive_temporary}")" == "${container_size}" && "$(sha256sum "${archive_temporary}" | awk '{print $1}')" == "${container_sha256}" ]]
mv -- "${archive_temporary}" "${archive}"
mv -- "${sidecar_temporary}" "${sidecar_archive}"
printf 'Published %s\n' "${archive}"
