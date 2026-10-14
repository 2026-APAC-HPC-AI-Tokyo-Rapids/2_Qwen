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
image="localhost/apac-qwen-vllm:${GITHUB_SHA}"
docker_tag="apac-qwen-vllm:${GITHUB_SHA}"
squashfs_work="${BUILD_DIR}/vllm.sqsh"
docker_work="${BUILD_DIR}/vllm.docker.tar"
archive="${CONTAINER_ARCHIVE_DIR}/${GITHUB_RUN_ID}.sqsh"
docker_archive="${CONTAINER_ARCHIVE_DIR}/${GITHUB_RUN_ID}.docker.tar.zst"
sidecar_archive="${CONTAINER_ARCHIVE_DIR}/${GITHUB_RUN_ID}.container-sidecar.json"
archive_temporary="${CONTAINER_ARCHIVE_DIR}/.${GITHUB_RUN_ID}.sqsh.${GITHUB_RUN_ATTEMPT}.tmp"
docker_temporary="${CONTAINER_ARCHIVE_DIR}/.${GITHUB_RUN_ID}.docker.tar.zst.${GITHUB_RUN_ATTEMPT}.tmp"
sidecar_temporary="${CONTAINER_ARCHIVE_DIR}/.${GITHUB_RUN_ID}.container-sidecar.json.${GITHUB_RUN_ATTEMPT}.tmp"
test_container=
mounted_container=

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${mounted_container}" ]]; then buildah --root "${storage_root}" --runroot "${run_root}" unmount "${mounted_container}" >/dev/null 2>&1 || true; fi
  if [[ -n "${test_container}" ]]; then buildah --root "${storage_root}" --runroot "${run_root}" rm "${test_container}" >/dev/null 2>&1 || true; fi
  rm -f -- "${archive_temporary}" "${docker_temporary}" "${sidecar_temporary}" "${squashfs_work}" "${docker_work}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

install -d -m 2775 "${BUILD_DIR}/logs" "${context}/containers" "${CONTAINER_ARCHIVE_DIR}" "${storage_root}"
storage_fstype="$(stat -f -c '%T' "${storage_root}")"
case "${storage_fstype}" in ext2/ext3|xfs) :;; *) printf 'Unsupported Buildah storage filesystem %s: %s\n' "${storage_fstype}" "${storage_root}" >&2; exit 1;; esac
cp "${GITHUB_WORKSPACE}/containers/vllm.Dockerfile" "${GITHUB_WORKSPACE}/containers/check-vllm-image.py" "${GITHUB_WORKSPACE}/containers/fix-vllm-dense-draft-ep.py" "${context}/containers/"
base_reference="$(sed -n 's/^FROM //p' "${context}/containers/vllm.Dockerfile")"
base_manifest_sha256="$(skopeo inspect --raw "docker://${base_reference}" | tee "${BUILD_DIR}/logs/base-oci-manifest.json" | sha256sum | awk '{print $1}')"
qwen_object_format="$(git -C "${GITHUB_WORKSPACE}" rev-parse --show-object-format)"
dockerfile_blob_oid="$(git -C "${GITHUB_WORKSPACE}" hash-object containers/vllm.Dockerfile)"
check_blob_oid="$(git -C "${GITHUB_WORKSPACE}" hash-object containers/check-vllm-image.py)"
repair_blob_oid="$(git -C "${GITHUB_WORKSPACE}" hash-object containers/fix-vllm-dense-draft-ep.py)"
jq -n --arg qwen_object_format "${qwen_object_format}" --arg qwen_commit "${GITHUB_SHA}" --arg dockerfile_blob_oid "${dockerfile_blob_oid}" --arg check_blob_oid "${check_blob_oid}" --arg repair_blob_oid "${repair_blob_oid}" --arg base_reference "${base_reference}" --arg base_manifest_sha256 "${base_manifest_sha256}" '{schema_version:"oci-embedded-manifest/v1",source_commits:[{repository_id:"apac2026-qwen",commit_role:"container_build_definition",object_hash_algorithm:$qwen_object_format,oid:$qwen_commit}],recipe:{path:"containers/vllm.Dockerfile",blob_oid:$dockerfile_blob_oid,build_steps:[{path:"containers/check-vllm-image.py",blob_oid:$check_blob_oid}]},applied_patches:[{path:"containers/fix-vllm-dense-draft-ep.py",blob_oid:$repair_blob_oid}],base_oci:{reference:$base_reference,manifest_sha256:$base_manifest_sha256}}' >"${context}/apac-manifest.json"

buildah --root "${storage_root}" --runroot "${run_root}" bud --isolation chroot --network host --layers --cache-from "${BUILD_CACHE_REPOSITORY}" --cache-to "${BUILD_CACHE_REPOSITORY}" --cache-ttl 336h --file "${context}/containers/vllm.Dockerfile" --tag "${image}" "${context}" 2>&1 | tee "${BUILD_DIR}/logs/buildah.log"
test_container="$(buildah --root "${storage_root}" --runroot "${run_root}" from "${image}")"
buildah --root "${storage_root}" --runroot "${run_root}" run --isolation chroot "${test_container}" python3 -B /opt/apac/check-vllm-image.py 2>&1 | tee "${BUILD_DIR}/logs/buildah-test.log"
buildah --root "${storage_root}" --runroot "${run_root}" push "${image}" "oci:${oci_build}:vllm" 2>&1 | tee "${BUILD_DIR}/logs/buildah-push.log"
skopeo copy "oci:${oci_build}:vllm" "oci:${oci_runtime}:vllm" 2>&1 | tee "${BUILD_DIR}/logs/skopeo-copy.log"
skopeo inspect --raw "oci:${oci_runtime}:vllm" >"${BUILD_DIR}/logs/oci-manifest.json"
umoci stat --json --image "${oci_runtime}:vllm" | tee "${BUILD_DIR}/logs/umoci-stat.json"
jq -e '.history | length > 0' "${BUILD_DIR}/logs/umoci-stat.json" >/dev/null
skopeo copy "oci:${oci_runtime}:vllm" "docker-archive:${docker_work}:${docker_tag}" 2>&1 | tee "${BUILD_DIR}/logs/docker-archive.log"
skopeo inspect "docker-archive:${docker_work}" >"${BUILD_DIR}/logs/docker-inspect.json"
zstd -3 -T0 -c "${docker_work}" >"${docker_temporary}"
zstd -t "${docker_temporary}"

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
docker_sha256="$(sha256sum "${docker_temporary}" | awk '{print $1}')"
docker_size="$(stat -c '%s' "${docker_temporary}")"
oci_manifest_sha256="$(sha256sum "${BUILD_DIR}/logs/oci-manifest.json" | awk '{print $1}')"
jq -n --argjson size_bytes "${container_size}" --arg sha256 "${container_sha256}" --argjson docker_size_bytes "${docker_size}" --arg docker_sha256 "${docker_sha256}" --arg docker_tag "${docker_tag}" --arg oci_manifest_sha256 "${oci_manifest_sha256}" --arg qwen_commit "${GITHUB_SHA}" --arg base_reference "${base_reference}" --arg buildah_version "$(buildah --version)" --arg skopeo_version "$(skopeo --version)" --arg umoci_version "$(umoci --version | head -n 1)" '{schema_version:"container-sidecar/v1",container_artifact:{artifact_relative_path:"vllm.sqsh",size_bytes:$size_bytes,sha256:$sha256},docker_artifact:{artifact_relative_path:"vllm.docker.tar.zst",size_bytes:$docker_size_bytes,sha256:$docker_sha256,tag:$docker_tag},oci_manifest:{sha256:$oci_manifest_sha256},source_commits:{qwen:$qwen_commit},base_oci:{reference:$base_reference},toolchain:{buildah:$buildah_version,skopeo:$skopeo_version,umoci:$umoci_version}}' >"${sidecar_temporary}"
cp --reflink=auto --sparse=always "${squashfs_work}" "${archive_temporary}"
[[ "$(stat -c '%s' "${archive_temporary}")" == "${container_size}" && "$(sha256sum "${archive_temporary}" | awk '{print $1}')" == "${container_sha256}" ]]
mv -- "${archive_temporary}" "${archive}"
mv -- "${docker_temporary}" "${docker_archive}"
mv -- "${sidecar_temporary}" "${sidecar_archive}"
printf 'Published %s and %s\n' "${archive}" "${docker_archive}"
