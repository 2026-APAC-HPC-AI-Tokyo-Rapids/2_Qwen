#!/usr/bin/env bash
set -euo pipefail
export BUILDAH_ISOLATION=chroot
export STORAGE_DRIVER=overlay
: "${BUILD_DIR:?}"
: "${CONTAINER_ARCHIVE_DIR:?}"
: "${GITHUB_WORKSPACE:?}"
: "${GITHUB_SHA:?}"
: "${GITHUB_RUN_ID:?}"
[[ "$(id -u)" -eq 0 ]]
storage_root=${BUILD_STORAGE_ROOT:-/var/lib/containers/storage}
run_root=${BUILD_DIR}/runroot
context=${BUILD_DIR}/context
image=localhost/apac-dynamo:${GITHUB_SHA}
test_container=
mounted=false
cleanup() {
  status=$?
  trap - EXIT
  if [[ "${mounted}" == true ]]; then buildah --root "${storage_root}" --runroot "${run_root}" unmount "${test_container}" >/dev/null 2>&1 || true; fi
  if [[ -n "${test_container}" ]]; then buildah --root "${storage_root}" --runroot "${run_root}" rm "${test_container}" >/dev/null 2>&1 || true; fi
  exit "${status}"
}
trap cleanup EXIT
install -d -m 0755 "${context}/containers" "${context}/configs" "${BUILD_DIR}/logs" "${CONTAINER_ARCHIVE_DIR}"
cp "${GITHUB_WORKSPACE}/containers/dynamo.Dockerfile" "${GITHUB_WORKSPACE}/containers/check-dynamo-image.py" "${context}/containers/"
cp "${GITHUB_WORKSPACE}/configs/firmus-dynamo.json" "${context}/configs/"
base_reference=$(jq -er '.dynamo.image | select(test("^nvcr.io/nvidia/ai-dynamo/(vllm|sglang)-runtime@sha256:[0-9a-f]{64}$"))' "${context}/configs/firmus-dynamo.json")
skopeo inspect --raw "docker://${base_reference}" >"${BUILD_DIR}/logs/base-oci-manifest.json"
[[ "sha256:$(sha256sum "${BUILD_DIR}/logs/base-oci-manifest.json" | awk '{print $1}')" == "${base_reference##*@}" ]]
jq -n --arg commit "${GITHUB_SHA}" --arg base "${base_reference}" --arg recipe_blob "$(git -C "${GITHUB_WORKSPACE}" hash-object containers/dynamo.Dockerfile)" --argjson config "$(cat "${context}/configs/firmus-dynamo.json")" '{schema_version:"oci-embedded-manifest/v1",source_commits:[{repository_id:"apac2026-qwen",commit_role:"container_build_definition",oid:$commit}],recipe:{path:"containers/dynamo.Dockerfile",blob_oid:$recipe_blob},base_oci:{reference:$base},dynamo:$config.dynamo,backend:$config.framework}' >"${context}/apac-manifest.json"
buildah --root "${storage_root}" --runroot "${run_root}" bud --isolation chroot --network host --layers --build-arg "BASE_IMAGE=${base_reference}" --file "${context}/containers/dynamo.Dockerfile" --tag "${image}" "${context}" 2>&1 | tee "${BUILD_DIR}/logs/buildah.log"
test_container=$(buildah --root "${storage_root}" --runroot "${run_root}" from "${image}")
buildah --root "${storage_root}" --runroot "${run_root}" run --isolation chroot "${test_container}" python3 -B /opt/apac/check-dynamo-image.py 2>&1 | tee "${BUILD_DIR}/logs/buildah-test.log"
buildah --root "${storage_root}" --runroot "${run_root}" push "${image}" "oci:${BUILD_DIR}/oci:dynamo" 2>&1 | tee "${BUILD_DIR}/logs/buildah-push.log"
skopeo inspect --raw "oci:${BUILD_DIR}/oci:dynamo" >"${BUILD_DIR}/logs/oci-manifest.json"
umoci stat --json --image "${BUILD_DIR}/oci:dynamo" >"${BUILD_DIR}/logs/umoci-stat.json"
jq -e '.history | length > 0' "${BUILD_DIR}/logs/umoci-stat.json" >/dev/null
rootfs=$(buildah --root "${storage_root}" --runroot "${run_root}" mount "${test_container}")
mounted=true
cmp "${rootfs}/opt/apac/manifest.json" "${context}/apac-manifest.json"
image_path=${CONTAINER_ARCHIVE_DIR}/${GITHUB_RUN_ID}.sqsh
mksquashfs "${rootfs}" "${image_path}" -noappend -all-root -no-xattrs -no-duplicates -no-fragments -comp gzip -Xcompression-level 2 -processors 8 -mem 8G 2>&1 | tee "${BUILD_DIR}/logs/mksquashfs.log"
jq -n --arg sha "$(sha256sum "${image_path}" | awk '{print $1}')" --argjson size "$(stat -c '%s' "${image_path}")" --arg commit "${GITHUB_SHA}" --arg base "${base_reference}" --arg oci "$(sha256sum "${BUILD_DIR}/logs/oci-manifest.json" | awk '{print $1}')" --argjson config "$(cat "${context}/configs/firmus-dynamo.json")" '{schema_version:"container-sidecar/v1",container_artifact:{artifact_relative_path:"dynamo.sqsh",sha256:$sha,size_bytes:$size},source_commits:{qwen:$commit,dynamo:$config.dynamo.commit},base_oci:{reference:$base},oci_manifest:{sha256:$oci},backend:$config.framework,dynamo:$config.dynamo}' >"${CONTAINER_ARCHIVE_DIR}/${GITHUB_RUN_ID}.container-sidecar.json"
