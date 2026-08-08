#!/usr/bin/env bash
set -euo pipefail

: "${BUILD_DIR:?}"
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${GITHUB_RUN_NUMBER:?}"
: "${GITHUB_WORKSPACE:?}"
: "${GIT_MIRROR_DIR:?}"
: "${SIF_ARCHIVE_DIR:?}"

[[ "${GITHUB_RUN_ID}" =~ ^[0-9]+$ ]]
[[ "${GITHUB_RUN_ATTEMPT}" =~ ^[0-9]+$ ]]
[[ "${GITHUB_RUN_NUMBER}" =~ ^[0-9]+$ ]]

ROOTFS_DIR="${BUILD_DIR}/sglang-rootfs"
PUBLISH_TMP="${SIF_ARCHIVE_DIR}/.${GITHUB_RUN_NUMBER}.sif.${GITHUB_RUN_ID}.${GITHUB_RUN_ATTEMPT}.tmp"
PATCH_MANIFEST="${BUILD_DIR}/patches.sha256"
PROVENANCE_FILE="${BUILD_DIR}/build-provenance.txt"
SGLANG_MIRROR="${GIT_MIRROR_DIR}/5_SGLang_Src.git"
SGLANG_SOURCE_DIR="${BUILD_DIR}/sglang-source"

cleanup() {
  status=$?
  trap - EXIT INT TERM
  rm -f -- "${PUBLISH_TMP}" 2>/dev/null || true
  if [[ -d "${ROOTFS_DIR}" ]]; then
    chmod -R u+rwX "${ROOTFS_DIR}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

rm -rf -- "${BUILD_DIR:?}"
install -d -m 2775 /dev/shm/qwen-sglang-build/cache "${BUILD_DIR}/containers" "${BUILD_DIR}/patches" "${BUILD_DIR}/logs" "${BUILD_DIR}/tmp"
cp "${GITHUB_WORKSPACE}/containers/sglang.def" "${BUILD_DIR}/containers/"
cp -a "${GITHUB_WORKSPACE}/patches/sglang" "${BUILD_DIR}/patches/"

# Clone the organization-managed main branch from the read-only shared mirror.
git clone --no-checkout "${SGLANG_MIRROR}" "${SGLANG_SOURCE_DIR}"
git -C "${SGLANG_SOURCE_DIR}" checkout --detach refs/heads/main

cd "${BUILD_DIR}"

recipe_sha256="$(sha256sum "${GITHUB_WORKSPACE}/containers/sglang.def" | awk '{ print $1 }')"
(
  cd "${GITHUB_WORKSPACE}"
  while IFS= read -r -d '' patch_file
  do
    sha256sum -- "${patch_file}"
  done < <(find patches/sglang -type f -print0 | sort -z)
) > "${PATCH_MANIFEST}"
patches_sha256="$(sha256sum "${PATCH_MANIFEST}" | awk '{ print $1 }')"

module load singularity-ce/4.1.3
export SINGULARITY_CACHEDIR=/dev/shm/qwen-sglang-build/cache
export SINGULARITY_TMPDIR="${BUILD_DIR}/tmp"

singularity --version | tee logs/build.log
command -v mksquashfs | tee -a logs/build.log
singularity build --fakeroot --writable-tmpfs --sandbox "${ROOTFS_DIR}" containers/sglang.def 2>&1 | tee -a logs/build.log
singularity inspect --json "${ROOTFS_DIR}" > "${BUILD_DIR}/inspect-metadata.json"
sglang_commit="$(tr -d '[:space:]' < "${ROOTFS_DIR}/opt/sglang/sglang-commit.txt")"
cuda_base_image="$(awk '$1 == "From:" { print $2; exit }' "${GITHUB_WORKSPACE}/containers/sglang.def")"
{
  printf 'workflow_run_id=%s\n' "${GITHUB_RUN_ID}"
  printf 'workflow_run_attempt=%s\n' "${GITHUB_RUN_ATTEMPT}"
  printf 'workflow_run_number=%s\n' "${GITHUB_RUN_NUMBER}"
  printf 'sglang_commit=%s\n' "${sglang_commit}"
  printf 'recipe_sha256=%s\n' "${recipe_sha256}"
  printf 'patches_sha256=%s\n' "${patches_sha256}"
  printf 'cuda_base_image=%s\n' "${cuda_base_image}"
  printf 'singularity_version=%s\n' "$(singularity --version)"
  printf 'built_at=%s\n' "$(date -u +%FT%TZ)"
} | tee "${PROVENANCE_FILE}" "${BUILD_DIR}/logs/build-provenance.txt"

singularity exec --fakeroot --bind "${BUILD_DIR}:${BUILD_DIR}" "${ROOTFS_DIR}" tar --numeric-owner --xattrs --xattrs-include='*' --acls --sparse -C "${ROOTFS_DIR}" -cf - . |
  mksquashfs - "${BUILD_DIR}/sglang-rootfs.squashfs" -tar -numeric-owner -default-uid 0 -default-gid 0 -default-mode 0755 -noappend -comp gzip -Xcompression-level 2 -processors 32 -mem 64G 2>&1 |
  tee -a logs/build.log

singularity sif new sglang.sif
singularity sif add sglang.sif containers/sglang.def --datatype 1 --groupid 1
singularity sif add sglang.sif "${BUILD_DIR}/inspect-metadata.json" --datatype 6 --groupid 1 --filename inspect-metadata.json
singularity sif add sglang.sif "${PROVENANCE_FILE}" --datatype 1 --groupid 1 --filename build-provenance.txt
singularity sif add sglang.sif "${BUILD_DIR}/sglang-rootfs.squashfs" --datatype 4 --parttype 2 --partfs 1 --partarch 2 --groupid 1
rm -f -- "${BUILD_DIR}/sglang-rootfs.squashfs"

singularity sif list sglang.sif | tee logs/sif-list.txt
singularity inspect --json sglang.sif > logs/inspect.json
singularity test --writable-tmpfs sglang.sif 2>&1 | tee logs/runtime-test.log

install -d -m 2775 "${SIF_ARCHIVE_DIR}"
cp sglang.sif "${PUBLISH_TMP}"
mv -- "${PUBLISH_TMP}" "${SIF_ARCHIVE_DIR}/${GITHUB_RUN_NUMBER}.sif"
echo "Published ${SIF_ARCHIVE_DIR}/${GITHUB_RUN_NUMBER}.sif" | tee -a logs/build.log
