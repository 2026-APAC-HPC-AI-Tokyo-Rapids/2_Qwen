#!/usr/bin/env bash
set -euo pipefail

: "${BUILD_DIR:?}"
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${GITHUB_RUN_NUMBER:?}"
: "${GITHUB_WORKSPACE:?}"
: "${SIF_ARCHIVE_DIR:?}"

[[ "${GITHUB_RUN_ID}" =~ ^[0-9]+$ ]]
[[ "${GITHUB_RUN_ATTEMPT}" =~ ^[0-9]+$ ]]
[[ "${GITHUB_RUN_NUMBER}" =~ ^[0-9]+$ ]]
[[ "${BUILD_DIR}" == "/dev/shm/qwen-sglang-build/run-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" ]]

publish_tmp="${SIF_ARCHIVE_DIR}/.${GITHUB_RUN_NUMBER}.sif.${GITHUB_RUN_ID}.${GITHUB_RUN_ATTEMPT}.tmp"
rootfs_dir="${BUILD_DIR}/sglang-rootfs"
squashfs_path="${BUILD_DIR}/sglang-rootfs.squashfs"
metadata_path="${BUILD_DIR}/inspect-metadata.json"

cleanup() {
  status=$?
  trap - EXIT INT TERM
  rm -f -- "${publish_tmp}" 2>/dev/null || true
  if [[ -d "${rootfs_dir}" ]]; then
    chmod -R u+rwX "${rootfs_dir}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

rm -rf -- "${BUILD_DIR}"
install -d -m 2775 /dev/shm/qwen-sglang-build/cache "${BUILD_DIR}/containers" "${BUILD_DIR}/patches" "${BUILD_DIR}/logs" "${BUILD_DIR}/tmp"
cp "${GITHUB_WORKSPACE}/containers/sglang.def" "${BUILD_DIR}/containers/"
cp -a "${GITHUB_WORKSPACE}/patches/sglang" "${BUILD_DIR}/patches/"
cd "${BUILD_DIR}"
module load singularity-ce/4.1.3
export SINGULARITY_CACHEDIR=/dev/shm/qwen-sglang-build/cache
export SINGULARITY_TMPDIR="${BUILD_DIR}/tmp"

singularity --version | tee logs/build.log
command -v mksquashfs | tee -a logs/build.log
singularity build --fakeroot --sandbox "${rootfs_dir}" containers/sglang.def 2>&1 | tee -a logs/build.log
singularity inspect --json "${rootfs_dir}" > "${metadata_path}"
singularity exec --fakeroot --bind "${BUILD_DIR}:${BUILD_DIR}" "${rootfs_dir}" tar --numeric-owner --xattrs --xattrs-include='*' --acls --sparse -C "${rootfs_dir}" -cf - . |
  mksquashfs - "${squashfs_path}" -tar -numeric-owner -default-uid 0 -default-gid 0 -default-mode 0755 -noappend -comp gzip -Xcompression-level 2 -processors 32 -mem 64G 2>&1 |
  tee -a logs/build.log

singularity sif new sglang.sif
singularity sif add sglang.sif containers/sglang.def --datatype 1 --groupid 1
singularity sif add sglang.sif "${metadata_path}" --datatype 6 --groupid 1 --filename inspect-metadata.json
singularity sif add sglang.sif "${squashfs_path}" --datatype 4 --parttype 2 --partfs 1 --partarch 2 --groupid 1
rm -f -- "${squashfs_path}"

singularity sif list sglang.sif | tee logs/sif-list.txt
singularity inspect --json sglang.sif > logs/inspect.json
singularity exec sglang.sif /opt/sglang/venv/bin/python -c 'import torch, sglang; print(torch.__version__); print(sglang.__file__)' 2>&1 | tee logs/import-test.log

install -d -m 2775 "${SIF_ARCHIVE_DIR}"
cp sglang.sif "${publish_tmp}"
mv -- "${publish_tmp}" "${SIF_ARCHIVE_DIR}/${GITHUB_RUN_NUMBER}.sif"
echo "Published ${SIF_ARCHIVE_DIR}/${GITHUB_RUN_NUMBER}.sif" | tee -a logs/build.log
