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

cleanup() {
  status=$?
  trap - EXIT INT TERM
  rm -f -- "${publish_tmp}" 2>/dev/null || true
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
singularity build --fakeroot sglang.sif containers/sglang.def 2>&1 | tee -a logs/build.log
singularity inspect --json sglang.sif > logs/inspect.json

install -d -m 2775 "${SIF_ARCHIVE_DIR}"
cp sglang.sif "${publish_tmp}"
mv -- "${publish_tmp}" "${SIF_ARCHIVE_DIR}/${GITHUB_RUN_NUMBER}.sif"
echo "Published ${SIF_ARCHIVE_DIR}/${GITHUB_RUN_NUMBER}.sif" | tee -a logs/build.log
