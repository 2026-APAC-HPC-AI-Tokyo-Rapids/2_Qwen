#!/usr/bin/env bash
set -euo pipefail

: "${BUILD_DIR:?}"
: "${GENKAI_HOST:?}"
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${GITHUB_RUN_NUMBER:?}"
: "${GITHUB_WORKSPACE:?}"

[[ "${GITHUB_RUN_ID}" =~ ^[0-9]+$ ]]
[[ "${GITHUB_RUN_ATTEMPT}" =~ ^[0-9]+$ ]]
[[ "${GITHUB_RUN_NUMBER}" =~ ^[0-9]+$ ]]
[[ "${BUILD_DIR}" == "/ssd_raid/build_sglang/run-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" ]]

remote_tmp="/fast/PROJECT/sglang-sifs/.${GITHUB_RUN_NUMBER}.sif.${GITHUB_RUN_ID}.${GITHUB_RUN_ATTEMPT}.tmp"

cleanup() {
  status=$?
  trap - EXIT INT TERM
  ssh "${GENKAI_HOST}" "rm -f -- '${remote_tmp}'" 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

rm -rf -- "${BUILD_DIR}"
install -d -m 2775 /ssd_raid/build_sglang/cache "${BUILD_DIR}/containers" "${BUILD_DIR}/patches" "${BUILD_DIR}/logs" "${BUILD_DIR}/tmp"
cp "${GITHUB_WORKSPACE}/containers/sglang.def" "${BUILD_DIR}/containers/"
cp -a "${GITHUB_WORKSPACE}/patches/sglang" "${BUILD_DIR}/patches/"
cd "${BUILD_DIR}"
export SINGULARITY_CACHEDIR=/ssd_raid/build_sglang/cache
export SINGULARITY_TMPDIR="${BUILD_DIR}/tmp"

singularity --version | tee logs/build.log
singularity build --fakeroot sglang.sif containers/sglang.def 2>&1 | tee -a logs/build.log
singularity inspect --json sglang.sif > logs/inspect.json

ssh "${GENKAI_HOST}" "install -d -m 2775 /fast/PROJECT/sglang-sifs"
scp sglang.sif "${GENKAI_HOST}:${remote_tmp}"
ssh "${GENKAI_HOST}" "mv -- '${remote_tmp}' '/fast/PROJECT/sglang-sifs/${GITHUB_RUN_NUMBER}.sif'"
echo "Published /fast/PROJECT/sglang-sifs/${GITHUB_RUN_NUMBER}.sif" | tee -a logs/build.log
