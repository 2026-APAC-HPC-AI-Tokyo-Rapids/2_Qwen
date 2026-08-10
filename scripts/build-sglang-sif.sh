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
METADATA_PUBLISH_TMP="${SIF_ARCHIVE_DIR}/.${GITHUB_RUN_NUMBER}.metadata.${GITHUB_RUN_ID}.${GITHUB_RUN_ATTEMPT}.tmp"
METADATA_ARCHIVE="${SIF_ARCHIVE_DIR}/${GITHUB_RUN_NUMBER}.metadata"
PATCH_MANIFEST="${BUILD_DIR}/patches.sha256"
PROVENANCE_FILE="${BUILD_DIR}/build-provenance.txt"
SGLANG_MIRROR="${GIT_MIRROR_DIR}/5_SGLang_Src.git"
SGLANG_SOURCE_DIR="${BUILD_DIR}/sglang-source"

cleanup() {
  status=$?
  trap - EXIT INT TERM
  rm -f -- "${PUBLISH_TMP}" 2>/dev/null || true
  if [[ -d "${METADATA_PUBLISH_TMP}" ]]; then
    rm -rf -- "${METADATA_PUBLISH_TMP:?}"
  fi
  if [[ -d "${ROOTFS_DIR}" ]]; then
    chmod -R u+rwX "${ROOTFS_DIR}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

rm -rf -- "${BUILD_DIR:?}"
install -d -m 2775 /dev/shm/qwen-sglang-build/cache "${BUILD_DIR}/containers" "${BUILD_DIR}/patches" "${BUILD_DIR}/logs" "${BUILD_DIR}/tmp" "${BUILD_DIR}/sif-metadata/sif-build/toolchain"
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

base_reference="$(awk '$1 == "From:" { print $2; exit }' "${GITHUB_WORKSPACE}/containers/sglang.def")"
uv run --project "${GITHUB_WORKSPACE}/qwen-experiment-metrics" --frozen --no-dev \
  python "${GITHUB_WORKSPACE}/scripts/resolve-oci.py" "${base_reference}" \
  --raw-output "${BUILD_DIR}/sif-metadata/sif-build/oci-inspect.raw.json" \
  >"${BUILD_DIR}/sif-metadata/base-oci.json"
resolved_base_reference="$(jq -er '.reference' "${BUILD_DIR}/sif-metadata/base-oci.json")"
[[ "${base_reference}" == "${resolved_base_reference}" ]]

qwen_object_format="$(git -C "${GITHUB_WORKSPACE}" rev-parse --show-object-format)"
sglang_object_format="$(git -C "${SGLANG_SOURCE_DIR}" rev-parse --show-object-format)"
qwen_commit="$(git -C "${GITHUB_WORKSPACE}" rev-parse HEAD)"
sglang_source_commit="$(git -C "${SGLANG_SOURCE_DIR}" rev-parse HEAD)"
recipe_blob_oid="$(git -C "${GITHUB_WORKSPACE}" rev-parse HEAD:containers/sglang.def)"
lock_size_bytes="$(stat --format=%s "${SGLANG_SOURCE_DIR}/uv.lock")"
lock_sha256="$(sha256sum "${SGLANG_SOURCE_DIR}/uv.lock" | awk '{ print $1 }')"
patch_records="$({
  while IFS= read -r -d '' patch_file
  do
    patch_path="${patch_file#${GITHUB_WORKSPACE}/}"
    patch_blob_oid="$(git -C "${GITHUB_WORKSPACE}" rev-parse "HEAD:${patch_path}")"
    jq -cn --arg path "${patch_path}" --arg blob_oid "${patch_blob_oid}" \
      '{path:$path,blob_oid:$blob_oid}'
  done < <(find "${GITHUB_WORKSPACE}/patches/sglang" -type f -name '*.patch' -print0 | sort -z)
} | jq -cs '.')"
jq -cS -n \
  --arg qwen_object_hash_algorithm "${qwen_object_format}" \
  --arg qwen_commit "${qwen_commit}" \
  --arg sglang_object_hash_algorithm "${sglang_object_format}" \
  --arg sglang_commit "${sglang_source_commit}" \
  --arg recipe_blob_oid "${recipe_blob_oid}" \
  --argjson patches "${patch_records}" \
  --argjson base_oci "$(jq 'del(.inspect_raw)' "${BUILD_DIR}/sif-metadata/base-oci.json")" \
  --arg lock_size_bytes "${lock_size_bytes}" \
  --arg lock_sha256 "${lock_sha256}" \
  '{schema_version:"sif-embedded-manifest/v1",source_commits:[{repository_id:"apac2026-qwen",commit_role:"sif_build_definition",object_hash_algorithm:$qwen_object_hash_algorithm,oid:$qwen_commit},{repository_id:"apac2026-sglang",commit_role:"sglang_source",object_hash_algorithm:$sglang_object_hash_algorithm,oid:$sglang_commit}],recipe:{path:"containers/sglang.def",blob_oid:$recipe_blob_oid},applied_patches:$patches,base_oci:$base_oci,lock_manifest:{path:"uv.lock",size_bytes:($lock_size_bytes|tonumber),sha256:$lock_sha256}}' \
  >"${BUILD_DIR}/containers/apac-manifest.json"
cp "${BUILD_DIR}/containers/apac-manifest.json" "${BUILD_DIR}/sif-metadata/embedded-manifest.json"

module load singularity-ce/4.1.3
export SINGULARITY_CACHEDIR=/dev/shm/qwen-sglang-build/cache
export SINGULARITY_TMPDIR="${BUILD_DIR}/tmp"

singularity --version | tee logs/build.log
command -v mksquashfs | tee -a logs/build.log
singularity build --fakeroot --writable-tmpfs --sandbox sglang-rootfs containers/sglang.def 2>&1 | tee -a logs/build.log
singularity inspect --json "${ROOTFS_DIR}" > "${BUILD_DIR}/inspect-metadata.json"
sglang_commit="$(tr -d '[:space:]' < "${ROOTFS_DIR}/opt/sglang/sglang-commit.txt")"
nccl_tests_commit="$(tr -d '[:space:]' < "${ROOTFS_DIR}/opt/sglang/nccl-tests-commit.txt")"
cuda_base_image="$(awk '$1 == "From:" { print $2; exit }' "${GITHUB_WORKSPACE}/containers/sglang.def")"
{
  printf 'workflow_run_id=%s\n' "${GITHUB_RUN_ID}"
  printf 'workflow_run_attempt=%s\n' "${GITHUB_RUN_ATTEMPT}"
  printf 'workflow_run_number=%s\n' "${GITHUB_RUN_NUMBER}"
  printf 'sglang_commit=%s\n' "${sglang_commit}"
  printf 'nccl_tests_ref=717b68318278e93f371d8ffb46b076069d7c7851\n'
  printf 'nccl_tests_commit=%s\n' "${nccl_tests_commit}"
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
set +e
singularity test --writable-tmpfs sglang.sif \
  >"${BUILD_DIR}/sif-metadata/sif-build/build-test.stdout" \
  2>"${BUILD_DIR}/sif-metadata/sif-build/build-test.stderr"
build_test_status=$?
set -e
printf '%d\n' "${build_test_status}" >"${BUILD_DIR}/sif-metadata/sif-build/build-test.exit-code"
cp "${BUILD_DIR}/sif-metadata/sif-build/build-test.stdout" logs/runtime-test.stdout
cp "${BUILD_DIR}/sif-metadata/sif-build/build-test.stderr" logs/runtime-test.stderr
[[ "${build_test_status}" -eq 0 ]]

capture_toolchain() {
  local role=$1
  shift
  local prefix="${BUILD_DIR}/sif-metadata/sif-build/toolchain/${role}"
  set +e
  "$@" >"${prefix}.stdout" 2>"${prefix}.stderr"
  local status=$?
  set -e
  printf '%d\n' "${status}" >"${prefix}.exit-code"
  [[ "${status}" -eq 0 ]]
}
capture_toolchain singularity_version singularity --version
capture_toolchain python_version singularity exec sglang.sif /opt/sglang/venv/bin/python --version
capture_toolchain cuda_version singularity exec sglang.sif /usr/local/cuda/bin/nvcc --version
capture_toolchain compiler_version singularity exec sglang.sif /usr/bin/gcc --version
singularity exec sglang.sif cat /opt/apac/manifest.json | cmp - "${BUILD_DIR}/sif-metadata/embedded-manifest.json"
singularity exec \
  --bind "${GITHUB_WORKSPACE}/scripts/inventory-sif-runtime.py:/inventory-sif-runtime.py:ro,${BUILD_DIR}/sif-metadata:/metadata" \
  sglang.sif /opt/sglang/venv/bin/python /inventory-sif-runtime.py
uv run --project "${GITHUB_WORKSPACE}/qwen-experiment-metrics" --frozen --no-dev \
  qwen-experiment-metrics create-sif-sidecar \
  --workspace "${GITHUB_WORKSPACE}" \
  --sglang-source "${SGLANG_SOURCE_DIR}" \
  --sif "${BUILD_DIR}/sglang.sif" \
  --base-oci "${BUILD_DIR}/sif-metadata/base-oci.json" \
  --embedded-manifest "${BUILD_DIR}/sif-metadata/embedded-manifest.json" \
  --runtime-inventory "${BUILD_DIR}/sif-metadata/runtime-inventory.json" \
  --output "${BUILD_DIR}/sif-metadata/sif-sidecar.json"

install -d -m 2775 "${SIF_ARCHIVE_DIR}"
test ! -e "${METADATA_ARCHIVE}"
cp -a "${BUILD_DIR}/sif-metadata" "${METADATA_PUBLISH_TMP}"
cp sglang.sif "${PUBLISH_TMP}"
mv -- "${PUBLISH_TMP}" "${SIF_ARCHIVE_DIR}/${GITHUB_RUN_NUMBER}.sif"
mv -- "${METADATA_PUBLISH_TMP}" "${METADATA_ARCHIVE}"
echo "Published ${SIF_ARCHIVE_DIR}/${GITHUB_RUN_NUMBER}.sif" | tee -a logs/build.log
