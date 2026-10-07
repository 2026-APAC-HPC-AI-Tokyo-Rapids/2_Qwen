#!/usr/bin/env bash
set -euo pipefail

: "${ACTION:?container action is required}"
: "${CONTAINER_ARCHIVE_DIR:?container archive directory is required}"
: "${GITHUB_RUN_ATTEMPT:?workflow run attempt is required}"
: "${GITHUB_RUN_ID:?workflow run ID is required}"

[[ "${ACTION}" == build || "${ACTION}" == reuse ]]
framework=${CONTAINER_FRAMEWORK:-sglang}
[[ "${framework}" == sglang || "${framework}" == vllm ]]

install -d -m 2775 "${CONTAINER_ARCHIVE_DIR}"
current_image="${CONTAINER_ARCHIVE_DIR}/${GITHUB_RUN_ID}.sqsh"
current_sidecar="${CONTAINER_ARCHIVE_DIR}/${GITHUB_RUN_ID}.container-sidecar.json"

if [[ "${ACTION}" == reuse ]]; then
  : "${SOURCE_RUN_ID:?reuse source run ID is required}"
  [[ "${SOURCE_RUN_ID}" =~ ^[1-9][0-9]*$ && "${SOURCE_RUN_ID}" != "${GITHUB_RUN_ID}" ]]
  source_image="${CONTAINER_ARCHIVE_DIR}/${SOURCE_RUN_ID}.sqsh"
  source_sidecar="${CONTAINER_ARCHIVE_DIR}/${SOURCE_RUN_ID}.container-sidecar.json"
  ln -f -- "${source_image}" "${current_image}"
  ln -f -- "${source_sidecar}" "${current_sidecar}"
  exit 0
fi

: "${CONTAINER_BUCKET:?container bucket is required}"
: "${CONTAINER_KEY:?container key is required}"
: "${CONTAINER_SHA256:?container SHA-256 is required}"
: "${CONTAINER_SIZE:?container size is required}"
: "${GITHUB_WORKSPACE:?workflow workspace is required}"
: "${SIDECAR_KEY:?container sidecar key is required}"
[[ "${CONTAINER_KEY}" == containers/*/"${framework}.sqsh" && "${SIDECAR_KEY}" == containers/*/container-sidecar.json ]]
[[ "${CONTAINER_KEY%/*}" == "${SIDECAR_KEY%/*}" ]]
[[ "${CONTAINER_SHA256}" =~ ^[0-9a-f]{64}$ && "${CONTAINER_SIZE}" =~ ^[1-9][0-9]*$ ]]

image_temporary="${CONTAINER_ARCHIVE_DIR}/.${GITHUB_RUN_ID}.sqsh.${GITHUB_RUN_ATTEMPT}.tmp"
sidecar_temporary="${CONTAINER_ARCHIVE_DIR}/.${GITHUB_RUN_ID}.container-sidecar.json.${GITHUB_RUN_ATTEMPT}.tmp"
cleanup() { status=$?; rm -f -- "${image_temporary}" "${sidecar_temporary}"; exit "${status}"; }
trap cleanup EXIT
uvx --from 'boto3>=1.40,<2' python "${GITHUB_WORKSPACE}/scripts/s3-object-transfer.py" download "${CONTAINER_BUCKET}" "${SIDECAR_KEY}" "${sidecar_temporary}"
uvx --from 'boto3>=1.40,<2' python "${GITHUB_WORKSPACE}/scripts/s3-object-transfer.py" download "${CONTAINER_BUCKET}" "${CONTAINER_KEY}" "${image_temporary}"
[[ "$(jq -er --arg path "${framework}.sqsh" 'select(.schema_version == "container-sidecar/v1" and .container_artifact.artifact_relative_path == $path) | .container_artifact.size_bytes' "${sidecar_temporary}")" == "${CONTAINER_SIZE}" ]]
[[ "$(jq -er '.container_artifact.sha256' "${sidecar_temporary}")" == "${CONTAINER_SHA256}" ]]
[[ "$(stat -c '%s' "${image_temporary}")" == "${CONTAINER_SIZE}" && "$(sha256sum "${image_temporary}" | awk '{print $1}')" == "${CONTAINER_SHA256}" ]]
mv -- "${image_temporary}" "${current_image}"
mv -- "${sidecar_temporary}" "${current_sidecar}"
trap - EXIT
