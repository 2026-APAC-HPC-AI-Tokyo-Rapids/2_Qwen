#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_ARCHIVE_DIR:?container archive directory is required}"
: "${GITHUB_OUTPUT:?GitHub output file is required}"
: "${GITHUB_RUN_ID:?workflow run ID is required}"
: "${GITHUB_SHA:?workflow commit is required}"
: "${REQUIRE_BUILD:?build selection is required}"

hash_progress_pid=
stop_hash_progress() {
  if [[ -n "${hash_progress_pid}" ]]; then
    kill "${hash_progress_pid}" 2>/dev/null || true
    wait "${hash_progress_pid}" 2>/dev/null || true
    hash_progress_pid=
  fi
}
trap stop_hash_progress EXIT

verify_sha256() {
  local archive_path=$1
  local expected=$2
  local actual

  while sleep 15; do printf 'container_sha256_check_in_progress=%s\n' "$(basename "${archive_path}")" >&2; done &
  hash_progress_pid=$!
  actual="$(sha256sum "${archive_path}" | awk '{print $1}')"
  stop_hash_progress
  [[ "${actual}" == "${expected}" ]]
}

if [[ "${REQUIRE_BUILD}" == true ]]; then printf 'action=build\n' >>"${GITHUB_OUTPUT}"; exit 0; fi

# ASPIRE serving pins a verified, already staged build rather than an ancestor's latest run.
if [[ -n "${1:-}" ]]; then
  config=$1
  run_id="$(jq -er .container_build_run_id "${config}")"
  expected_size="$(jq -er .container_size_bytes "${config}")"
  archive="${CONTAINER_ARCHIVE_DIR}/${run_id}.sqsh"
  sidecar="${CONTAINER_ARCHIVE_DIR}/${run_id}.container-sidecar.json"
  expected_sha256="$(jq -r '.container_sha256 // empty' "${config}")"
  if [[ -z "${expected_sha256}" ]]; then
    # A newly built image can be pinned by build run + source commit before its
    # hash is available locally. Verify the actual image against that sidecar.
    source_commit="$(jq -er '.container_source_commit | select(test("^[0-9a-f]{40}$"))' "${config}")"
    expected_sha256="$(jq -er --arg commit "${source_commit}" 'select(.source_commits.qwen == $commit) | .container_artifact.sha256 | select(test("^[0-9a-f]{64}$"))' "${sidecar}")"
  fi
  [[ "$(stat -c '%s' "${archive}")" == "${expected_size}" ]]
  jq -e --arg sha "${expected_sha256}" --argjson size "${expected_size}" '.schema_version == "container-sidecar/v1" and .container_artifact.artifact_relative_path == "sglang.sqsh" and .container_artifact.sha256 == $sha and .container_artifact.size_bytes == $size' "${sidecar}" >/dev/null
  verify_sha256 "${archive}" "${expected_sha256}"
  printf 'action=reuse\nsource_run_id=%s\ncontainer_sha256=%s\ncontainer_size=%s\n' "${run_id}" "${expected_sha256}" "${expected_size}" >>"${GITHUB_OUTPUT}"
  exit 0
fi

commit="${GITHUB_SHA}"
while [[ -n "${commit}" ]]; do
  while IFS= read -r run_id; do
    [[ "${run_id}" != "${GITHUB_RUN_ID}" ]] || continue
    archive="${CONTAINER_ARCHIVE_DIR}/${run_id}.sqsh"
    sidecar="${CONTAINER_ARCHIVE_DIR}/${run_id}.container-sidecar.json"
    [[ -s "${archive}" && -s "${sidecar}" ]] || continue
    expected_size="$(jq -er 'select(.schema_version == "container-sidecar/v1" and .container_artifact.artifact_relative_path == "sglang.sqsh") | .container_artifact.size_bytes' "${sidecar}" 2>/dev/null)" || continue
    expected_sha256="$(jq -er '.container_artifact.sha256 | select(test("^[0-9a-f]{64}$"))' "${sidecar}" 2>/dev/null)" || continue
    [[ "$(stat -c '%s' "${archive}")" == "${expected_size}" ]] || continue
    verify_sha256 "${archive}" "${expected_sha256}" || continue
    printf 'action=reuse\nsource_run_id=%s\ncontainer_sha256=%s\ncontainer_size=%s\n' "${run_id}" "${expected_sha256}" "${expected_size}" >>"${GITHUB_OUTPUT}"
    exit 0
  done < <(gh run list --workflow qwen.yml --commit "${commit}" --limit 1000 --json databaseId --jq '.[].databaseId' 2>/dev/null || true)
  commit="$(git rev-parse --verify "${commit}^" 2>/dev/null || true)"
done

printf 'action=build\n' >>"${GITHUB_OUTPUT}"
