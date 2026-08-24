#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_ARCHIVE_DIR:?container archive directory is required}"
: "${GITHUB_OUTPUT:?GitHub output file is required}"
: "${GITHUB_RUN_ID:?workflow run ID is required}"
: "${GITHUB_SHA:?workflow commit is required}"
: "${REQUIRE_BUILD:?build selection is required}"

if [[ "${REQUIRE_BUILD}" == true ]]; then printf 'action=build\ncandidates=[]\n' >>"${GITHUB_OUTPUT}"; exit 0; fi

candidates=()
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
    [[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${expected_sha256}" ]] || continue
    candidates+=("$(jq -cn --arg run_id "${run_id}" --arg sha256 "${expected_sha256}" --arg size "${expected_size}" '{run_id:$run_id,sha256:$sha256,size:$size}')")
    if [[ "${#candidates[@]}" -ge 16 ]]; then break 2; fi
  done < <(gh run list --workflow qwen.yml --commit "${commit}" --limit 1000 --json databaseId --jq '.[].databaseId' 2>/dev/null || true)
  commit="$(git rev-parse --verify "${commit}^" 2>/dev/null || true)"
done

if [[ "${#candidates[@]}" -eq 0 ]]; then printf 'action=build\ncandidates=[]\n' >>"${GITHUB_OUTPUT}"; exit 0; fi
candidates_json="$(printf '%s\n' "${candidates[@]}" | jq -sc '.')"
printf 'action=reuse\nsource_run_id=%s\ncontainer_sha256=%s\ncontainer_size=%s\ncandidates=%s\n' "$(jq -r '.[0].run_id' <<<"${candidates_json}")" "$(jq -r '.[0].sha256' <<<"${candidates_json}")" "$(jq -r '.[0].size' <<<"${candidates_json}")" "${candidates_json}" >>"${GITHUB_OUTPUT}"
