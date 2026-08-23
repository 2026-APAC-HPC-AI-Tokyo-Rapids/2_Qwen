#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?AWS region is required}"
: "${CONTAINER_BUCKET:?container bucket is required}"
: "${GITHUB_OUTPUT:?GitHub output file is required}"
: "${GITHUB_SHA:?workflow commit is required}"
: "${REQUIRE_BUILD:?build selection is required}"

if [[ "${REQUIRE_BUILD}" == true ]]; then printf 'action=build\n' >>"${GITHUB_OUTPUT}"; exit 0; fi

commit="${GITHUB_SHA}"
while [[ -n "${commit}" ]]; do
  sidecar_keys="$(aws s3api list-objects-v2 --region "${AWS_REGION}" --bucket "${CONTAINER_BUCKET}" --prefix "containers/${commit}/" --query "reverse(sort_by(Contents[?ends_with(Key, '/container-sidecar.json')], &LastModified))[].Key" --output text 2>/dev/null || true)"
  while IFS= read -r sidecar_key; do
    [[ -n "${sidecar_key}" && "${sidecar_key}" == containers/"${commit}"/*/container-sidecar.json ]] || continue
    sidecar="$(aws s3 cp --region "${AWS_REGION}" "s3://${CONTAINER_BUCKET}/${sidecar_key}" - --only-show-errors 2>/dev/null)" || continue
    expected_size="$(jq -er --arg commit "${commit}" 'select(.schema_version == "container-sidecar/v1" and .source_commits.qwen == $commit) | .container_artifact.size_bytes' <<<"${sidecar}" 2>/dev/null)" || continue
    expected_sha256="$(jq -er '.container_artifact.sha256 | select(test("^[0-9a-f]{64}$"))' <<<"${sidecar}" 2>/dev/null)" || continue
    image_key="${sidecar_key%/container-sidecar.json}/sglang.sqsh"
    image_head="$(aws s3api head-object --region "${AWS_REGION}" --bucket "${CONTAINER_BUCKET}" --key "${image_key}" 2>/dev/null)" || continue
    [[ "$(jq -er '.ContentLength' <<<"${image_head}")" == "${expected_size}" ]] || continue
    [[ "$(jq -er '.Metadata.sha256' <<<"${image_head}")" == "${expected_sha256}" ]] || continue
    printf 'action=reuse\nsource_commit=%s\ncontainer_key=%s\nsidecar_key=%s\n' "${commit}" "${image_key}" "${sidecar_key}" >>"${GITHUB_OUTPUT}"
    exit 0
  done < <(tr '\t' '\n' <<<"${sidecar_keys}")
  commit="$(git rev-parse "${commit}^" 2>/dev/null || true)"
done

printf 'action=build\n' >>"${GITHUB_OUTPUT}"
