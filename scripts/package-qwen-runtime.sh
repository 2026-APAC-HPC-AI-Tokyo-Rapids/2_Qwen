#!/usr/bin/env bash
set -euo pipefail

: "${RUNTIME_DIR:?runtime directory is required}"

manifest="${RUNTIME_DIR}/control/run-manifest.json"
run_purpose="$(jq -er '.run_purpose' "${manifest}")"
package_args=()
while IFS= read -r -d '' member; do relative="${member#"${RUNTIME_DIR}/"}"; package_args+=(--member "${relative%%/*}=${relative}"); done < <(find "${RUNTIME_DIR}/control" "${RUNTIME_DIR}/domain" "${RUNTIME_DIR}/logs" "${RUNTIME_DIR}/results" "${RUNTIME_DIR}/source" -type f -print0 | sort -z)
[[ ! -e "${RUNTIME_DIR}/run-success" ]] || package_args+=(--member success=run-success)
profile_args=()
if [[ "${run_purpose}" == profile ]]; then while IFS= read -r -d '' member; do profile_args+=(--profile-member "${member#"${RUNTIME_DIR}/"}"); done < <(find "${RUNTIME_DIR}/profile/raw" -type f -print0 | sort -z); fi
"${RUNTIME_DIR}/qwen-experiment-metrics/.venv/bin/qwen-experiment-metrics" package --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" "${package_args[@]}" "${profile_args[@]}"
