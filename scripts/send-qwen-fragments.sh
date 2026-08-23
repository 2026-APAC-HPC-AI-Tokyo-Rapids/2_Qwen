#!/usr/bin/env bash
set -euo pipefail

: "${METRICS_CREDENTIAL_JSON:?metrics credential JSON is required}"
: "${RUNTIME_DIR:?runtime directory is required}"

credential_file="$(mktemp /dev/shm/qwen-bigquery.XXXXXXXXXX)"
cleanup() {
  status=$?
  trap - EXIT INT TERM
  rm -f -- "${credential_file}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

chmod 0600 "${credential_file}"
printf '%s' "${METRICS_CREDENTIAL_JSON}" >"${credential_file}"
export EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_FILE="${credential_file}"
export EXPERIMENT_METRICS_BIGQUERY_CONFIG="${RUNTIME_DIR}/qwen-experiment-metrics/pyproject.toml"
run_id="$(jq -er '.run_id' "${RUNTIME_DIR}/control/run-manifest.json")"
"${RUNTIME_DIR}/qwen-experiment-metrics/.venv/bin/qwen-experiment-metrics" send-fragments --run "${run_id}" --runtime-dir "${RUNTIME_DIR}"
