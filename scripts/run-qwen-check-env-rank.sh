#!/usr/bin/env bash
set -euo pipefail

: "${RUNTIME_DIR:?runtime directory is required}"

host="$(hostname)"
host="${host%%.*}"
rank=""
while read -r mapped_host mapped_rank; do
  if [[ "${mapped_host}" == "${host}" ]]; then rank="${mapped_rank}"; break; fi
done <"${RUNTIME_DIR}/logs/rank-map.tsv"
[[ "${rank}" =~ ^[0-9]+$ ]]
printf 'launcher_hostname=%s launcher_rank=%s\n' "${host}" "${rank}"
export CHECK_ENV_RANK="${rank}"
export CHECK_ENV_DEFER_AGGREGATE=1
exec "${RUNTIME_DIR}/scripts/run-qwen-check-env.sh"
