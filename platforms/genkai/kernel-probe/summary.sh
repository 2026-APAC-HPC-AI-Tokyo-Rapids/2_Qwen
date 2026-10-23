#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?}"
printf '| Trial | Result |\n| --- | --- |\n'
for trial in 1 2 3; do
  result=$(grep '^PASS ' "${RUNTIME_DIR}/results/trial-${trial}.log" 2>/dev/null || true)
  while IFS= read -r line; do printf '| %s | %s |\n' "${trial}" "${line}"; done <<<"${result:-not completed}"
done
if [[ -f "${RUNTIME_DIR}/results/exit-code" ]]; then printf '\nExit code: %s\n' "$(cat "${RUNTIME_DIR}/results/exit-code")"; fi
