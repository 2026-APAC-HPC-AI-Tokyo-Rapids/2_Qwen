#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?}"
printf '| Trial | Result |\n| --- | --- |\n'
for trial in 1 2 3; do
  result=$(grep '^PASS ' "${RUNTIME_DIR}/results/trial-${trial}.log" 2>/dev/null || true)
  printf '| %s | %s |\n' "${trial}" "${result:-not completed}"
done
if [[ -f "${RUNTIME_DIR}/results/exit-code" ]]; then printf '\nExit code: %s\n' "$(cat "${RUNTIME_DIR}/results/exit-code")"; fi
