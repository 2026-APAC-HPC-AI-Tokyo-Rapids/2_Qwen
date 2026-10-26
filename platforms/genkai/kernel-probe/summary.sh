#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?}"
if [[ "$(jq -r '.calibration // false' "${RUNTIME_DIR}/source/provenance.json")" == true ]]; then
  printf '| Shape | Candidate | Median µs | p10 µs | p90 µs | Candidate/stock | Block 1 ratio | Block 2 ratio |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
  if [[ -f "${RUNTIME_DIR}/results/trial-1.log" ]]; then sed -n 's/^CALIB //p' "${RUNTIME_DIR}/results/trial-1.log"; fi
  printf '\n'
  if [[ -f "${RUNTIME_DIR}/results/exit-code" ]]; then printf 'Exit code: %s\n' "$(cat "${RUNTIME_DIR}/results/exit-code")"; fi
  exit 0
fi
printf '| Trial | Result |\n| --- | --- |\n'
for trial in 1 2 3; do
  result=$(grep '^PASS ' "${RUNTIME_DIR}/results/trial-${trial}.log" 2>/dev/null || true)
  while IFS= read -r line; do printf '| %s | %s |\n' "${trial}" "${line}"; done <<<"${result:-not completed}"
done
if [[ -f "${RUNTIME_DIR}/results/exit-code" ]]; then printf '\nExit code: %s\n' "$(cat "${RUNTIME_DIR}/results/exit-code")"; fi
