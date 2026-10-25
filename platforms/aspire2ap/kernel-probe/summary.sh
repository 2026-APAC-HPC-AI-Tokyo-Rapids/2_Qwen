#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?}"
bash -lc 'myprojects -p 59010014' >"${RUNTIME_DIR}/logs/su-after.txt"
printf '### ASPIRE kernel probe\n\n1 H100 GPU; PBS walltime 5 minutes; calibration process limit 165 seconds.\n\n'
for phase in before after; do
  printf 'SU balance (%s submission/run):\n\n```text\n' "${phase}"
  if [[ -f "${RUNTIME_DIR}/logs/su-${phase}.txt" ]]; then cat "${RUNTIME_DIR}/logs/su-${phase}.txt"; else printf 'Not recorded\n'; fi
  printf '```\n\n'
done
if [[ -f "${RUNTIME_DIR}/source/provenance.json" ]]; then cat "${RUNTIME_DIR}/source/provenance.json"; printf '\n\n'; fi
if [[ -f "${RUNTIME_DIR}/results/trial-1.log" ]]; then grep -E '^(PASS |CALIB )' "${RUNTIME_DIR}/results/trial-1.log" || true; fi
if [[ -f "${RUNTIME_DIR}/results/exit-code" ]]; then printf '\nExit code: %s\n' "$(cat "${RUNTIME_DIR}/results/exit-code")"; fi
