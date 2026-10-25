#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?}"
printf '### Firmus kernel probe\n\nOne exclusive H200 node (8 GPUs allocated); one GPU visible to the probe.\nSlurm walltime 5 minutes; NCU permission check 30 seconds; one calibration trial, process limit 165 seconds.\n\n'
for file in logs/provenance.json logs/slurm-accounting.txt results/ncu-permission.csv results/exit-code; do
  if [[ -f "${RUNTIME_DIR}/${file}" ]]; then printf '\n%s\n```text\n' "${file}"; cat "${RUNTIME_DIR}/${file}"; printf '\n```\n'; fi
done
if [[ -f "${RUNTIME_DIR}/results/trial-1.log" ]]; then grep -E '^(PASS |CALIB )' "${RUNTIME_DIR}/results/trial-1.log" || true; fi
