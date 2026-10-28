#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?}"
if [[ -f "${RUNTIME_DIR}/logs/provenance.json" ]] && [[ "$(jq -r .probe_gpus "${RUNTIME_DIR}/logs/provenance.json")" == 8 ]]; then
    printf '### Firmus EAGLE parallel TP4 draft acceptance diagnostic (17 minute allocation)\n'
    if [[ -f "${RUNTIME_DIR}/results/eagle-acceptance.json" ]]; then cat "${RUNTIME_DIR}/results/eagle-acceptance.json"; fi
elif [[ -f "${RUNTIME_DIR}/logs/provenance.json" ]] && [[ "$(jq -r .benchmark "${RUNTIME_DIR}/logs/provenance.json")" == kernels/probes/prepare-prefill-memory.sh ]]; then
    printf '### Firmus Prefill memory probe\n\nOne H200 visible; 5 minute allocation, one trial capped at 165 seconds. No counter profiler.\n\n'
    if [[ -f "${RUNTIME_DIR}/results/prefill-memory/results.json" ]]; then
        jq '{candidate_name, candidate_eligible, timing: [.timing[] | {rows, dtype, median_us, ratio}]}' "${RUNTIME_DIR}/results/prefill-memory/results.json"
    fi
elif [[ -f "${RUNTIME_DIR}/logs/provenance.json" ]] && [[ "$(jq -r .benchmark "${RUNTIME_DIR}/logs/provenance.json")" == kernels/probes/prepare-fa3-exp.sh ]]; then
    printf '### Firmus FA3 exponent probe\n\nOne exclusive H200 node (8 GPUs allocated); one GPU visible. Five minute allocation; one trial capped at 165 seconds. No counter profiler. Numerical PASS and performance eligibility are separate.\n\n'
    if [[ -f "${RUNTIME_DIR}/results/fa3-exp/results.json" ]]; then
        jq '{candidate_name, candidate_eligible, timing}' "${RUNTIME_DIR}/results/fa3-exp/results.json"
    fi
else
    printf '### Firmus kernel probe\n\none exclusive H200 node (8 GPUs allocated); one GPU visible to the probe.\nSlurm walltime 5 minutes; NCU permission check 30 seconds; one calibration trial, process limit 165 seconds.\n\n'
fi
for file in logs/provenance.json logs/slurm-accounting.txt results/ncu-permission.csv results/exit-code; do
  if [[ -f "${RUNTIME_DIR}/${file}" ]]; then printf '\n%s\n```text\n' "${file}"; cat "${RUNTIME_DIR}/${file}"; printf '\n```\n'; fi
done
if [[ -f "${RUNTIME_DIR}/results/trial-1.log" ]]; then grep -E '^(PASS |CALIB )' "${RUNTIME_DIR}/results/trial-1.log" || true; fi
