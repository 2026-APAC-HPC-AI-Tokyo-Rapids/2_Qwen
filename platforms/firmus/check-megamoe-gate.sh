#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?}"
: "${RUNTIME_DIR:?}"
state=$(squeue --noheader --format='%i|%T' | awk -F'|' -v id="${SLURM_JOB_ID}" '$1 == id {print $2}')
if [[ -n "${state}" ]]; then printf 'scheduler_job_pending=%s state=%s\n' "${SLURM_JOB_ID}" "${state}"; exit 75; fi
sacct --allocations --jobs="${SLURM_JOB_ID}" --noheader --parsable2 --format=JobID,State,ExitCode,Elapsed,AllocNodes >"${RUNTIME_DIR}/logs/slurm-accounting.txt"
cat "${RUNTIME_DIR}/logs/slurm-accounting.txt"
cat "${RUNTIME_DIR}/logs/slurm-${SLURM_JOB_ID}.log"
awk -F'|' -v id="${SLURM_JOB_ID}" '$1 == id && $2 == "COMPLETED" && $3 == "0:0" {ok=1} END {exit !ok}' "${RUNTIME_DIR}/logs/slurm-accounting.txt"
test -e "${RUNTIME_DIR}/run-success"
python3 "${RUNTIME_DIR}/scripts/megamoe/validate_adapter_gate.py" "${RUNTIME_DIR}"
# This gate has no timing/adoption claim; actual adapter correctness is required.
test ! -e "${RUNTIME_DIR}/results/score.json"
