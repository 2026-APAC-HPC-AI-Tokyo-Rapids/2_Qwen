#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?Slurm job ID is required}"
: "${RUNTIME_DIR:?runtime directory is required}"

state="$(squeue --noheader --format='%i|%T' | awk -F'|' -v job_id="${SLURM_JOB_ID}" '$1 == job_id {print $2}')"
if [[ -n "${state}" ]]; then printf 'scheduler_job_pending=%s state=%s\n' "${SLURM_JOB_ID}" "${state}"; exit 75; fi
sacct --allocations --jobs="${SLURM_JOB_ID}" --noheader --parsable2 --format=JobID,State,ExitCode,Elapsed,AllocNodes >"${RUNTIME_DIR}/logs/slurm-accounting.txt"
cat "${RUNTIME_DIR}/logs/slurm-accounting.txt"
cat "${RUNTIME_DIR}/logs/slurm-${SLURM_JOB_ID}.log"
state="$(awk -F'|' -v job_id="${SLURM_JOB_ID}" '$1 == job_id {print $2}' "${RUNTIME_DIR}/logs/slurm-accounting.txt")"
exit_code="$(awk -F'|' -v job_id="${SLURM_JOB_ID}" '$1 == job_id {print $3}' "${RUNTIME_DIR}/logs/slurm-accounting.txt")"
test "${state}" = COMPLETED
test "${exit_code}" = 0:0
test -e "${RUNTIME_DIR}/run-success"
test -s "${RUNTIME_DIR}/results/profile_export_aiperf.json"
test -s "${RUNTIME_DIR}/results/score.json"
