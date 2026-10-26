#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${RUNTIME_DIR:?}"
state=$(squeue --noheader --user="$(id -un)" --format='%i|%T' | awk -F'|' -v job_id="${SLURM_JOB_ID}" '$1 == job_id {print $2}')
if [[ -n "${state}" ]]; then printf 'scheduler_job_pending=%s state=%s\n' "${SLURM_JOB_ID}" "${state}"; exit 75; fi
sacct --allocations --jobs="${SLURM_JOB_ID}" --noheader --parsable2 --format=JobID,State,ExitCode,Elapsed,AllocTRES >"${RUNTIME_DIR}/logs/slurm-accounting.txt"
cat "${RUNTIME_DIR}/logs/slurm-accounting.txt"
for log in driver.log slurm.log slurm.stderr.log; do if [[ -f "${RUNTIME_DIR}/logs/${log}" ]]; then cat "${RUNTIME_DIR}/logs/${log}"; fi; done
state=$(awk -F'|' -v job_id="${SLURM_JOB_ID}" '$1 == job_id {print $2}' "${RUNTIME_DIR}/logs/slurm-accounting.txt")
exit_code=$(awk -F'|' -v job_id="${SLURM_JOB_ID}" '$1 == job_id {print $3}' "${RUNTIME_DIR}/logs/slurm-accounting.txt")
[[ "${state}" == COMPLETED && "${exit_code}" == 0:0 ]]
test -e "${RUNTIME_DIR}/run-success"
test "$(cat "${RUNTIME_DIR}/results/exit-code")" = 0
grep -q '^PASS ' "${RUNTIME_DIR}/results/trial-1.log"
