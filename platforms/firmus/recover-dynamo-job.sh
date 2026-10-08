#!/usr/bin/env bash
# Temporary recovery of already-submitted jobs; never calls sbatch.
set -euo pipefail
: "${RECOVERY_JOB_ID:?}"
: "${RUNTIME_DIR:?}"
[[ "${RECOVERY_JOB_ID}" =~ ^[0-9]+$ && "${RUNTIME_DIR}" == "${RUNTIME_ROOT}/dynamo-"* ]]
test -s "${RUNTIME_DIR}/source/provenance.json"
job=$(scontrol show job "${RECOVERY_JOB_ID}")
grep -Fq "WorkDir=${RUNTIME_DIR}" <<<"${job}"
printf '%s\n' "${job}" >"${RUNTIME_DIR}/logs/slurm-before-recovery.txt"
if grep -Fq 'JobState=PENDING' <<<"${job}"; then
  scontrol update "JobId=${RECOVERY_JOB_ID}" TimeLimit=00:17:00
else
  # A lost SSH reply can hide a successful update; attach without resubmitting.
  grep -Fq 'TimeLimit=00:17:00' <<<"${job}"
fi
scontrol show job "${RECOVERY_JOB_ID}" >"${RUNTIME_DIR}/logs/slurm-after-recovery.txt"
cat "${RUNTIME_DIR}/logs/slurm-after-recovery.txt"
jq -n --arg run "${GITHUB_RUN_ID}" --arg commit "${GITHUB_SHA}" --arg job "${RECOVERY_JOB_ID}" '{recovery_workflow_run_id:$run,recovery_commit:$commit,slurm_job_id:$job,time_limit:"00:17:00",new_job_submitted:false}' >"${RUNTIME_DIR}/source/recovery.json"
if [[ "${RECOVERY_JOB_ID}" == 21736 ]]; then
  duplicate_state=$(sacct --allocations --jobs=21733 --noheader --parsable2 --format=JobID,State | awk -F'|' '$1 == 21733 {print $2}')
  if [[ "${duplicate_state}" == PENDING ]]; then
    duplicate=$(scontrol show job 21733)
    grep -Fq 'WorkDir=/scratch/home/ACCOUNT/qwen/runs/dynamo-37726198579-1.URxt10gJ1o' <<<"${duplicate}"
    grep -Fq 'JobState=PENDING' <<<"${duplicate}"
    scancel 21733
    printf 'Cancelled own superseded pending SGLang job 21733.\n'
  else
    [[ "${duplicate_state}" == CANCELLED* ]]
    printf 'Superseded SGLang job 21733 is already cancelled.\n'
  fi
fi
