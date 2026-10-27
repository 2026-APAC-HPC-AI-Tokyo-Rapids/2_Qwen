#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?Slurm job ID is required}"
: "${RUNTIME_DIR:?runtime directory is required}"

# Query the user's queue so a completed job disappearing from squeue is an empty result.
state="$(squeue --noheader --user="$(id -un)" --format='%i|%T' | awk -F'|' -v job_id="${SLURM_JOB_ID}" '$1 == job_id {print $2}')"
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
if [[ -s "${RUNTIME_DIR}/source/firmus-framework.json" && "$(jq -r '.framework' "${RUNTIME_DIR}/source/firmus-framework.json")" == sglang && "$(jq -r '.speculation.algorithm' "${RUNTIME_DIR}/source/firmus-framework.json")" == EAGLE3 ]]; then test -s "${RUNTIME_DIR}/results/eagle-acceptance.json"; fi

if [[ -s "${RUNTIME_DIR}/source/firmus-framework.json" ]]; then
  routing_capture_samples=$(jq -r '.prefill.routing_capture_samples // 0' "${RUNTIME_DIR}/source/firmus-framework.json")
  if (( routing_capture_samples > 0 )); then
    jq -e --argjson samples "${routing_capture_samples}" '.schema == "prefill-routing-summary/v1" and .valid == true and .expected_workers == 3 and .expected_ranks == 4 and .samples_per_rank == $samples' "${RUNTIME_DIR}/results/prefill-routing-summary.json" >/dev/null
  fi
fi
