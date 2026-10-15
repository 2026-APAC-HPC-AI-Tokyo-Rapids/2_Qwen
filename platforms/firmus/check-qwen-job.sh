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
if jq -e '.measurement_mode == "profile-only" and .profile == true and .framework == "vllm"' "${RUNTIME_DIR}/source/provenance.json" >/dev/null; then
  test ! -e "${RUNTIME_DIR}/results/score.json"
  test -s "${RUNTIME_DIR}/profile/client/profile_export.jsonl"
  jq -e '.was_cancelled == false and .error_summary == [] and .request_count.avg > 0' "${RUNTIME_DIR}/profile/client/profile_export_aiperf.json" >/dev/null
  jq -e '.valid == true and .scope == "all-workers" and (.workers | keys) == ["decode-0", "prefill-0", "prefill-1", "prefill-2"] and all(.workers[]; .gpu_count == 4)' "${RUNTIME_DIR}/profile/validation.json" >/dev/null
  jq -e '.completed == true and .correctness_passed == true and .graph_passed == true and .native_factory_runner == true and .capacity_requested == 4096 and (.cases | length) == 4 and ([.cases[].records[]] | length) == 64' "${RUNTIME_DIR}/results/gate.json" >/dev/null
else
  test -s "${RUNTIME_DIR}/results/profile_export_aiperf.json"
  test -s "${RUNTIME_DIR}/results/score.json"
fi
