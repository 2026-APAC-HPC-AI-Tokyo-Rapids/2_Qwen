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
jq -e '.diagnostic == "vllm-sm90-megamoe" and .measurement_mode == "kernel-gate" and .profile == false and .world_size == 4' "${RUNTIME_DIR}/source/provenance.json" >/dev/null
jq -e '.schema == "vllm-sm90-actual-adapter-gate/v1" and .completed == true and .correctness_passed == true and .graph_passed == true and .world_size == 4 and (.gpus | length) == 4 and ([.gpus[].rank] | sort) == [0,1,2,3] and .adapter_manifest.backend == "apac_sm90_megamoe" and .capacity_requested == 4096 and ([.cases[].counts] == [[1,1,1,1],[128,0,17,64],[128,128,128,128],[4096,4096,4096,4096]]) and all(.cases[]; (.records | length) == 16 and ([.records[] | [.rank,.replay]] | sort | unique | length) == 16)' "${RUNTIME_DIR}/results/gate.json" >/dev/null
# This gate has no timing/adoption claim; actual adapter correctness is required.
test ! -e "${RUNTIME_DIR}/results/score.json"
