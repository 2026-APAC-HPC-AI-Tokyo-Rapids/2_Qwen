#!/usr/bin/env bash
# User-requested cancellation only. This script never allocates a GPU.
set -euo pipefail
job=26800
runtime=/scratch/home/ACCOUNT/qwen/runs/kernel-bench-40563900060-1.7EoJzJoLiD
if info=$(scontrol show job -o "${job}" 2>/dev/null); then
  [[ " ${info} " == *" WorkDir=${runtime} "* ]]
  [[ " ${info} " == *" Command=${runtime}/scripts/run.sbatch "* ]]
  scancel "${job}"
fi
for ((attempt=0; attempt<12; attempt++)); do
  active=$(squeue --noheader --jobs="${job}" --format='%i|%T' 2>/dev/null || true)
  [[ -n "${active}" ]] || break
  sleep 5
done
test -z "${active}"
record=$(sacct --allocations --jobs="${job}" --noheader --parsable2 --format=JobID,State,ExitCode,Elapsed,AllocNodes)
printf '%s\n' "${record}"
state=$(awk -F'|' -v id="${job}" '$1 == id {print $2}' <<<"${record}")
case "${state}" in CANCELLED*|COMPLETED|FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY) ;; *) printf 'Unconfirmed terminal state: %s\n' "${state}" >&2; exit 1 ;; esac
# Artifacts for run40563900060 were uploaded before this cancellation request.
rm -rf -- "${runtime}"
test ! -e "${runtime}"
printf 'Confirmed: job %s inactive; runtime removed; no new allocation submitted.\n' "${job}"
