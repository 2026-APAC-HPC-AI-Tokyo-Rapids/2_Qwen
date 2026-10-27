#!/bin/bash
#SBATCH -J megamoe-kernel-bench
#SBATCH -p hpcai
#SBATCH -N 1
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:4
#SBATCH --propagate=NONE
#SBATCH --no-requeue
# SM90 MegaMoE EP4 kernel bench inside the pinned serving image on one H200 node.
# The image plus configured serving overlays is the in-job reference. Each
# DeepGEMM candidate overlay then gets the numerical gate, the timing bench
# (output hashes compared with the image) and, when staged, the instrumented
# timeline run compared bitwise with its own bench, all three with the
# candidate's overlay/env/<name> KEY=VALUE lines when it has one (validated by the
# provisioning script; the image steps never get them). With an SGLang overlay the
# decode-kernel checks (and bench-firmus-sglang-kernels.py when staged) run for
# the image tree and the patched tree. Every step's exit code goes to
# results/steps.tsv and the job continues; only a failed image bench stops it.
# The summary script turns steps.tsv and the results into the verdict.
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR}
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} node=$(hostname)"
job_id=${SLURM_JOB_ID}
run_container() {
  local gpus=$1
  shift
  # Node-local Triton cache is supplied by the shared step runner.
  srun -N1 -n1 --exact --gres="gpu:${gpus}" --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${ROOT}/sglang.sqsh" --container-mounts="/scratch:/scratch,${ROOT}/home:${HOME},${ROOT}:${ROOT}" "$@"
}
source "${ROOT}/scripts/run-megamoe-kernel-bench.sh"
