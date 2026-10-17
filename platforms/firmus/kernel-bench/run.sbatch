#!/bin/bash
#SBATCH -J megamoe-kernel-bench
#SBATCH -p hpcai
#SBATCH -N 1
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:4
#SBATCH --propagate=NONE
#SBATCH --no-requeue
# SM90 MegaMoE EP4 kernel bench inside the pinned serving image on one H200 node:
# numerical gate and timing with the candidate overlay (or the image's own
# DeepGEMM), then the instrumented timeline run compared bitwise against it.
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR}
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} node=$(hostname)"
variant=image
candidate_env=()
if [[ -d "${ROOT}/overlay/candidate" ]]; then
  variant=candidate
  candidate_env=(PYTHONPATH="${ROOT}/overlay/candidate")
fi
step() {
  local log=$1
  shift
  echo "[$(date -Iseconds)] start ${log}"
  srun -N1 -n1 --exact --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${ROOT}/sglang.sqsh" --container-mounts="/scratch:/scratch,${ROOT}/home:${HOME},${ROOT}:${ROOT}" env DG_JIT_CACHE_DIR="${ROOT}/scratch/deep-gemm-jit" "$@" >"${ROOT}/logs/${log}.log" 2>&1
  echo "[$(date -Iseconds)] done ${log}"
}

step gate "${candidate_env[@]}" /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${ROOT}/scripts/check-firmus-megamoe.py"
step bench "${candidate_env[@]}" /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${ROOT}/scripts/bench-firmus-megamoe.py" --variant "${variant}" --output "${ROOT}/results/bench.jsonl" --scratch "${ROOT}/scratch"
step timeline PYTHONPATH="${ROOT}/overlay/timeline" /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${ROOT}/scripts/bench-firmus-megamoe.py" --variant timeline --output "${ROOT}/results/timeline.jsonl" --timeline-dir "${ROOT}/results/timeline" --reference "${ROOT}/results/bench.jsonl" --scratch "${ROOT}/scratch"
touch "${ROOT}/run-success"
