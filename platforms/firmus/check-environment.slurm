#!/usr/bin/env bash
#SBATCH --partition=hpcai
#SBATCH --time=00:02:00

set -euo pipefail
source "${HOME}/.local/.setenv"
module purge
module load nvhpc-hpcx-cuda13/26.5
test -n "${FIRMUS_RUN_DIR}"
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -lc 'set -euo pipefail; source "${HOME}/.local/.setenv"; module purge; module load nvhpc-hpcx-cuda13/26.5; hostname -f; nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv; mpirun --version; ucx_info -v; nvcc --version; command -v enroot || true; enroot version || true; command -v nvidia-container-cli; nvidia-container-cli --version'
