#!/usr/bin/env bash
#SBATCH --partition=hpcai
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00

set -euo pipefail
source "${HOME}/.local/.setenv"
module purge
module load nvhpc-hpcx-cuda13/26.5
export CONTAINER_IMAGE="${RUNTIME_DIR}/sglang.sqsh"
export ENROOT_LIBRARY_PATH="${RUNTIME_DIR}/enroot-library"
export ENROOT_SYSCONF_PATH="${RUNTIME_DIR}/enroot-system"
export ENROOT_MOUNT_HOME=no
exec "${RUNTIME_DIR}/scripts/run-qwen-pd-job.sh"
