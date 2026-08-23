#!/usr/bin/env bash
#SBATCH --partition=hpcai
#SBATCH --time=00:17:00

set -euo pipefail
source "${HOME}/.local/.setenv"
module purge
module load nvhpc-hpcx-cuda13/26.5
exec "${RUNTIME_DIR}/scripts/run-qwen35-job.sh"
