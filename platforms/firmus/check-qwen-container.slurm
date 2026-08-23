#!/usr/bin/env bash
#SBATCH --partition=hpcai
#SBATCH --time=00:17:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
source "${HOME}/.local/.setenv"
exec "${RUNTIME_DIR}/scripts/run-qwen-check-env.sh"
