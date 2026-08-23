#!/usr/bin/env bash
#SBATCH --partition=shared
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
source "${HOME}/.local/.setenv"
exec "${RUNTIME_DIR}/scripts/run-qwen-check-env.sh"
