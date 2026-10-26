#!/usr/bin/env bash
# run.sbatch has already entered the single-GPU Pyxis/Enroot container.
set -euo pipefail
exec "$@"
