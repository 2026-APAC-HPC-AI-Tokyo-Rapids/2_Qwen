#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?}"
bash "${RUNTIME_DIR}/scripts/run-container.sh" env DG_PRINT_CONFIGS=1 /opt/sglang/venv/bin/python "${RUNTIME_DIR}/source/bench.py"
