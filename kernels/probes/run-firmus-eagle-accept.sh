#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="${RUNTIME_DIR}/tokenizer-cache" SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export PYTHONPATH="${RUNTIME_DIR}/source"
exec /opt/sglang/venv/bin/python -u "${RUNTIME_DIR}/source/eagle-accept.py" "${RUNTIME_DIR}" "$(date +%s)" --firmus
