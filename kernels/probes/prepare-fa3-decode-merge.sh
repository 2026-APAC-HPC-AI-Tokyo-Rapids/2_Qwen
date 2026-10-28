#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --compile-only ]]; then
    install -d "${RUNTIME_DIR}/source" "${RUNTIME_DIR}/logs" "${RUNTIME_DIR}/results"
    cd "${RUNTIME_DIR}/source"
else
    source "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
fi
cp "${GITHUB_WORKSPACE}/kernels/probes/fa3-decode-merge.py" bench.py
cp "${GITHUB_WORKSPACE}/scripts/check-firmus-decode-kernels.py" .
if [[ -f "${RUNTIME_DIR}/scripts/build-env.sh" ]]; then source "${RUNTIME_DIR}/scripts/build-env.sh"; else source "${GITHUB_WORKSPACE}/platforms/genkai/kernel-probe/build-env.sh"; fi
python3 "${GITHUB_WORKSPACE}/kernels/probes/fa3-decode-merge-build.py"
