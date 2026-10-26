#!/usr/bin/env bash
set -euo pipefail
source "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
cp "${GITHUB_WORKSPACE}/kernels/probes/dense-prefill.py" bench.py
cp "${GITHUB_WORKSPACE}/scripts/check-firmus-decode-kernels.py" .
jq -n --arg commit "${GITHUB_SHA}" '{commit:$commit,kernel:"Prefill dense GEMM static N/K",scope:"one GPU, QKV and O, no serving"}' >provenance.json
