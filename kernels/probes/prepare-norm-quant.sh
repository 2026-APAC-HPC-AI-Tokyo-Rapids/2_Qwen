#!/usr/bin/env bash
set -euo pipefail
source "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
cp "${GITHUB_WORKSPACE}/kernels/probes/norm-quant.py" bench.py
cp "${GITHUB_WORKSPACE}/scripts/check-firmus-decode-kernels.py" .
jq -n --arg commit "${GITHUB_SHA}" '{commit:$commit,kernel:"Prefill add-RMSNorm and group-128 FP8 quant",scope:"one GPU, no serving"}' >provenance.json
