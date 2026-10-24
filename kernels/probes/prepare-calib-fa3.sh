#!/usr/bin/env bash
set -euo pipefail
source "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
cp "${GITHUB_WORKSPACE}/kernels/probes/calib-fa3.py" bench.py
cp "${GITHUB_WORKSPACE}/kernels/probes/calib-common.py" .
cp "${GITHUB_WORKSPACE}/scripts/check-firmus-decode-kernels.py" "${GITHUB_WORKSPACE}/scripts/select-firmus-decode-fa3.py" .
jq -n --arg commit "${GITHUB_SHA}" --arg image "${image}" '{schema:"genkai-calibration/v1",calibration:true,commit:$commit,image:$image,kernel:"FA3 TARGET_VERIFY B32 Q4",scope:"one GPU, shared metadata, 94 layers, two paired blocks, no serving",reference_runs:[39901974117]}' >provenance.json
