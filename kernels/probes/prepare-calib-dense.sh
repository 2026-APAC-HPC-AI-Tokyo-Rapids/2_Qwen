#!/usr/bin/env bash
set -euo pipefail
source "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
cp "${GITHUB_WORKSPACE}/kernels/probes/calib-dense.py" bench.py
cp "${GITHUB_WORKSPACE}/kernels/probes/calib-common.py" .
cp "${GITHUB_WORKSPACE}/scripts/check-firmus-decode-kernels.py" "${GITHUB_WORKSPACE}/scripts/select-firmus-decode-dense.py" .
jq -n --arg commit "${GITHUB_SHA}" --arg image "${image}" '{schema:"genkai-calibration/v1",calibration:true,commit:$commit,image:$image,kernel:"Decode dense FP8 M128 QKV/O",scope:"one GPU, two interleaved blocks, no serving",reference_runs:[39901974117]}' >provenance.json
