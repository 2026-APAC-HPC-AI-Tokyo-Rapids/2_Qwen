#!/usr/bin/env bash
set -euo pipefail
source "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
cp "${GITHUB_WORKSPACE}/kernels/probes/prefill-serving.py" bench.py
jq -n --arg commit "${GITHUB_SHA}" '{commit:$commit,scope:"installed SGLang one-layer production integration; no distributed MoE or serving score"}' >provenance.json
