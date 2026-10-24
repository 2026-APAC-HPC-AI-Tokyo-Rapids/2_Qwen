#!/usr/bin/env bash
set -euo pipefail
source "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
# The reference is the same image/overlays with only this candidate reversed.
header=python/sglang/kernels/jit/csrc/elementwise/rope.cuh
install -D "${header}" "baseline/${header}"
(cd baseline && git apply -R --include="${header}" "${GITHUB_WORKSPACE}/patches/sglang/prefill-qkv-fast-fp8.patch")
mv "baseline/${header}" "${header%.cuh}-baseline.cuh"
rm -rf baseline
cp "${GITHUB_WORKSPACE}/kernels/probes/prefill-serving.py" bench.py
jq -n --arg commit "${GITHUB_SHA}" '{commit:$commit,scope:"installed SGLang one-layer production integration; no distributed MoE or serving score"}' >provenance.json
