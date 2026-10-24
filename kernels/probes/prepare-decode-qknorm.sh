#!/usr/bin/env bash
set -euo pipefail
source "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
# Compare the identical serving chain with only the new conversion reversed.
header=python/sglang/kernels/jit/csrc/elementwise/rope.cuh
install -D "${header}" "baseline/${header}"
(cd baseline && git apply -R --include="${header}" "${GITHUB_WORKSPACE}/patches/sglang/decode-qknorm-fast-fp8.patch")
mv "baseline/${header}" "${header%.cuh}-decode-baseline.cuh"
rm -rf baseline
cp "${GITHUB_WORKSPACE}/kernels/probes/decode-qknorm.py" bench.py
cp "${GITHUB_WORKSPACE}/scripts/check-firmus-decode-kernels.py" .
jq -n --arg commit "${GITHUB_SHA}" --arg image "${image}" '{commit:$commit,image:$image,kernel:"Decode QK-norm/RoPE exact hardware FP8",scope:"one GPU, same serving overlays, only shared FP8 helper differs; no serving score"}' >provenance.json
