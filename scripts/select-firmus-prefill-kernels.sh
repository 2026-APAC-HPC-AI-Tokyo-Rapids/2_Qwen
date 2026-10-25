#!/bin/bash
# Run inside each Prefill worker's allocation, on exactly its four visible GPUs.
# FA3 and exact QK-norm screens share the same 175-second process budget.
# A failed screen leaves the worker on the fixed controls.
set -uo pipefail
root=$1
worker=$2
out="${root}/results/prefill-selection/${worker}"
mkdir -p "${out}"
timeout --kill-after=5 175 /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${root}/scripts/select-firmus-prefill-fa3.py" --output "${out}/fa3.json" >"${out}/fa3.log" 2>&1
status=$?
printf 'fa3=%s\n' "${status}" >"${out}/status.txt"
cat "${out}/status.txt"
