#!/bin/bash
# Pre-serving Decode kernel selection on four H200s, inside the SGLang container
# with the provisioning overlays mounted. Each probe checks its candidates
# numerically, times them against the current path in the same process, and
# emits env lines only for a candidate that clears its rule; everything else
# keeps the current kernels. The Decode worker sources the combined env file.
# A probe that fails or times out selects nothing: its new code is unverified
# on GPU, and one failed probe must not cost the queued serving allocation.
set -uo pipefail
root=$1
out="${root}/results/decode-selection"
scratch="/tmp/qwen-decode-select-${SLURM_JOB_ID}"
mkdir -p "${out}" "${scratch}"
export TRITON_CACHE_DIR="${scratch}/triton" DG_JIT_CACHE_DIR="${scratch}/deep-gemm" SGLANG_DG_CACHE_DIR="${scratch}/deep-gemm"
python=/opt/sglang/venv/bin/python
torchrun=/opt/sglang/venv/bin/torchrun
declare -A status
# No Decode expert-count dump exists; the measured Prefill frequencies stand in, labelled as such.
PYTHONPATH="${root}/scripts" "${python}" -c 'import json, sys; from firmus_megamoe_replicas import REAL_ROUTING_LAYERS as r; json.dump({"source": {"phase": "prefill-proxy", "run_id": "39051152456", "note": "logical counts summed over the three Prefill workers; Decode keeps the trivial placement"}, "layers": [{"layer": k, "counts": v["counts"], "physical_to_logical_map": list(range(128))} for k, v in sorted(r.items())]}, open(sys.argv[1], "w"))' "${out}/routing-prefill-proxy.json"
timeout 1200 "${torchrun}" --standalone --nproc-per-node=4 "${root}/scripts/select-firmus-decode-dense.py" --output "${out}/dense.json" --env-output "${out}/dense.env" >"${out}/dense.log" 2>&1
status[dense]=$?
timeout 600 "${torchrun}" --standalone --nproc-per-node=4 "${root}/scripts/bench-firmus-sglang-kernels.py" --select-decode-gate --output "${out}/gate.json" >"${out}/gate.log" 2>&1
status[gate]=$?
timeout 600 "${torchrun}" --standalone --nproc-per-node=4 "${root}/scripts/bench-firmus-megamoe.py" --variant decode-l2-probe --tune-l2-prefetch --decode-routing "${out}/routing-prefill-proxy.json" --output "${out}/l2-prefetch.json" --scratch "${scratch}/l2-prefetch" >"${out}/l2-prefetch.log" 2>&1
status[l2]=$?
: >"${root}/source/decode-kernels.env"
if (( status[dense] == 0 )); then cat "${out}/dense.env" >>"${root}/source/decode-kernels.env"; fi
if (( status[gate] == 0 )); then "${python}" -c 'import json, sys; [print(f"{k}={v}") for k, v in json.load(open(sys.argv[1]))["env"].items()]' "${out}/gate.json" >>"${root}/source/decode-kernels.env"; fi
if (( status[l2] == 0 )); then "${python}" -c 'import json, sys; [print(f"{k}={v}") for k, v in json.load(open(sys.argv[1]))["env"].items()]' "${out}/l2-prefetch.json" >>"${root}/source/decode-kernels.env"; fi
printf 'dense=%s gate=%s l2=%s\n' "${status[dense]}" "${status[gate]}" "${status[l2]}" >"${out}/status.txt"
cat "${out}/status.txt" "${root}/source/decode-kernels.env"
