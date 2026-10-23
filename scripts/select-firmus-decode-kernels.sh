#!/bin/bash
# Pre-serving Decode kernel selection on four H200s, inside the SGLang container
# with the provisioning overlays mounted. Each probe checks its candidates
# numerically, times them against the current path in the same process, and
# emits env lines only for a candidate that clears its rule; everything else
# keeps the current kernels. The 17-minute site limit leaves no serial time, so
# sglang.sbatch runs "moe" (router gate GEMM, MegaMoE L2 prefetch) on the head
# node's spare GPUs during the numeric gate and "dense" on the Decode GPUs while
# the Prefill workers start; it concatenates the two env parts for Decode.
# A probe that fails or times out selects nothing: its new code is unverified
# on GPU, and one failed probe must not cost the queued serving allocation.
set -uo pipefail
root=$1
part=$2
out="${root}/results/decode-selection"
scratch="/tmp/qwen-decode-select-${SLURM_JOB_ID}-${part}"
mkdir -p "${out}" "${scratch}"
export TRITON_CACHE_DIR="${scratch}/triton" DG_JIT_CACHE_DIR="${scratch}/deep-gemm" SGLANG_DG_CACHE_DIR="${scratch}/deep-gemm"
python=/opt/sglang/venv/bin/python
torchrun=/opt/sglang/venv/bin/torchrun
env_lines='import json, sys; [print(f"{k}={v}") for k, v in json.load(open(sys.argv[1]))["env"].items()]'
: >"${out}/${part}.env"
if [[ "${part}" == dense ]]; then
  timeout 420 "${torchrun}" --standalone --nproc-per-node=4 "${root}/scripts/select-firmus-decode-dense.py" --output "${out}/dense.json" --env-output "${out}/dense-selected.env" >"${out}/dense.log" 2>&1
  dense=$?
  if (( dense == 0 )); then cat "${out}/dense-selected.env" >>"${out}/dense.env"; fi
  printf 'dense=%s\n' "${dense}" >"${out}/dense-status.txt"
else
  # No Decode expert-count dump exists; the measured Prefill frequencies stand in, labelled as such.
  PYTHONPATH="${root}/scripts" "${python}" -c 'import json, sys; from firmus_megamoe_replicas import REAL_ROUTING_LAYERS as r; json.dump({"source": {"phase": "prefill-proxy", "run_id": "39051152456", "note": "logical counts summed over the three Prefill workers; Decode keeps the trivial placement"}, "layers": [{"layer": k, "counts": v["counts"], "physical_to_logical_map": list(range(128))} for k, v in sorted(r.items())]}, open(sys.argv[1], "w"))' "${out}/routing-prefill-proxy.json"
  timeout 240 "${torchrun}" --standalone --nproc-per-node=4 "${root}/scripts/bench-firmus-sglang-kernels.py" --select-decode-gate --output "${out}/gate.json" >"${out}/gate.log" 2>&1
  gate=$?
  timeout 240 "${torchrun}" --standalone --nproc-per-node=4 "${root}/scripts/bench-firmus-megamoe.py" --variant decode-l2-probe --tune-l2-prefetch --decode-routing "${out}/routing-prefill-proxy.json" --output "${out}/l2-prefetch.json" --scratch "${scratch}/l2-prefetch" >"${out}/l2-prefetch.log" 2>&1
  l2=$?
  if (( gate == 0 )); then "${python}" -c "${env_lines}" "${out}/gate.json" >>"${out}/moe.env"; fi
  if (( l2 == 0 )); then "${python}" -c "${env_lines}" "${out}/l2-prefetch.json" >>"${out}/moe.env"; fi
  printf 'gate=%s l2=%s\n' "${gate}" "${l2}" >"${out}/moe-status.txt"
fi
cat "${out}/${part}.env"
