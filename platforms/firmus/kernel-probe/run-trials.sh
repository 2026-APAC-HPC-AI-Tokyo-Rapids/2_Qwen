#!/usr/bin/env bash
# Runs inside the one-GPU Slurm container step, after image startup.
set -euo pipefail
: "${RUNTIME_DIR:?}"
cd "${RUNTIME_DIR}"
export CUDA_VISIBLE_DEVICES=0
export PATH="${RUNTIME_DIR}/bin:${PATH}"
export PYTHONPATH="${RUNTIME_DIR}/source/python"
node_tmp=$(mktemp -d /tmp/firmus-probe.XXXXXXXX)
trap 'rm -rf -- "${node_tmp:?}"' EXIT
export TVM_FFI_CACHE_DIR="${node_tmp}/tvm" TRITON_CACHE_DIR="${node_tmp}/triton" DG_JIT_CACHE_DIR="${node_tmp}/deep-gemm" DG_PRINT_CONFIGS=1
# Check a hardware metric, not merely kernel tracing, for M1 as well as M2.
printf 'Checking Nsight Compute GPU performance-counter permission on one GPU\n'
status=0
timeout 30 ncu --metrics sm__cycles_elapsed.avg --launch-count 1 --clock-control none --csv "${RUNTIME_DIR}/source/ncu-permission" >results/ncu-permission.csv 2>&1 || status=$?
cat results/ncu-permission.csv
if grep -q ERR_NVGPUCTRPERM results/ncu-permission.csv; then printf 'FATAL: ERR_NVGPUCTRPERM: Firmus GPU performance counters are not permitted; no probe trial was run.\n' >&2; exit 1; fi
if (( status != 0 )); then printf 'FATAL: Nsight Compute permission preflight failed (exit=%s); no probe trial was run.\n' "${status}" >&2; exit "${status}"; fi
grep -q 'sm__cycles_elapsed.avg' results/ncu-permission.csv || { printf 'FATAL: NCU preflight produced no hardware-counter result; no probe trial was run.\n' >&2; exit 1; }
python3 -c 'import json, sys; sys.exit(0 if json.load(open("source/provenance.json")).get("calibration") is True else 1)'
printf 'trial=1\n'
timeout 165 ./source/probe 2>&1 | tee results/trial-1.log
grep -q '^PASS ' results/trial-1.log
touch run-success
