#!/usr/bin/env bash
# Shared EP4 experiment protocol; source after defining ROOT, job_id and run_container.
torchrun=(/opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4)
prefill_memory_args=()
campaign="${ROOT}/source/patches/deep-gemm-bench"
if [[ -e "${campaign}/prefill-memory" ]]; then prefill_memory_args=(--prefill-memory); fi
if [[ -e "${campaign}/prefill-only" ]]; then prefill_memory_args=(--prefill-only); fi
timeline_candidates=()
step_exit=0
step() {
  local variant=$1 name=$2 started=${SECONDS} gpus=4
  if [[ "${variant}" == prefill-memory ]]; then gpus=1; fi
  shift 2
  echo "[$(date -Iseconds)] start ${variant} ${name}"
  step_exit=0
  # Keep allocation/container mechanics in the platform wrapper.
  run_container "${gpus}" env DG_SM90_PREFILL_2WG=1 DG_SM90_DECODE_INDEPENDENT_WG=1 SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY=1 DG_JIT_CACHE_DIR="${ROOT}/scratch/deep-gemm-jit" TRITON_CACHE_DIR="/tmp/qwen-kernel-bench-${job_id}-${variant}-${name}/triton" "$@" >"${ROOT}/logs/${variant}-${name}.log" 2>&1 || step_exit=$?
  printf '%s\t%s\t%s\t%s\n' "${variant}" "${name}" "${step_exit}" "$((SECONDS - started))" >>"${ROOT}/results/steps.tsv"
  echo "[$(date -Iseconds)] done ${variant} ${name} exit=${step_exit}"
}
timeline() {
  local variant=$1
  shift
  if [[ -d "${ROOT}/overlay/timeline/${variant}" ]]; then step "${variant}" timeline PYTHONPATH="${ROOT}/overlay/timeline/${variant}" "$@" "${torchrun[@]}" "${ROOT}/scripts/bench-firmus-megamoe.py" --variant "${variant}+timeline" "${prefill_memory_args[@]}" --output "${ROOT}/results/${variant}/timeline.jsonl" --timeline-dir "${ROOT}/results/${variant}/timeline" --reference "${ROOT}/results/${variant}/bench.jsonl" --scratch "${ROOT}/scratch/${variant}"; fi
}

install -d "${ROOT}/results/image" "${ROOT}/scratch/image"
step image bench PYTHONPATH="${ROOT}/image/opt/sglang/venv/lib/python3.12/site-packages" "${torchrun[@]}" "${ROOT}/scripts/bench-firmus-megamoe.py" --variant image "${prefill_memory_args[@]}" --output "${ROOT}/results/image/bench.jsonl" --scratch "${ROOT}/scratch/image"
(( step_exit == 0 ))
if [[ ! -e "${campaign}/prefill-only" ]]; then timeline image; fi

if [[ -d "${ROOT}/overlay/sglang" ]]; then
  # The candidate builds its JIT kernels in its own TVM-FFI cache; the image
  # tree keeps the default cache under HOME that the MegaMoE gate also fills.
  # The resolver asserts which tree sglang.kernels.jit is imported from.
  resolve='import runpy, sys, sglang.kernels.jit as jit; print(jit.__file__, flush=True); assert jit.__file__.startswith(sys.argv[1] + "/"), jit.__file__; runpy.run_path(sys.argv[2], run_name="__main__")'
  install -d "${ROOT}/results/sglang"
  for variant in image candidate; do
    tree=/opt/sglang/source/python
    variant_env=()
    if [[ "${variant}" == candidate ]]; then tree="${ROOT}/overlay/sglang/python"; variant_env=(PYTHONPATH="${tree}" TVM_FFI_CACHE_DIR="${ROOT}/scratch/tvm-ffi-candidate"); fi
    step "sglang-${variant}" check "${variant_env[@]}" /opt/sglang/venv/bin/python -c "${resolve}" "${tree}" "${ROOT}/scripts/check-firmus-decode-kernels.py"
    if [[ -e "${ROOT}/scripts/bench-firmus-sglang-kernels.py" ]]; then step "sglang-${variant}" bench "${variant_env[@]}" /opt/sglang/venv/bin/python "${ROOT}/scripts/bench-firmus-sglang-kernels.py" --output "${ROOT}/results/sglang/${variant}.jsonl" --variant "${variant}"; fi
  done
fi

# Omit the trailing slash from the glob so a base candidate precedes its
# longer combined name (bulk-scatter before bulk-scatter-wg1-skip).
for dir in "${ROOT}"/overlay/candidates/*; do
  [[ -d "${dir}" ]] || continue
  variant="$(basename "${dir}")"
  candidate_env=()
  if [[ -e "${ROOT}/overlay/env/${variant}" ]]; then mapfile -t candidate_env <"${ROOT}/overlay/env/${variant}"; fi
  install -d "${ROOT}/results/${variant}" "${ROOT}/scratch/${variant}"
  step "${variant}" gate PYTHONPATH="${dir%/}" "${candidate_env[@]}" "${torchrun[@]}" "${ROOT}/scripts/check-firmus-megamoe.py" --megamoe-only
  (( step_exit == 0 )) || continue
  step "${variant}" bench PYTHONPATH="${dir%/}" "${candidate_env[@]}" "${torchrun[@]}" "${ROOT}/scripts/bench-firmus-megamoe.py" --variant "${variant}" "${prefill_memory_args[@]}" --output "${ROOT}/results/${variant}/bench.jsonl" --reference "${ROOT}/results/image/bench.jsonl" --scratch "${ROOT}/scratch/${variant}"
  (( step_exit == 0 )) || continue
  if [[ -e "${campaign}/prefill-only" ]]; then timeline_candidates+=("${variant}"); else timeline "${variant}" "${candidate_env[@]}"; fi
done
# The QKV JIT uses the configured serving source, not the pinned image source.
# One GPU stays visible; the other three allocation GPUs are unused in this step.
if [[ -e "${campaign}/prefill-memory" ]]; then
  qkv_resolve='import runpy, sys, sglang.kernels.jit as jit; print(jit.__file__, flush=True); assert jit.__file__.startswith(sys.argv[1] + "/"), jit.__file__; sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0], run_name="__main__")'
  step prefill-memory probe CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${ROOT}/image/opt/sglang/source/python" TVM_FFI_CACHE_DIR="${ROOT}/scratch/tvm-ffi-prefill-memory" /opt/sglang/venv/bin/python -c "${qkv_resolve}" "${ROOT}/image/opt/sglang/source/python" "${ROOT}/scripts/prefill-memory.py" --header "${ROOT}/source/prefill-memory/rope-heads1.cuh" --output "${ROOT}/results/prefill-memory" --run
fi
# A final control brackets the focused MoE experiment independently of QKV.
if (( ${#prefill_memory_args[@]} )); then
  install -d "${ROOT}/results/control-repeat" "${ROOT}/scratch/control-repeat"
  step control-repeat bench PYTHONPATH="${ROOT}/image/opt/sglang/venv/lib/python3.12/site-packages" "${torchrun[@]}" "${ROOT}/scripts/bench-firmus-megamoe.py" --variant control-repeat "${prefill_memory_args[@]}" --output "${ROOT}/results/control-repeat/bench.jsonl" --reference "${ROOT}/results/image/bench.jsonl" --scratch "${ROOT}/scratch/control-repeat"
fi
# Keep instrumented calls after the uninstrumented image/candidate/repeat sequence.
if [[ -e "${campaign}/prefill-only" ]]; then
  timeline image
  for variant in "${timeline_candidates[@]}"; do
    candidate_env=()
    if [[ -e "${ROOT}/overlay/env/${variant}" ]]; then mapfile -t candidate_env <"${ROOT}/overlay/env/${variant}"; fi
    timeline "${variant}" "${candidate_env[@]}"
  done
fi
touch "${ROOT}/run-success"
