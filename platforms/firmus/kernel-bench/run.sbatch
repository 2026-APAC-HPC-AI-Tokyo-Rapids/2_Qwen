#!/bin/bash
#SBATCH -J megamoe-kernel-bench
#SBATCH -p hpcai
#SBATCH -N 1
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:4
#SBATCH --propagate=NONE
#SBATCH --no-requeue
# SM90 MegaMoE EP4 kernel bench inside the pinned serving image on one H200 node.
# The image plus configured serving overlays is the in-job reference. Each
# DeepGEMM candidate overlay then gets the numerical gate, the timing bench
# (output hashes compared with the image) and, when staged, the instrumented
# timeline run compared bitwise with its own bench, all three with the
# candidate's overlay/env/<name> KEY=VALUE lines when it has one (validated by the
# provisioning script; the image steps never get them). With an SGLang overlay the
# decode-kernel checks (and bench-firmus-sglang-kernels.py when staged) run for
# the image tree and the patched tree. Every step's exit code goes to
# results/steps.tsv and the job continues; only a failed image bench stops it.
# The summary script turns steps.tsv and the results into the verdict.
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR}
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} node=$(hostname)"
torchrun=(/opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4)
step_exit=0
step() {
  local variant=$1 name=$2 started=${SECONDS}
  shift 2
  echo "[$(date -Iseconds)] start ${variant} ${name}"
  step_exit=0
  # Match serving's Prefill geometry for both the control and every candidate.
  # Also exercise the 256-row masked idle rank in the numerical gate.
  srun -N1 -n1 --exact --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${ROOT}/sglang.sqsh" --container-mounts="/scratch:/scratch,${ROOT}/home:${HOME},${ROOT}:${ROOT}" env DG_SM90_PREFILL_2WG=1 SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY=1 DG_JIT_CACHE_DIR="${ROOT}/scratch/deep-gemm-jit" "$@" >"${ROOT}/logs/${variant}-${name}.log" 2>&1 || step_exit=$?
  printf '%s\t%s\t%s\t%s\n' "${variant}" "${name}" "${step_exit}" "$((SECONDS - started))" >>"${ROOT}/results/steps.tsv"
  echo "[$(date -Iseconds)] done ${variant} ${name} exit=${step_exit}"
}
timeline() {
  local variant=$1
  shift
  if [[ -d "${ROOT}/overlay/timeline/${variant}" ]]; then step "${variant}" timeline PYTHONPATH="${ROOT}/overlay/timeline/${variant}" "$@" "${torchrun[@]}" "${ROOT}/scripts/bench-firmus-megamoe.py" --variant "${variant}+timeline" --output "${ROOT}/results/${variant}/timeline.jsonl" --timeline-dir "${ROOT}/results/${variant}/timeline" --reference "${ROOT}/results/${variant}/bench.jsonl" --scratch "${ROOT}/scratch/${variant}"; fi
}

install -d "${ROOT}/results/image" "${ROOT}/scratch/image"
step image bench PYTHONPATH="${ROOT}/image/opt/sglang/venv/lib/python3.12/site-packages" "${torchrun[@]}" "${ROOT}/scripts/bench-firmus-megamoe.py" --variant image --output "${ROOT}/results/image/bench.jsonl" --scratch "${ROOT}/scratch/image"
(( step_exit == 0 ))
timeline image

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

for dir in "${ROOT}"/overlay/candidates/*/; do
  [[ -d "${dir}" ]] || continue
  variant="$(basename "${dir}")"
  candidate_env=()
  if [[ -e "${ROOT}/overlay/env/${variant}" ]]; then mapfile -t candidate_env <"${ROOT}/overlay/env/${variant}"; fi
  install -d "${ROOT}/results/${variant}" "${ROOT}/scratch/${variant}"
  step "${variant}" gate PYTHONPATH="${dir%/}" "${candidate_env[@]}" "${torchrun[@]}" "${ROOT}/scripts/check-firmus-megamoe.py" --megamoe-only
  (( step_exit == 0 )) || continue
  step "${variant}" bench PYTHONPATH="${dir%/}" "${candidate_env[@]}" "${torchrun[@]}" "${ROOT}/scripts/bench-firmus-megamoe.py" --variant "${variant}" --output "${ROOT}/results/${variant}/bench.jsonl" --reference "${ROOT}/results/image/bench.jsonl" --scratch "${ROOT}/scratch/${variant}"
  (( step_exit == 0 )) || continue
  timeline "${variant}" "${candidate_env[@]}"
done
touch "${ROOT}/run-success"
