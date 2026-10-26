#!/bin/bash
#SBATCH -J qwen-kernel-probe
#SBATCH -p hpcai
#SBATCH -A hpcai
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH -t 00:05:00
#SBATCH --propagate=NONE
#SBATCH --no-requeue
# hpcai is node-exclusive; benchmark provenance determines step GPU count.
set -euo pipefail
: "${RUNTIME_DIR:?}"
cd "${RUNTIME_DIR}"
exec >logs/driver.log 2>&1
trap 'status=$?; printf "%s\n" "$status" > results/exit-code; printf "probe_exit_code=%s\n" "$status"' EXIT
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
date -u
cat source/provenance.json
# The metadata directory includes the checkout table missing from the old image.
probe_gpus=$(jq -er .probe_gpus source/provenance.json)
mounts="/scratch:/scratch,${RUNTIME_DIR}/home:${HOME},${RUNTIME_DIR}:${RUNTIME_DIR}"
if [[ "${probe_gpus}" == 4 ]]; then
    mounts+=",${RUNTIME_DIR}/diagnostic/opt/sglang/source/python/sglang/srt/managers/scheduler_components/batch_result_processor.py:/opt/sglang/source/python/sglang/srt/managers/scheduler_components/batch_result_processor.py:ro,/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8:/model:ro"
else
    [[ "${probe_gpus}" == 1 ]]
    mounts+=",${RUNTIME_DIR}/metadata:/opt/apac/fa3-verify:ro,${RUNTIME_DIR}/profiler/opt/nvidia/nsight-compute/2025.3.1:/opt/fa3-ncu:ro"
fi
srun -N1 -n1 --exact --gres=gpu:"${probe_gpus}" --cpus-per-task=16 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${RUNTIME_DIR}/sglang.sqsh" --container-mounts="${mounts}" env RUNTIME_DIR="${RUNTIME_DIR}" bash "${RUNTIME_DIR}/scripts/run-trials.sh"
date -u
