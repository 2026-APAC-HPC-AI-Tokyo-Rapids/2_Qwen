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
# hpcai is node-exclusive: allocate the node, restrict the sole step to one GPU.
set -euo pipefail
: "${RUNTIME_DIR:?}"
cd "${RUNTIME_DIR}"
exec >logs/driver.log 2>&1
trap 'status=$?; printf "%s\n" "$status" > results/exit-code; printf "probe_exit_code=%s\n" "$status"' EXIT
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
date -u
cat source/provenance.json
# The metadata directory includes the checkout table missing from the old image.
srun -N1 -n1 --exact --gres=gpu:1 --cpus-per-task=16 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${RUNTIME_DIR}/sglang.sqsh" --container-mounts="/scratch:/scratch,${RUNTIME_DIR}/home:${HOME},${RUNTIME_DIR}:${RUNTIME_DIR},${RUNTIME_DIR}/metadata:/opt/apac/fa3-verify:ro,${RUNTIME_DIR}/profiler/opt/nvidia/nsight-compute/2025.3.1:/opt/fa3-ncu:ro" env RUNTIME_DIR="${RUNTIME_DIR}" bash "${RUNTIME_DIR}/scripts/run-trials.sh"
date -u
