#!/usr/bin/env bash
#SBATCH -J vllm-sm90-megamoe-gate
#SBATCH -p hpcai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH -t 00:15:00
#SBATCH --propagate=NONE
#SBATCH --no-requeue
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:?}
[[ "${SLURM_NNODES}" == 1 ]]
export NVIDIA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:?}
export NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
export NCCL_DEBUG=WARN
srun --mpi=none --export=ALL --gpu-bind=none -N1 -n1 --gres=gpu:4 --cpus-per-task=16 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,NCCL_DEBUG --container-image="${ROOT}/vllm.sqsh" --container-mounts="${ROOT}:${ROOT}" --container-workdir="${ROOT}" bash -c 'export CUDA_VISIBLE_DEVICES=0,1,2,3; export HOME="$1/home" XDG_CACHE_HOME="$1/home/.cache" TRITON_CACHE_DIR="$1/home/.triton"; exec torchrun --standalone --nproc_per_node=4 "$1/scripts/megamoe/gpu_gate.py" --output "$1/results/gate.json"' gate "${ROOT}" >"${ROOT}/logs/gpu-gate.log" 2>&1
test -s "${ROOT}/results/gate.json"
touch "${ROOT}/run-success"
