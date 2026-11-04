#!/usr/bin/env bash
# Host-side supervisor, always enclosed by the sbatch 105-second group timeout.
set -euo pipefail
root=${1:?runtime directory}
head_node=${2:?head node}
tail_node=${3:?tail node}
image=${4:?container image}
run_id=${5:?run id}
image_sha=${6:?image sha256}
shift 6
mount_options=("$@")
[[ ${#mount_options[@]} == 3 ]]
pids=()
codes=()
for worker in 0 1 2; do
  node=${head_node}
  if (( worker == 2 )); then node=${tail_node}; fi
  # Foreground keeps these children in the outer 105-second timeout's process
  # group; each worker also has its own 95-second deadline (97 incl. kill grace).
  timeout --foreground --kill-after=2 95 srun -N1 -n1 --exact -w "${node}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE --container-image="${image}" --container-mounts="${mount_options[worker]}" bash "${root}/scripts/run-firmus-megamoe-rescale-probe.sh" "${root}" "${worker}" "${run_id}" "${image_sha}" >"${root}/logs/megamoe-rescale-probe-${worker}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if wait "${pid}"; then codes+=(0); else codes+=("$?"); fi
done
uv run --no-project --python 3.12 python "${root}/scripts/probe-firmus-megamoe-rescale.py" --aggregate "${root}/results/megamoe-rescale-probe" --output "${root}/results/megamoe-rescale-probe/summary.json" --run-id "${run_id}" --worker-exit-codes "${codes[@]}" --image-sha256 "${image_sha}"
