#!/usr/bin/env bash
# Run once per node through pbsdsh, within the serving PBS allocation.
set -euo pipefail
root=${1:?runtime directory required}
node_rank=${2:?node rank required}
[[ "${node_rank}" == 0 || "${node_rank}" == 1 ]]
scratch="$(mktemp -d "${TMPDIR:-/tmp}/qwen-kernel-gates.XXXXXXXXXX")"
pids=()
cleanup() {
  status=$?
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  rm -rf -- "${scratch:?}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_screen() (
  name=$1
  gpus=$2
  shift 2
  export ENROOT_CACHE_PATH="${scratch}/${name}/cache" ENROOT_DATA_PATH="${scratch}/${name}/data" ENROOT_RUNTIME_PATH="${scratch}/${name}/runtime" ENROOT_TEMP_PATH="${scratch}/${name}/temp"
  install -d -m 0700 "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}" "${scratch}/${name}/home"
  exec enroot start -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_DISABLE_REQUIRE=1 -e "CUDA_VISIBLE_DEVICES=${gpus}" -e HOME=/runtime-home -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY=1 -e DG_SM90_DECODE_INDEPENDENT_WG=1 -e TRITON_CACHE_DIR=/runtime-home/triton -e DG_JIT_CACHE_DIR=/runtime-home/deep-gemm -e SGLANG_DG_CACHE_DIR=/runtime-home/deep-gemm -m "${scratch}/${name}/home:/runtime-home" -m "${root}:/runtime:none:bind,rw,x-create=dir" -m "${root}/overlay/sglang/python/sglang:/opt/sglang/source/python/sglang:none:bind,ro,x-create=dir" -m "${root}/overlay/deep-gemm/deep_gemm:/opt/sglang/venv/lib/python3.12/site-packages/deep_gemm:none:bind,ro,x-create=dir" "${CONTAINER_IMAGE}" "$@"
)
if [[ "${node_rank}" == 0 ]]; then
  while IFS=$'\t' read -r worker gpus; do
    run_screen "prefill-${worker}" "${gpus}" bash /runtime/scripts/select-firmus-prefill-kernels.sh /runtime "prefill-${worker}" >"${root}/logs/prefill-selection-${worker}.log" 2>&1 &
    pids+=("$!")
  done < <(jq -r '.workers[] | select(.role=="prefill") | [.index,.gpus] | @tsv' "${root}/source/aspire-serving.json")
  # The unchanged screen records a rejection/timeout and aggregation retains control.
  for pid in "${pids[@]}"; do wait "${pid}" || true; done
  pids=()
  python3 "${root}/scripts/select-firmus-prefill-fa3.py" --aggregate "${root}/results/prefill-selection" --workers 2 --output "${root}/results/prefill-selection/selection.json" --env-output "${root}/source/prefill-kernels.env"
else
  # Same numeric gate flags as Firmus; its source config is staged for the unchanged diagnostic shapes.
  run_screen numeric-gate 0,1,2,3 /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 /runtime/scripts/check-firmus-megamoe.py --dense-selected --fa3-selected >"${root}/logs/sm90-megamoe-gate.log" 2>&1
  # FA3 B40/42/44 and dense M176 selection do not represent ASPIRE B<=22 / M<=88.
  : >"${root}/source/decode-kernels.env"
fi
