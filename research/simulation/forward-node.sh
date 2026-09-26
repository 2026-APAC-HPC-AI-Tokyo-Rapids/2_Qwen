#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?runtime directory required}"
exec >"${RUNTIME_DIR}/logs/bootstrap-$(hostname -s).log" 2>&1
source "${HOME}/.local/.setenv"
: "${PBS_JOBID:?PBS allocation required}"
rank="$(awk -v host="$(hostname -s)" '$0==host {print NR-1}' "${RUNTIME_DIR}/logs/hosts.txt")"
[[ "${rank}" == 0 || "${rank}" == 1 ]]
node_tmp="$(mktemp -d "${TMPDIR:-/tmp}/qwen-forward.XXXXXXXXXX")"
pids=()
cleanup() { status=$?; trap - EXIT; for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done; for pid in "${pids[@]}"; do wait "${pid}" 2>/dev/null || true; done; rm -rf -- "${node_tmp:?}"; exit "${status}"; }
trap cleanup EXIT
export ENROOT_LIBRARY_PATH="${RUNTIME_DIR}/enroot-library" ENROOT_SYSCONF_PATH="${RUNTIME_DIR}/enroot-system" ENROOT_MOUNT_HOME=no
nvidia-smi -q -x >"${RUNTIME_DIR}/results/node-${rank}-gpu.xml"
nvidia-smi topo -m >"${RUNTIME_DIR}/results/node-${rank}-topology.txt"
plan="${RUNTIME_DIR}/source/forward-plan.json"
mapfile -t lengths < <(jq -r '.input_lengths[]' "${plan}")
mapfile -t batches < <(jq -r '. as $p | range($p.repetitions) | $p.batch_sizes[]' "${plan}")
while IFS=$'\t' read -r name tp ep gpus; do
  worker_tmp="${node_tmp}/${name}"
  install -d -m 0700 "${worker_tmp}/cache" "${worker_tmp}/data" "${worker_tmp}/runtime" "${worker_tmp}/temp" "${worker_tmp}/home"
  args=(/usr/local/bin/uv run --offline --no-project --python /opt/sglang/venv/bin/python python -m sglang.bench_one_batch --model-path /model --trust-remote-code --context-length 8192 --tp-size "${tp}" --ep-size "${ep}" --mem-fraction-static 0.90 --max-total-tokens 131072 --batch-size "${batches[@]}" --input-len "${lengths[@]}" --output-len "$(jq -r '.output_length' "${plan}")" --run-name "${name}" --result-filename "/calibration/results/${name}.jsonl")
  (export ENROOT_CACHE_PATH="${worker_tmp}/cache" ENROOT_DATA_PATH="${worker_tmp}/data" ENROOT_RUNTIME_PATH="${worker_tmp}/runtime" ENROOT_TEMP_PATH="${worker_tmp}/temp"; exec enroot start -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_DISABLE_REQUIRE=1 -e MELLANOX_VISIBLE_DEVICES=all -e MELLANOX_MOUNT_DRIVER=yes -e "CUDA_VISIBLE_DEVICES=${gpus}" -e HOME=/runtime-home -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e XDG_CACHE_HOME=/runtime-home/.cache -e FLASHINFER_WORKSPACE_BASE=/runtime-home/.cache/flashinfer -e SGLANG_DG_CACHE_DIR=/runtime-home/.cache/deep-gemm -m "${worker_tmp}/home:/runtime-home" -m "${RUNTIME_DIR}:/calibration" -m /scratch/users/industry/ai-hpc/ACCOUNT/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8:/model "${RUNTIME_DIR}/sglang.sqsh" "${args[@]}") >"${RUNTIME_DIR}/logs/${name}.log" 2>&1 &
  pids+=("$!")
done < <(jq -r --argjson node "${rank}" '.workers[] | select(.node==$node) | [.name,.tp,.ep,.gpus] | @tsv' "${plan}")
for pid in "${pids[@]}"; do wait "${pid}"; done
pids=()
