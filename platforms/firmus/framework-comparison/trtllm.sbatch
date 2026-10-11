#!/bin/bash
#SBATCH -J trtllm-firmus-p6d1
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:8
#SBATCH --propagate=NONE
#SBATCH --no-requeue
#SBATCH -o runs/trtllm-%j.out
#SBATCH -e runs/trtllm-%j.err
# TensorRT-LLM PD with the layout of NVIDIA's Dynamo Qwen3-235B-A22B-FP8 Hopper recipe:
# six TP2 context workers (four on head, two on tail) and one TP4/EP4 generation worker on tail.
# Native trtllm-serve workers and disaggregated gateway, official 60/120/30 client.
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR}
TRTLLM=${ROOT}/trtllm.sqsh
AIPERF=${ROOT}/sglang.sqsh
MODEL=/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8
R_PORT=8000
READY_SECS=660  # Leave the 60/120/30 client and drain inside the 17-minute partition limit.
N=$(jq -er .concurrency "${ROOT}/source/firmus-framework.json")
ART=${ROOT}/logs
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_DISABLE_REQUIRE=1
export MELLANOX_VISIBLE_DEVICES=all MELLANOX_MOUNT_DRIVER=yes
[[ "${PROFILE:-false}" == false ]]
jq -e '.framework == "trtllm" and .workload == "official"' "${ROOT}/source/firmus-framework.json" >/dev/null

NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD=${NODES[0]}
TAIL=${NODES[1]}
ip_of() { getent ahostsv4 "$1" | awk '{print $1; exit}'; }
HEAD_IP=$(ip_of "${HEAD}")
TAIL_IP=$(ip_of "${TAIL}")

mounts() {
  mkdir -p "${ROOT}/home/$1"
  echo "/scratch:/scratch,${ROOT}/home/$1:${HOME},${ROOT}:${ROOT},/dev/infiniband:/dev/infiniband"
}

wait_http() {
  local deadline=$((SECONDS + READY_SECS)) pid status
  while true; do
    for pid in "${worker_pids[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        status=0
        wait "${pid}" || status=$?
        printf 'server_exited pid=%s status=%s waiting_for=%s\n' "${pid}" "${status}" "$1" >&2
        if (( status == 0 )); then status=1; fi
        return "${status}"
      fi
    done
    if curl --fail --silent --max-time 2 -o /dev/null "$1"; then return 0; fi
    if (( SECONDS >= deadline )); then printf 'server_readiness_timeout url=%s\n' "$1" >&2; return 124; fi
    sleep 2
  done
}

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} user=$(whoami)"
echo "head=${HEAD}(${HEAD_IP}) tail=${TAIL}(${TAIL_IP}) model=${MODEL} isl=4000 osl=200 N=${N} topology=P6TP2-D1TP4EP4 engine=trtllm kv=fp8 speculation=none"
test -r "${TRTLLM}" && test -r "${AIPERF}" && test -f "${MODEL}/config.json"
mkdir -p "${ART}"
worker_pids=()
cleanup_all() {
  status=$?
  trap - EXIT INT TERM
  for pid in "${worker_pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${worker_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup_all EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for worker in 0 1 2 3; do printf 'prefill-%s\t%s\t%s\t2\n' "${worker}" "${HEAD_IP}" "$((30100 + worker))"; done >"${ROOT}/source/worker-endpoints.tsv"
for worker in 4 5; do printf 'prefill-%s\t%s\t%s\t2\n' "${worker}" "${TAIL_IP}" "$((30100 + worker))"; done >>"${ROOT}/source/worker-endpoints.tsv"
printf 'decode-0\t%s\t30200\t4\n' "${TAIL_IP}" >>"${ROOT}/source/worker-endpoints.tsv"

# Engine settings follow the recipe's prefill.yaml/decode.yaml; block reuse stays off because cache-bust defeats it.
cat >"${ROOT}/source/context.yaml" <<'EOF'
backend: pytorch
trust_remote_code: true
tensor_parallel_size: 2
moe_tensor_parallel_size: 2
moe_expert_parallel_size: 1
enable_attention_dp: false
enable_chunked_prefill: false
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.7
  dtype: fp8
cache_transceiver_config:
  backend: DEFAULT
cuda_graph_config:
  enable_padding: true
  max_batch_size: 2
disable_overlap_scheduler: true
print_iter_log: true
EOF
cat >"${ROOT}/source/generation.yaml" <<'EOF'
backend: pytorch
trust_remote_code: true
tensor_parallel_size: 4
moe_expert_parallel_size: 4
moe_tensor_parallel_size: 1
enable_attention_dp: false
enable_chunked_prefill: false
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.9
  dtype: fp8
cache_transceiver_config:
  backend: DEFAULT
cuda_graph_config:
  enable_padding: true
  max_batch_size: 512
disable_overlap_scheduler: false
print_iter_log: false
EOF
{
  printf 'hostname: 0.0.0.0\nport: %s\nbackend: pytorch\ncontext_servers:\n  num_instances: 6\n  urls:\n' "${R_PORT}"
  awk -F'\t' '$1 ~ /^prefill/ {printf "    - \"%s:%s\"\n", $2, $3}' "${ROOT}/source/worker-endpoints.tsv"
  printf 'generation_servers:\n  num_instances: 1\n  urls:\n'
  awk -F'\t' '$1 ~ /^decode/ {printf "    - \"%s:%s\"\n", $2, $3}' "${ROOT}/source/worker-endpoints.tsv"
} >"${ROOT}/source/disagg.yaml"

cat >"${ROOT}/source/worker.sh" <<'EOF'
set -euo pipefail
name=$1
port=$2
model=$3
tp=$4
ep=$5
root=$6
role=${name%%-*}
UCX_IB=$(for d in /sys/class/infiniband/*; do
  [ "$(cat "$d/ports/1/state" 2>/dev/null)" = "4: ACTIVE" ] || continue
  echo "$(basename "$d"):1"
done | paste -sd,)
[ -n "$UCX_IB" ] || { echo "FATAL: no active HCA on $(hostname)"; exit 1; }
export UCX_NET_DEVICES=$UCX_IB
export UCX_POSIX_USE_PROC_LINK=n
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# mpirun below starts the local TP ranks; hide the enclosing Slurm step so Open MPI launches locally.
for variable in $(compgen -e | grep -E '^(SLURM_|PMI_|PMIX_)' || true); do unset "${variable}"; done
if [[ "$role" == prefill ]]; then server_role=CONTEXT; config=$root/source/context.yaml; batch=2; tokens=8192; else server_role=GENERATION; config=$root/source/generation.yaml; batch=512; tokens=1024; fi
echo "$name host=$(hostname) hca=$UCX_IB gpu=$CUDA_VISIBLE_DEVICES tp=$tp ep=$ep role=$server_role batch=$batch tokens=$tokens"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit --format=csv
python3 -c "from importlib.metadata import version; print('tensorrt_llm', version('tensorrt_llm'))"
if [[ "$name" == prefill-0 ]]; then
  # The gateway needs the CUDA driver to import tensorrt_llm; it polls every worker until ready.
  trtllm-serve disaggregated -c "$root/source/disagg.yaml" -t 900 -r 1800 >"$root/logs/gateway.log" 2>&1 &
fi
exec mpirun --oversubscribe --bind-to none -np "$tp" trtllm-llmapi-launch trtllm-serve "$model" --host 0.0.0.0 --port "$port" --backend pytorch --served_model_name qwen --tp_size "$tp" --ep_size "$ep" --max_batch_size "$batch" --max_num_tokens "$tokens" --max_seq_len 8192 --server_role "$server_role" --config "$config"
EOF

for worker in 0 1 2 3; do
  srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:2 --cpus-per-task=16 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${TRTLLM}" --container-mounts="$(mounts "prefill-${worker}")" bash "${ROOT}/source/worker.sh" "prefill-${worker}" "$((30100 + worker))" "${MODEL}" 2 1 "${ROOT}" >"${ART}/prefill-${worker}.log" 2>&1 &
  worker_pids+=("$!")
done
for worker in 4 5; do
  srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:2 --cpus-per-task=16 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${TRTLLM}" --container-mounts="$(mounts "prefill-${worker}")" bash "${ROOT}/source/worker.sh" "prefill-${worker}" "$((30100 + worker))" "${MODEL}" 2 1 "${ROOT}" >"${ART}/prefill-${worker}.log" 2>&1 &
  worker_pids+=("$!")
done
srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${TRTLLM}" --container-mounts="$(mounts decode-0)" bash "${ROOT}/source/worker.sh" decode-0 30200 "${MODEL}" 4 4 "${ROOT}" >"${ART}/decode-0.log" 2>&1 &
worker_pids+=("$!")

while IFS=$'\t' read -r worker host port _tp; do
  wait_http "http://${host}:${port}/health"; echo "${worker} ready after ${SECONDS}s"
done <"${ROOT}/source/worker-endpoints.tsv"
wait_http "http://${HEAD_IP}:${R_PORT}/health"
echo "READY after ${SECONDS}s"
curl --fail --silent --show-error --max-time 15 "http://${HEAD_IP}:${R_PORT}/v1/models" >"${ROOT}/results/gateway-models.json"

srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=32 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${ART}/aiperf.log" 2>&1
echo "AIPERF ok"
touch "${ROOT}/run-success"
