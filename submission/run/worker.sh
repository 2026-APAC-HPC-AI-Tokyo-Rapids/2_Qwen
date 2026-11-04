#!/bin/bash
# Start one SGLang worker inside the serving container; submit.sbatch runs it through srun.
#
# usage: worker.sh NAME PORT MODEL TP HOST_IP [launch_server flags of this role...]
#   NAME is prefill-0, prefill-1, prefill-2 or decode-0. The role's flags come from submit.sbatch;
#   the flags every worker shares are at the end of this file.
set -euo pipefail

name=$1
port=$2
model=$3
tp=$4
export SGLANG_HOST_IP=$5
shift 5
role=${name%%-*}
runtime_dir=$(dirname "$(dirname "$0")")
python=/opt/sglang/venv/bin/python

# ---- Team switches shared by both roles ----

# Qwen PP4 path (prefill-pp4-ep1.patch); the Decode peer also needs it for the global target/draft KV region IDs.
export SGLANG_QWEN_PP4_EP1=1
# Millisecond scheduler log times: the start-burst first tokens of run 41020974704 stall for 0.8-1.5 s,
# and second-resolution batch lines cannot place the Prefill admissions and releases in that window.
export SGLANG_LOG_MS=1
# Decode admission without host waits on the WAR-fenced stream (decode-async-admission.patch). Prefill sets
# it too: there it only reaches the min_new_tokens penalizer, and prefill-async-admission.patch removes the
# other host waits that stalled each Prefill launch until the in-flight batch ended (~5 ms GPU idle per iteration).
export SGLANG_DECODE_ASYNC_ADMISSION=1

# ---- Role-specific switches ----

if [[ "${role}" == prefill ]]; then
  # Layers per pipeline stage; the last stage also runs lm_head and the EAGLE draft extend (about two layers of time).
  partition=$("${python}" -c 'import json,sys; print(",".join(map(str, json.load(open(sys.argv[1]))["prefill"]["pp_layer_partition"])))' "${runtime_dir}/source/firmus-framework.json")
  export DG_SM90_PREFILL_LOCAL_EP1=1 SGLANG_PP_LAYER_PARTITION="${partition}" SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=0
  # Stage s runs on visible device s. Move the lightest (last) stage to this worker's slowest GPU at the 700 W cap.
  gpu_order=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | "${python}" "${runtime_dir}/scripts/order-firmus-pp4-gpus.py" "${runtime_dir}/source/firmus-gpu-sm-clock.json" "${CUDA_VISIBLE_DEVICES}")
  export CUDA_VISIBLE_DEVICES="${gpu_order}"
  # A 3994-token prompt must not straddle two batches (prefill-request-aligned-chunks.patch).
  export SGLANG_PREFILL_REQUEST_ALIGNED_CHUNKS=1
  # Equal-load DP ties rotate instead of always choosing the first rank (dp-total-requests-rotate-ties.patch).
  export SGLANG_DP_TOTAL_REQUESTS_ROTATE_TIES=1
  # The Prefill MegaMoE tile uses two math warpgroups (sm90-prefill-2wg-host-epi.patch).
  export DG_SM90_PREFILL_2WG=1
  # The MegaMoE symmetric buffer holds one Prefill batch per DP rank: chunked prefill 8192 / DP 1.
  megamoe_tokens=8192
else
  # The Decode SwapAB MegaMoE gets two independent math-warpgroup task queues (sm90-decode-independent-wg patches;
  # kernel bench 39945097647: -2.3% at 128 tokens per rank).
  export DG_SM90_DECODE_INDEPENDENT_WG=1
  # Decode batches stay far below 4096 tokens per rank.
  megamoe_tokens=4096
fi
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=${megamoe_tokens} SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1

# Kernel choices adopted by the in-job gates of Actions run 41573354672, one file per role (installed by prepare.sh).
set -a
source "${runtime_dir}/source/${role}-kernels.env"
set +a

# ---- KV transfer, caches and offline mode ----

# NIXL over UCX on every active InfiniBand port of this node.
UCX_IB=$(for device in /sys/class/infiniband/*; do [[ "$(cat "${device}/ports/1/state" 2>/dev/null)" == "4: ACTIVE" ]] && echo "$(basename "${device}"):1"; done | paste -sd,)
[[ -n "${UCX_IB}" ]] || { echo "FATAL: no active HCA on $(hostname)"; exit 1; }
export UCX_NET_DEVICES=${UCX_IB} UCX_POSIX_USE_PROC_LINK=n
# Both endpoints own all KV heads (attention TP1), so NIXL moves pages directly; staging would only allocate unused pools.
export SGLANG_DISAGG_STAGING_BUFFER=0
export SGLANG_DG_CACHE_DIR="${SLURM_SUBMIT_DIR}/home/deep-gemm"
export TRITON_CACHE_DIR=/tmp/sglang-${name}/triton TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-${name}/inductor
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "${name} host=$(hostname) ip=${SGLANG_HOST_IP} hca=${UCX_IB} gpu=${CUDA_VISIBLE_DEVICES} tp=${tp}"
env | grep -E '^(SGLANG|DG)_' | sort
nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit --format=csv

# ---- Launch ----

# Queue-aware DP routing (total_requests) replaces bootstrap-room hashing on Prefill and round-robin on Decode;
# publishing the load every iteration lets completed requests release their routing budget at once.
command=("${python}" -m sglang.launch_server --model-path "${model}" --served-model-name qwen --trust-remote-code --context-length 8192 --tp-size "${tp}")
command+=(--mem-fraction-static 0.90 --page-size 64 --kv-cache-dtype fp8_e4m3 --attention-backend fa3 --nccl-port "$((port + 1000))")
command+=(--disaggregation-mode "${role}" --disaggregation-transfer-backend nixl --load-balance-method total_requests --load-snapshot-publish-interval 1)
command+=("$@" --host 0.0.0.0 --port "${port}")
# DP attention derives six TCP endpoints from this address: give each worker on a node its own range,
# below the Firmus ephemeral port range (32768-60999).
if [[ "${role}" == prefill ]]; then command+=(--dist-init-addr "127.0.0.1:$((32000 + ${name##*-} * 100))"); else command+=(--dist-init-addr "127.0.0.1:29000"); fi
exec "${command[@]}"
