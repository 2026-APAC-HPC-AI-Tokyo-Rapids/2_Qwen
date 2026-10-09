#!/usr/bin/env bash
#SBATCH --partition=hpcai
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=00:17:00

set -euo pipefail
if [[ "${QWEN_PLATFORM:-firmus}" == aspire2ap ]]; then
  set -E
  trap 'printf "submission_failed mode=%s line=%s status=%s\n" "${1:-orchestrate}" "${LINENO}" "$?" >&2' ERR
fi
caller_enroot_library_path="${ENROOT_LIBRARY_PATH:-}"
caller_enroot_mount_home="${ENROOT_MOUNT_HOME:-}"
caller_enroot_sysconf_path="${ENROOT_SYSCONF_PATH:-}"
source "${HOME}/.local/.setenv"
if [[ -n "${caller_enroot_library_path}" ]]; then export ENROOT_LIBRARY_PATH="${caller_enroot_library_path}"; fi
if [[ -n "${caller_enroot_mount_home}" ]]; then export ENROOT_MOUNT_HOME="${caller_enroot_mount_home}"; fi
if [[ -n "${caller_enroot_sysconf_path}" ]]; then export ENROOT_SYSCONF_PATH="${caller_enroot_sysconf_path}"; fi
if [[ "${QWEN_PLATFORM:-firmus}" == firmus ]]; then module purge; module load nvhpc-hpcx-cuda13/26.5; fi

script_path="$(realpath "${BASH_SOURCE[0]}")"
submission_dir="$(cd "$(dirname "${script_path}")/.." && pwd)"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-${submission_dir}/build/sglang.sqsh}"
export MODEL_PATH="${MODEL_PATH:-/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
export OUTPUT_DIR="${OUTPUT_DIR:-${submission_dir}/results/logs/${SLURM_JOB_ID}}"
export PROFILE="${PROFILE:-false}"
export CONCURRENCY="${CONCURRENCY:-16}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e}"
export QWEN_PLATFORM="${QWEN_PLATFORM:-firmus}"
export NODE_COUNT="${NODE_COUNT:-2}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
mode="${1:-orchestrate}"
if [[ "${QWEN_PLATFORM}" == aspire2ap ]]; then CONCURRENCY="$(jq -er ' .concurrency' "${OUTPUT_DIR}/source/aspire-serving.json")"; fi

sample_system() {
  local rank="$1" profile_dir="$2" timestamp user nice system idle iowait irq softirq steal interface rx_bytes rx_packets rx_errors tx_bytes tx_packets tx_errors counters device
  local cpu_output="${profile_dir}/cpu-node-${rank}.csv"
  local network_output="${profile_dir}/network-node-${rank}.csv"
  local infiniband_output="${profile_dir}/infiniband-node-${rank}.csv"
  printf 'timestamp,user,nice,system,idle,iowait,irq,softirq,steal\n' >"${cpu_output}"
  printf 'timestamp,interface,rx_bytes,rx_packets,rx_errors,tx_bytes,tx_packets,tx_errors\n' >"${network_output}"
  printf 'timestamp,device,port_xmit_data,port_rcv_data,port_xmit_packets,port_rcv_packets\n' >"${infiniband_output}"
  while true; do
    timestamp="$(date -u +%FT%T.%NZ)"
    read -r _ user nice system idle iowait irq softirq steal _ </proc/stat
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "${timestamp}" "${user}" "${nice}" "${system}" "${idle}" "${iowait}" "${irq}" "${softirq}" "${steal}" >>"${cpu_output}"
    while IFS=': ' read -r interface rx_bytes rx_packets rx_errors _ _ _ _ _ tx_bytes tx_packets tx_errors _; do
      [[ -n "${interface}" && "${rx_bytes}" =~ ^[0-9]+$ ]] || continue
      printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "${timestamp}" "${interface}" "${rx_bytes}" "${rx_packets}" "${rx_errors}" "${tx_bytes}" "${tx_packets}" "${tx_errors}" >>"${network_output}"
    done </proc/net/dev
    for counters in /sys/class/infiniband/*/ports/1/counters; do
      [[ -d "${counters}" ]] || continue
      [[ -r "${counters}/port_xmit_data" && -r "${counters}/port_rcv_data" && -r "${counters}/port_xmit_packets" && -r "${counters}/port_rcv_packets" ]] || continue
      device="${counters#/sys/class/infiniband/}"
      device="${device%%/*}"
      printf '%s,%s,%s,%s,%s,%s\n' "${timestamp}" "${device}" "$(<"${counters}/port_xmit_data")" "$(<"${counters}/port_rcv_data")" "$(<"${counters}/port_xmit_packets")" "$(<"${counters}/port_rcv_packets")" >>"${infiniband_output}"
    done
    sleep 1
  done
}

# Libraries own cache validity; keep their files across jobs without a custom key.
restore_worker_cache() {
  local archive="$1" directory="$2"
  install -d -m 0700 "${directory}"
  if [[ -f "${archive}" ]]; then
    tar --no-same-owner --no-same-permissions -xf "${archive}" -C "${directory}"
    printf 'jit_cache_restored=%s time=%s\n' "${archive}" "$(date -u +%FT%TZ)"
  fi
}

save_worker_cache() {
  local directory="$1" archive="$2" pending
  install -d -m 0700 "$(dirname "${archive}")" || return 1
  pending="$(mktemp "${archive}.XXXXXXXXXX")" || return 1
  if tar -cf "${pending}" -C "${directory}" . && mv -f -- "${pending}" "${archive}"; then
    printf 'jit_cache_saved=%s time=%s\n' "${archive}" "$(date -u +%FT%TZ)"
  else
    rm -f -- "${pending}"
    return 1
  fi
}

run_node() {
  node_rank="${QWEN_NODE_RANK:-${SLURM_PROCID:-${PMI_RANK:-}}}"
  if [[ -z "${node_rank}" && -s "${OUTPUT_DIR}/source/node-addresses.tsv" ]]; then node_rank="$(awk -v host="$(hostname -s)" '$2 == host {print $1}' "${OUTPUT_DIR}/source/node-addresses.tsv")"; fi
  [[ "${node_rank}" =~ ^[0-9]+$ && "${node_rank}" -lt "${NODE_COUNT}" ]]
  hostname_short="$(hostname -s)"
  node_ip="$(getent ahostsv4 "${hostname_short}" | awk '$2 == "STREAM" { print $1; exit }')"
  [[ "${node_ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]
  node_tmp="$(mktemp -d "${TMPDIR:-/tmp}/qwen-submission-node-${node_rank}.XXXXXXXXXX")"
  pids=()
  cache_workers=()
  cache_root="$(dirname "${OUTPUT_DIR}")/jit-cache/${QWEN_PLATFORM}"
  sampler_pid=
  system_sampler_pid=
  cleanup_node() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "${sampler_pid}" ]]; then kill "${sampler_pid}" 2>/dev/null || true; wait "${sampler_pid}" 2>/dev/null || true; fi
    if [[ -n "${system_sampler_pid}" ]]; then kill "${system_sampler_pid}" 2>/dev/null || true; wait "${system_sampler_pid}" 2>/dev/null || true; fi
    for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
    for cache_worker in "${cache_workers[@]}"; do
      if ! save_worker_cache "${node_tmp}/${cache_worker}/home/.cache" "${cache_root}/${cache_worker}.tar"; then printf 'jit_cache_save_failed worker=%s\n' "${cache_worker}" >&2; fi
    done
    rm -rf -- "${node_tmp:?}"
    exit "${status}"
  }
  trap cleanup_node EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  if [[ "${QWEN_PLATFORM}" == aspire2ap && "$(jq -r '.model_storage // "shared"' "${OUTPUT_DIR}/source/aspire-serving.json")" == node_local ]]; then
    bash "${OUTPUT_DIR}/scripts/stage-qwen-node-model.sh" "${MODEL_PATH}" "${node_tmp}/model" "${OUTPUT_DIR}/logs/node-${node_rank}-model-copy.log"
    MODEL_PATH="${node_tmp}/model"
  fi

  hca_devices=()
  for state_file in /sys/class/infiniband/*/ports/1/state; do if grep -q ACTIVE "${state_file}"; then device="${state_file#/sys/class/infiniband/}"; hca_devices+=("${device%%/*}:1"); fi; done
  [[ "${#hca_devices[@]}" -gt 0 ]]
  ucx_devices="$(IFS=,; printf '%s' "${hca_devices[*]}")"
  printf 'hostname=%s\nnode_ip=%s\nnode_rank=%s\nucx_net_devices=%s\n' "${hostname_short}" "${node_ip}" "${node_rank}" "${ucx_devices}" >"${OUTPUT_DIR}/logs/node-${node_rank}-environment.log"
  nvidia-smi -L >>"${OUTPUT_DIR}/logs/node-${node_rank}-environment.log"
  if [[ "${PROFILE}" == true ]]; then
    install -d -m 0755 "${OUTPUT_DIR}/profile"
    nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm --format=csv -l 1 >"${OUTPUT_DIR}/profile/gpu-node-${node_rank}.csv" & sampler_pid=$!
    sample_system "${node_rank}" "${OUTPUT_DIR}/profile" & system_sampler_pid=$!
  fi

  start_worker() {
    role="$1"
    worker="$2"
    tp="$3"
    port="$4"
    bootstrap="$5"
    gpus="$6"
    ep="${7:-}"
    worker_tmp="${node_tmp}/${role}-${worker}"
    install -d -m 0700 "${worker_tmp}/cache" "${worker_tmp}/data" "${worker_tmp}/home" "${worker_tmp}/runtime" "${worker_tmp}/temp"
    restore_worker_cache "${cache_root}/${role}-${worker}.tar" "${worker_tmp}/home/.cache"
    cache_workers+=("${role}-${worker}")
    local max_total_tokens=131072
    # Pair decode admission capacity with the c64 workload in this trial.
    if [[ "${QWEN_PLATFORM}" == aspire2ap && "${role}" == decode ]]; then max_total_tokens=196608; fi
    args=(/opt/sglang/venv/bin/python -m sglang.launch_server --model-path /model --served-model-name qwen --trust-remote-code --context-length 8192 --tp-size "${tp}" --mem-fraction-static 0.90 --max-total-tokens "${max_total_tokens}" --disaggregation-mode "${role}" --disaggregation-transfer-backend nixl --host 0.0.0.0 --port "${port}")
    if [[ "${QWEN_PLATFORM}" == aspire2ap ]]; then args+=(--nccl-port "$((port + 1000))"); fi
    if [[ "${QWEN_PLATFORM}" == aspire2ap ]]; then args+=(--kv-cache-dtype "$(jq -r '.kv_cache_dtype // "auto"' "${OUTPUT_DIR}/source/aspire-serving.json")"); fi
    if [[ "${QWEN_PLATFORM}" == aspire2ap && "${role}" == prefill ]]; then args+=(--moe-runner-backend cutlass); fi
    if [[ "${role}" == prefill ]]; then args+=(--disaggregation-bootstrap-port "${bootstrap}"); if [[ "${QWEN_PLATFORM}" == aspire2ap ]]; then args+=(--ep-size "${ep:-1}"); fi; else args+=(--ep-size "${ep}" --max-running-requests 256); fi
    draft_mount=()
    if [[ "${QWEN_PLATFORM}" == aspire2ap && "$(jq -r '.speculation.algorithm' "${OUTPUT_DIR}/source/aspire-serving.json")" == EAGLE3 ]]; then
      args+=(--speculative-algorithm EAGLE3 --speculative-draft-model-path /draft --speculative-num-steps "$(jq -r '.speculation.steps' "${OUTPUT_DIR}/source/aspire-serving.json")" --speculative-eagle-topk 1 --speculative-num-draft-tokens "$(jq -r '.speculation.draft_tokens' "${OUTPUT_DIR}/source/aspire-serving.json")")
      draft_mount=(-m "${OUTPUT_DIR}/draft:/draft")
    fi
    if [[ "${QWEN_PLATFORM}" == aspire2ap && "${role}" == decode ]]; then
      mapfile -t decode_graph_bs < <(jq -r '.decode_cuda_graph_bs // [] | .[]' "${OUTPUT_DIR}/source/aspire-serving.json")
      if [[ "${#decode_graph_bs[@]}" -gt 0 ]]; then args+=(--cuda-graph-bs-decode "${decode_graph_bs[@]}"); fi
    fi
    # Adopted prefill-only NCCL symmetric memory; BF16/FP8 precision stays unchanged.
    if [[ "${QWEN_PLATFORM}" == aspire2ap && "${role}" == prefill ]]; then args+=(--enable-symm-mem); fi
    prefill_env=()
    # Adopted prefill bit-copy shuffle and integer routing; D and wheel GEMMs unchanged.
    if [[ "${QWEN_PLATFORM}" == aspire2ap && "${role}" == prefill ]]; then prefill_env=(-e SGLANG_PREFILL_FUSED_FP8_SHUFFLE=1 -e SGLANG_PREFILL_TILED_ROUTING=1); fi
    worker_command=(enroot start -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_DISABLE_REQUIRE=1 -e MELLANOX_VISIBLE_DEVICES=all -e MELLANOX_MOUNT_DRIVER=yes -e "CUDA_VISIBLE_DEVICES=${gpus}" -e "UCX_NET_DEVICES=${ucx_devices}" -e "SGLANG_HOST_IP=${node_ip}" -e HOME=/runtime-home -e XDG_CACHE_HOME=/runtime-home/.cache -e CUDA_CACHE_PATH=/runtime-home/.cache/cuda -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e FLASHINFER_WORKSPACE_BASE=/runtime-home/.cache/flashinfer -e SGLANG_DG_CACHE_DIR=/runtime-home/.cache/deep-gemm -e TORCHINDUCTOR_CACHE_DIR=/runtime-home/.cache/torchinductor -e TRITON_CACHE_DIR=/runtime-home/.cache/triton -m "${MODEL_PATH}:/model" -m "${worker_tmp}/home:/runtime-home" "${draft_mount[@]}" "${prefill_env[@]}" "${CONTAINER_IMAGE}" "${args[@]}")
    if [[ "${PROFILE}" == true ]]; then worker_command=(env "QWEN_PROFILER_INVOCATION_PATH=${OUTPUT_DIR}/profile/profiler-invocation-node-${node_rank}-${role}-${worker}.argv" "${OUTPUT_DIR}/scripts/exec-profiler.sh" "${worker_command[@]}"); fi
    printf '%q ' "${worker_command[@]}" >"${OUTPUT_DIR}/logs/${role}-${worker}-command.txt"
    printf '\n' >>"${OUTPUT_DIR}/logs/${role}-${worker}-command.txt"
    (export ENROOT_CACHE_PATH="${worker_tmp}/cache" ENROOT_DATA_PATH="${worker_tmp}/data" ENROOT_RUNTIME_PATH="${worker_tmp}/runtime" ENROOT_TEMP_PATH="${worker_tmp}/temp"; exec "${worker_command[@]}") >"${OUTPUT_DIR}/logs/${role}-${worker}.log" 2>&1 &
    pids+=("$!")
  }

  case "${QWEN_PLATFORM}" in
    firmus)
      [[ "${NODE_COUNT}" -eq 2 && "${GPUS_PER_NODE}" -eq 8 ]]
      if [[ "${node_rank}" -eq 0 ]]; then start_worker prefill 0 2 30100 9000 0,1; start_worker prefill 1 2 30101 9001 2,3; start_worker prefill 2 2 30102 9002 4,5; start_worker prefill 3 2 30103 9003 6,7; else start_worker decode 0 4 30200 unused 0,1,2,3 4; start_worker decode 1 4 30201 unused 4,5,6,7 4; fi
      ;;
    aspire2ap)
      [[ "${NODE_COUNT}" -eq 2 && "${GPUS_PER_NODE}" -eq 8 ]]
      while IFS=$'\t' read -r role worker tp ep port bootstrap gpus; do start_worker "${role}" "${worker}" "${tp}" "${port}" "${bootstrap}" "${gpus}" "${ep}"; done < <(jq -r --argjson node "${node_rank}" '.workers[] | select(.node==$node) | [.role,.index,.tp,.ep,.port,.bootstrap,.gpus] | @tsv' "${OUTPUT_DIR}/source/aspire-serving.json")
      ;;
    genkai)
      [[ "${GPUS_PER_NODE}" -eq 4 && ( "${NODE_COUNT}" -eq 2 || "${NODE_COUNT}" -eq 4 ) ]]
      prefill_nodes="$((NODE_COUNT / 2))"
      if [[ "${node_rank}" -lt "${prefill_nodes}" ]]; then start_worker prefill "${node_rank}" 4 "$((30100 + node_rank))" "$((9000 + node_rank))" 0,1,2,3; else decode_index="$((node_rank - prefill_nodes))"; start_worker decode "${decode_index}" 4 "$((30200 + decode_index))" unused 0,1,2,3 4; fi
      ;;
    *) printf 'unsupported Qwen platform: %s\n' "${QWEN_PLATFORM}" >&2; exit 64 ;;
  esac
  while true; do
    for pid in "${pids[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        if [[ "${QWEN_PLATFORM}" == aspire2ap ]]; then status=0; wait "${pid}" || status=$?; printf 'worker_exited node=%s pid=%s status=%s\n' "${node_rank}" "${pid}" "${status}" >"${OUTPUT_DIR}/logs/worker-failed-${node_rank}"; exit 1; else wait "${pid}"; fi
      fi
    done
    sleep 2
  done
}

run_router() {
  router_tmp="$(mktemp -d "${TMPDIR:-/tmp}/qwen-submission-router.XXXXXXXXXX")"
  child_pid=
  cleanup_router() { status=$?; trap - EXIT INT TERM; if [[ -n "${child_pid}" ]]; then kill "${child_pid}" 2>/dev/null || true; wait "${child_pid}" 2>/dev/null || true; fi; rm -rf -- "${router_tmp:?}"; exit "${status}"; }
  trap cleanup_router EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  export ENROOT_CACHE_PATH="${router_tmp}/cache" ENROOT_DATA_PATH="${router_tmp}/data" ENROOT_RUNTIME_PATH="${router_tmp}/runtime" ENROOT_TEMP_PATH="${router_tmp}/temp"
  install -d -m 0700 "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}"
  router_args=(/opt/sglang/venv/bin/python -m sglang_router.launch_router --pd-disaggregation)
  while IFS=$'\t' read -r role host port bootstrap; do if [[ "${role}" == prefill ]]; then router_args+=(--prefill "http://${host}:${port}" "${bootstrap}"); else router_args+=(--decode "http://${host}:${port}"); fi; done <"${OUTPUT_DIR}/source/worker-endpoints.tsv"
  router_args+=(--host 0.0.0.0 --port 8000)
  if [[ "${QWEN_PLATFORM}" == aspire2ap ]]; then router_args+=(--prefill-policy "$(jq -r ' .prefill_routing_policy' "${OUTPUT_DIR}/source/aspire-serving.json")"); fi
  enroot start -e NVIDIA_VISIBLE_DEVICES=none -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -m "${MODEL_PATH}:/model" "${CONTAINER_IMAGE}" "${router_args[@]}" &
  child_pid=$!
  wait "${child_pid}"
  child_pid=
}

run_aiperf() {
  aiperf_tmp="$(mktemp -d "${TMPDIR:-/tmp}/qwen-submission-aiperf.XXXXXXXXXX")"
  child_pid=
  cleanup_aiperf() { status=$?; trap - EXIT INT TERM; if [[ -n "${child_pid}" ]]; then kill "${child_pid}" 2>/dev/null || true; wait "${child_pid}" 2>/dev/null || true; fi; rm -rf -- "${aiperf_tmp:?}"; exit "${status}"; }
  trap cleanup_aiperf EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  export ENROOT_CACHE_PATH="${aiperf_tmp}/cache" ENROOT_DATA_PATH="${aiperf_tmp}/data" ENROOT_RUNTIME_PATH="${aiperf_tmp}/runtime" ENROOT_TEMP_PATH="${aiperf_tmp}/temp"
  install -d -m 0700 "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}"
  command=(/opt/sglang/venv/bin/aiperf profile --url http://127.0.0.1:8000 --endpoint-type chat --endpoint /v1/chat/completions --model qwen --tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507 --streaming -H 'Accept: text/event-stream' --concurrency "${CONCURRENCY}" --workers-max "${CONCURRENCY}" --api-key NOT_USED --warmup-duration 60 --benchmark-duration 120 --benchmark-grace-period 30 --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 --output-tokens-mean 200 --output-tokens-stddev 0 --extra-inputs min_tokens:200 --extra-inputs max_tokens:200 --extra-inputs ignore_eos:true --extra-inputs temperature:0.0 --extra-inputs repetition_penalty:1.0 --num-dataset-entries 12800 --random-seed 2026 --cache-bust first_turn_prefix --goodput "time_to_first_token:3000 inter_token_latency:20" --ui simple)
  printf '%q ' "${command[@]}" >"${OUTPUT_DIR}/results/aiperf-command.txt"
  printf '\n' >>"${OUTPUT_DIR}/results/aiperf-command.txt"
  enroot start -e NVIDIA_VISIBLE_DEVICES=none -e HOME=/tmp -e HF_HOME=/hf-cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -m "${OUTPUT_DIR}/results:/results" -m "${OUTPUT_DIR}/tokenizer-cache:/hf-cache" "${CONTAINER_IMAGE}" /bin/bash -ec 'cd /results && printf "aiperf_version=%s\nmode=official\nconcurrency=%s\ngateway=http://127.0.0.1:8000/v1/chat/completions\ntokenizer_revision=%s\n" "$(/opt/sglang/venv/bin/python -c "from importlib.metadata import version; print(version(\"aiperf\"))")" "$1" "$(cat /hf-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/refs/main)" >aiperf-provenance.txt; shift; exec "$@"' _ "${CONCURRENCY}" "${command[@]}" &
  child_pid=$!
  wait "${child_pid}"
  child_pid=
  export_file="$(find "${OUTPUT_DIR}/results/artifacts" -mindepth 2 -maxdepth 2 -type f -name profile_export_aiperf.json -print -quit)"
  [[ -n "${export_file}" && -s "${export_file}" ]]
  cp "${export_file}" "${OUTPUT_DIR}/results/profile_export_aiperf.json"
  if enroot start -e NVIDIA_VISIBLE_DEVICES=none -e HOME=/tmp -m "${OUTPUT_DIR}/results:/results" "${CONTAINER_IMAGE}" /opt/sglang/venv/bin/python - /results/profile_export_aiperf.json /results/score.json --concurrency "${CONCURRENCY}" --require-eligible --command /results/aiperf-command.txt --provenance /results/aiperf-provenance.txt <<'PY'
#!/usr/bin/env python3
"""Validate AIPerf's official Qwen metrics and emit the competition score."""

from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def metric(metrics: dict[str, Any], name: str, *, nullable: bool = False) -> float:
    value = metrics.get(name)
    if value is None and nullable:
        return 0.0
    if not isinstance(value, dict) or not isinstance(value.get("avg"), (int, float)):
        raise TypeError(f"missing numeric {name}.avg")
    result = float(value["avg"])
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}.avg")
    return result


def metric_statistics(metrics: dict[str, Any], name: str) -> dict[str, float]:
    value = metrics.get(name)
    if not isinstance(value, dict):
        return {}
    statistics = {}
    for statistic in ("min", "max", "count", "sum"):
        candidate = value.get(statistic)
        if isinstance(candidate, (int, float)):
            candidate = float(candidate)
            if not math.isfinite(candidate):
                raise ValueError(f"non-finite {name}.{statistic}")
            statistics[statistic] = candidate
    return statistics


def parse_result(metrics: dict[str, Any], concurrency: int) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    goodput = metric(metrics, "goodput")
    fraction = metric(metrics, "good_request_fraction")
    requests = metric(metrics, "request_count")
    errors = metric(metrics, "error_request_count", nullable=True)
    input_length = metric(metrics, "input_sequence_length")
    input_statistics = metric_statistics(metrics, "input_sequence_length")
    output_length = metric(metrics, "output_sequence_length")
    output_statistics = metric_statistics(metrics, "output_sequence_length")
    osl_mismatch_count = metric(metrics, "osl_mismatch_count")
    if requests <= 0 or goodput < 0 or not 0 <= fraction <= 1 or errors < 0:
        raise ValueError("invalid request or goodput metrics")
    eligible = fraction >= 0.90
    result = {
        "schema_version": "qwen3-pd-score/v1",
        "concurrency": concurrency,
        "goodput_requests_per_second": goodput,
        "good_request_fraction": fraction,
        "eligible": eligible,
        "score_output_tokens_per_second": goodput * 200 if eligible else 0.0,
        "request_count": requests,
        "error_request_count": errors,
        "observed_input_sequence_length_avg": input_length,
        "observed_output_sequence_length_avg": output_length,
        "osl_mismatch_count": osl_mismatch_count,
        "sla": {"time_to_first_token_ms": 3000, "inter_token_latency_ms": 20},
    }
    for statistic, value in input_statistics.items():
        result[f"observed_input_sequence_length_{statistic}"] = value
    for statistic, value in output_statistics.items():
        result[f"observed_output_sequence_length_{statistic}"] = value
    return result


TOKENIZER_REVISION = "ac9c66cc9b46af7306746a9250f23d47083d689e"


def expected_official_command(gateway: str, concurrency: int) -> list[str]:
    url = urlsplit(gateway)
    expected = ["/opt/sglang/venv/bin/aiperf", "profile", "--url", f"http://{url.netloc}"]
    expected += shlex.split('--endpoint-type chat --endpoint /v1/chat/completions --model qwen --tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507 --streaming -H "Accept: text/event-stream"')
    expected += ["--concurrency", str(concurrency), "--workers-max", str(concurrency)]
    expected += shlex.split('--api-key NOT_USED --warmup-duration 60 --benchmark-duration 120 --benchmark-grace-period 30 --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 --output-tokens-mean 200 --output-tokens-stddev 0 --extra-inputs min_tokens:200 --extra-inputs max_tokens:200 --extra-inputs ignore_eos:true --extra-inputs temperature:0.0 --extra-inputs repetition_penalty:1.0 --num-dataset-entries 12800 --random-seed 2026 --cache-bust first_turn_prefix --goodput "time_to_first_token:3000 inter_token_latency:20" --ui simple')
    return expected


def validate_official_workload(command: Path, provenance: Path, concurrency: int) -> list[str]:
    """Require the fixed command and measured runtime identity, without tolerance."""
    identity = dict(line.split("=", 1) for line in provenance.read_text().splitlines())
    gateway = identity.get("gateway", "")
    url = urlsplit(gateway)
    if url.scheme != "http" or not url.hostname or url.path != "/v1/chat/completions" or url.query or url.fragment or url.username or url.password:
        raise ValueError("invalid gateway in AIPerf provenance")
    required = {"aiperf_version": "0.12.0", "mode": "official", "concurrency": str(concurrency), "tokenizer_revision": TOKENIZER_REVISION}
    if any(identity.get(key) != value for key, value in required.items()):
        raise ValueError("AIPerf provenance does not match the pinned official workload")
    expected = expected_official_command(gateway, concurrency)
    if shlex.split(command.read_text()) != expected:
        raise ValueError("AIPerf command does not match the fixed official workload")
    return expected


def validate_official_export(metrics: dict[str, Any], result: dict[str, Any], expected_command: list[str]) -> None:
    if metrics.get("aiperf_version") != "0.12.0":
        raise ValueError("AIPerf export version does not match 0.12.0")
    if metrics.get("was_cancelled") is not False:
        raise ValueError("AIPerf export was cancelled")
    if metrics.get("error_summary") != []:
        raise ValueError("AIPerf export contains request errors")
    duration = metric(metrics, "benchmark_duration")
    good_requests = metric(metrics, "good_request_count")
    if duration < 120 or good_requests < 0 or good_requests > result["request_count"]:
        raise ValueError("AIPerf export has incomplete benchmark accounting")
    denominator = result["request_count"] + result["error_request_count"]
    if denominator <= 0 or not math.isclose(result["good_request_fraction"], good_requests / denominator, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("AIPerf good request accounting is inconsistent")
    run_info = metrics.get("run_info")
    if not isinstance(run_info, dict) or run_info.get("random_seed") != 2026 or not isinstance(run_info.get("cli_command"), str):
        raise ValueError("AIPerf export run identity is incomplete")
    export_command = shlex.split(run_info["cli_command"])
    if export_command and export_command[0] == "aiperf":
        export_command[0] = expected_command[0]
    try:
        api_key_index = export_command.index("--api-key") + 1
        if export_command[api_key_index] == "<redacted>":
            export_command[api_key_index] = "NOT_USED"
    except (ValueError, IndexError):
        raise ValueError("AIPerf export command is incomplete") from None
    if export_command != expected_command:
        raise ValueError("AIPerf export command does not match the fixed official workload")
    config = metrics.get("input_config")
    if not isinstance(config, dict):
        raise TypeError("AIPerf export input configuration is missing")
    expected_phases = [
        {"name": "warmup", "kind": "warmup", "type": "concurrency", "exclude_from_results": True, "duration": 60.0, "concurrency": result["concurrency"]},
        {"name": "profiling", "kind": "profiling", "type": "concurrency", "duration": 120.0, "concurrency": result["concurrency"], "grace_period": 30.0},
    ]
    if config.get("phases") != expected_phases:
        raise ValueError("AIPerf export phases do not match the fixed official workload")


def validate_official_result(metrics: dict[str, Any], result: dict[str, Any], command: Path | None, provenance: Path | None) -> dict[str, Any]:
    if command is None or provenance is None:
        raise ValueError("official validation requires --command and --provenance")
    expected_command = validate_official_workload(command, provenance, result["concurrency"])
    validate_official_export(metrics, result, expected_command)
    if not result["eligible"]:
        raise ValueError("good_request_fraction is below 0.90")
    # AIPerf 0.12.0 subtracts the rounded 14-token cache-bust estimate from the
    # synthetic ISL. The actual first_turn_prefix marker varies from 12 to 16
    # tokens, so individual client observations span 3984 through 3988 tokens.
    input_length = result["observed_input_sequence_length_avg"]
    required_statistics = ("min", "max", "count", "sum")
    input_statistics = {name: result.get(f"observed_input_sequence_length_{name}") for name in required_statistics}
    if any(value is None for value in input_statistics.values()):
        raise ValueError("input sequence length lacks distribution evidence")
    minimum, maximum = input_statistics["min"], input_statistics["max"]
    count, total = input_statistics["count"], input_statistics["sum"]
    valid_distribution = 3984 <= minimum <= maximum <= 3988
    valid_aggregate = count == result["request_count"] and count > 0 and total.is_integer() and math.isclose(input_length, total / count, rel_tol=0.0, abs_tol=1e-12)
    if not valid_distribution or not valid_aggregate:
        raise ValueError("input sequence length is inconsistent with pinned AIPerf's cache-bust distribution")
    # AIPerf 0.12.0 defines an OSL mismatch as a difference greater than the
    # smaller of 5% and 50 tokens. At the fixed requested OSL of 200, values
    # from 190 through 210 are therefore within its pinned tolerance. Validate
    # the raw distribution independently so an altered mismatch threshold
    # cannot make an out-of-range result pass.
    required_statistics = ("min", "max", "count", "sum")
    output_statistics = {
        name: result.get(f"observed_output_sequence_length_{name}")
        for name in required_statistics
    }
    if any(value is None for value in output_statistics.values()):
        raise ValueError("output sequence length lacks distribution evidence")
    minimum, maximum = output_statistics["min"], output_statistics["max"]
    count, total = output_statistics["count"], output_statistics["sum"]
    output_length = result["observed_output_sequence_length_avg"]
    valid_distribution = 190 <= minimum <= maximum <= 210
    valid_aggregate = (
        count == result["request_count"]
        and count > 0
        and total.is_integer()
        and math.isclose(output_length, total / count, rel_tol=0.0, abs_tol=1e-12)
    )
    if not valid_distribution or not valid_aggregate:
        raise ValueError("output sequence length is inconsistent with pinned AIPerf tolerance")
    if result["osl_mismatch_count"] != 0:
        raise ValueError("AIPerf reported output sequence length mismatches")
    return {"valid": True, "configured_input_sequence_length": 4000, "configured_output_sequence_length": 200, "client_input_token_adjustment": 4000 - input_length, "aiperf_version": "0.12.0", "tokenizer_revision": TOKENIZER_REVISION}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--require-eligible", action="store_true")
    parser.add_argument("--command", type=Path)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    metrics = json.loads(args.export.read_text())
    result = parse_result(metrics, args.concurrency)
    validation_error = None
    if args.require_eligible:
        try:
            result["official_validation"] = validate_official_result(metrics, result, args.command, args.provenance)
        except (TypeError, ValueError, OSError) as error:
            validation_error = str(error)
            result["official_validation"] = {"valid": False, "error": validation_error}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if validation_error is not None:
        raise SystemExit(f"invalid official result: {validation_error}")


if __name__ == "__main__":
    main()
PY
  then parser_status=0; else parser_status=$?; fi
  if [[ "${QWEN_PLATFORM}" == aspire2ap ]]; then
    while IFS=$'\t' read -r role host port _; do curl --fail --silent --max-time 10 "http://${host}:${port}/get_server_info" >"${OUTPUT_DIR}/results/server-info-${role}-${port}.json"; done <"${OUTPUT_DIR}/source/worker-endpoints.tsv"
    enroot start -e NVIDIA_VISIBLE_DEVICES=none -e HOME=/tmp -m "${OUTPUT_DIR}:/calibration" "${CONTAINER_IMAGE}" /usr/local/bin/uv run --offline --no-project --python /opt/sglang/venv/bin/python python /calibration/scripts/greedy_probe.py /calibration/results
  fi
  return "${parser_status}"
}

if [[ "${mode}" == node ]]; then run_node; exit; fi
if [[ "${mode}" == router ]]; then run_router; exit; fi
if [[ "${mode}" == aiperf ]]; then run_aiperf; exit; fi
[[ "${mode}" == orchestrate ]]
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]
[[ "${PROFILE}" == true || "${PROFILE}" == false ]]
[[ "${NODE_COUNT}" =~ ^[1-9][0-9]*$ && "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]
case "${QWEN_PLATFORM}" in firmus) [[ "${NODE_COUNT}" -eq 2 && "${GPUS_PER_NODE}" -eq 8 ]] ;; aspire2ap) [[ "${NODE_COUNT}" -eq 2 && "${GPUS_PER_NODE}" -eq 8 ]] ;; genkai) [[ "${GPUS_PER_NODE}" -eq 4 && ( "${NODE_COUNT}" -eq 2 || "${NODE_COUNT}" -eq 4 ) ]] ;; *) exit 64 ;; esac
[[ -s "${CONTAINER_IMAGE}" ]]
[[ -s "${MODEL_PATH}/config.json" && -s "${MODEL_PATH}/model.safetensors.index.json" ]]
[[ "$(head -n 1 "${MODEL_PATH}/.cache/huggingface/download/config.json.metadata")" == e156cb4efae43fbee1a1ab073f946a1377e6b969 ]]
mapfile -t model_shards < <(jq -r '.weight_map[]' "${MODEL_PATH}/model.safetensors.index.json" | sort -u)
mapfile -t expected_model_shards < <(printf 'model-%05d-of-00024.safetensors\n' {1..24})
[[ "${model_shards[*]}" == "${expected_model_shards[*]}" ]]
for model_shard in "${expected_model_shards[@]}"; do [[ -s "${MODEL_PATH}/${model_shard}" ]]; [[ "$(head -n 1 "${MODEL_PATH}/.cache/huggingface/download/${model_shard}.metadata")" == e156cb4efae43fbee1a1ab073f946a1377e6b969 ]]; done
jq -e '.model_type == "qwen3_moe" and .architectures == ["Qwen3MoeForCausalLM"] and .quantization_config.quant_method == "fp8" and .quantization_config.fmt == "e4m3"' "${MODEL_PATH}/config.json" >/dev/null
tokenizer_files=(config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json)
tokenizer_hashes=(ad59a339665ee1c68cbff06ac72c9e59f5183ebfaf3698c98737e003a870e9e2 835fffe355c9438e7a25be099b3fccaa98350b83451f9fd2d99512e74f1ade48 599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3 aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3 ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910)
for index in "${!tokenizer_files[@]}"; do tokenizer_file="${tokenizer_files[${index}]}"; [[ "$(sha256sum "${TOKENIZER_PATH}/${tokenizer_file}" | awk '{print $1}')" == "${tokenizer_hashes[${index}]}" ]]; done
install -d -m 0755 "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/results" "${OUTPUT_DIR}/tokenizer-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/refs" "${OUTPUT_DIR}/tokenizer-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e"
shared_script_path="${OUTPUT_DIR}/submit.sbatch"
if [[ "${script_path}" != "${shared_script_path}" ]]; then install -m 0755 "${script_path}" "${shared_script_path}"; script_path="${shared_script_path}"; fi
tokenizer_snapshot="${OUTPUT_DIR}/tokenizer-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e"
if [[ "$(realpath "${TOKENIZER_PATH}")" != "$(realpath "${tokenizer_snapshot}")" ]]; then for tokenizer_file in "${tokenizer_files[@]}"; do cp -L "${TOKENIZER_PATH}/${tokenizer_file}" "${tokenizer_snapshot}/${tokenizer_file}"; done; fi
printf '%s' ac9c66cc9b46af7306746a9250f23d47083d689e >"${OUTPUT_DIR}/tokenizer-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/refs/main"
case "${QWEN_PLATFORM}" in
  firmus) mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}") ;;
  aspire2ap) mapfile -t nodes < <(awk '!seen[$0]++ {sub(/\..*/, "", $0); print}' "${PBS_NODEFILE:?PBS node file is required}") ;;
  genkai)
    node_probe="${OUTPUT_DIR}/source/node-probe.tsv"
    mpirun -np "${NODE_COUNT}" -ppn 1 -genv PATH "${PATH}" /bin/bash -c 'printf "%s\t%s\n" "${PMI_RANK}" "$(hostname -s)"' >"${node_probe}"
    mapfile -t nodes < <(sort -n -k1,1 "${node_probe}" | cut -f2)
    ;;
esac
[[ "${#nodes[@]}" -eq "${NODE_COUNT}" && "$(printf '%s\n' "${nodes[@]}" | sort -u | wc -l)" -eq "${NODE_COUNT}" ]]
: >"${OUTPUT_DIR}/source/node-addresses.tsv"
node_ips=()
for rank in "${!nodes[@]}"; do node_ip="$(getent ahostsv4 "${nodes[${rank}]}" | awk '$2 == "STREAM" { print $1; exit }')"; [[ "${node_ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; node_ips+=("${node_ip}"); printf '%s\t%s\t%s\n' "${rank}" "${nodes[${rank}]}" "${node_ip}" >>"${OUTPUT_DIR}/source/node-addresses.tsv"; done
: >"${OUTPUT_DIR}/source/worker-endpoints.tsv"
case "${QWEN_PLATFORM}" in
  firmus)
    for worker in 0 1 2 3; do printf 'prefill\t%s\t%s\t%s\n' "${node_ips[0]}" "$((30100 + worker))" "$((9000 + worker))" >>"${OUTPUT_DIR}/source/worker-endpoints.tsv"; done
    for worker in 0 1; do printf 'decode\t%s\t%s\tunused\n' "${node_ips[1]}" "$((30200 + worker))" >>"${OUTPUT_DIR}/source/worker-endpoints.tsv"; done
    ;;
  aspire2ap)
    while IFS=$'\t' read -r role rank port bootstrap; do printf '%s\t%s\t%s\t%s\n' "${role}" "${node_ips[${rank}]}" "${port}" "${bootstrap}" >>"${OUTPUT_DIR}/source/worker-endpoints.tsv"; done < <(jq -r '.workers[] | [.role,.node,.port,.bootstrap] | @tsv' "${OUTPUT_DIR}/source/aspire-serving.json")
    ;;
  genkai)
    prefill_nodes="$((NODE_COUNT / 2))"
    for ((rank = 0; rank < prefill_nodes; rank++)); do printf 'prefill\t%s\t%s\t%s\n' "${node_ips[${rank}]}" "$((30100 + rank))" "$((9000 + rank))" >>"${OUTPUT_DIR}/source/worker-endpoints.tsv"; done
    for ((rank = prefill_nodes; rank < NODE_COUNT; rank++)); do printf 'decode\t%s\t%s\tunused\n' "${node_ips[${rank}]}" "$((30200 + rank - prefill_nodes))" >>"${OUTPUT_DIR}/source/worker-endpoints.tsv"; done
    ;;
esac
server_pid=
router_pid=
cleanup_all() { status=$?; trap - EXIT INT TERM; if [[ -n "${router_pid}" ]]; then kill "${router_pid}" 2>/dev/null || true; wait "${router_pid}" 2>/dev/null || true; fi; if [[ -n "${server_pid}" ]]; then kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true; fi; exit "${status}"; }
trap cleanup_all EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
case "${QWEN_PLATFORM}" in
  firmus) srun --overlap --nodes="${NODE_COUNT}" --ntasks="${NODE_COUNT}" --ntasks-per-node=1 --distribution=block:block "${script_path}" node >"${OUTPUT_DIR}/logs/server-step.log" 2>&1 & ;;
  aspire2ap) /opt/pbs/bin/pbsdsh -- env "QWEN_PLATFORM=${QWEN_PLATFORM}" "OUTPUT_DIR=${OUTPUT_DIR}" "MODEL_PATH=${MODEL_PATH}" "CONTAINER_IMAGE=${CONTAINER_IMAGE}" "PROFILE=${PROFILE}" "NODE_COUNT=${NODE_COUNT}" "GPUS_PER_NODE=${GPUS_PER_NODE}" "ENROOT_LIBRARY_PATH=${ENROOT_LIBRARY_PATH}" "ENROOT_SYSCONF_PATH=${ENROOT_SYSCONF_PATH}" "ENROOT_MOUNT_HOME=${ENROOT_MOUNT_HOME}" "${script_path}" node >"${OUTPUT_DIR}/logs/server-step.log" 2>&1 & ;;
  genkai) mpirun -np "${NODE_COUNT}" -ppn 1 -genv PATH "${PATH}" "${script_path}" node >"${OUTPUT_DIR}/logs/server-step.log" 2>&1 & ;;
esac
server_pid=$!
worker_urls=()
while IFS=$'\t' read -r _ host port _; do worker_urls+=("http://${host}:${port}/health"); done <"${OUTPUT_DIR}/source/worker-endpoints.tsv"
if [[ "${QWEN_PLATFORM}" == aspire2ap ]]; then
  deadline="$((SECONDS + 1500))"
  while true; do kill -0 "${server_pid}"; if compgen -G "${OUTPUT_DIR}/logs/worker-failed-*" >/dev/null; then cat "${OUTPUT_DIR}"/logs/worker-failed-* >&2; exit 1; fi; ready=0; for url in "${worker_urls[@]}"; do if curl --fail --silent --max-time 2 "${url}" >/dev/null; then ready="$((ready + 1))"; fi; done; if [[ "${ready}" -eq "${#worker_urls[@]}" ]]; then break; fi; if (( SECONDS >= deadline )); then printf 'worker_readiness_timeout ready=%s expected=%s utc=%s\n' "${ready}" "${#worker_urls[@]}" "$(date -u +%FT%TZ)" >&2; exit 1; fi; sleep 5; done
else
  deadline="$((SECONDS + 900))"
  while true; do kill -0 "${server_pid}"; ready=0; for url in "${worker_urls[@]}"; do if curl --fail --silent --max-time 2 "${url}" >/dev/null; then ready="$((ready + 1))"; fi; done; if [[ "${ready}" -eq "${#worker_urls[@]}" ]]; then break; fi; (( SECONDS < deadline )); sleep 5; done
fi
"${script_path}" router >"${OUTPUT_DIR}/logs/router.log" 2>&1 &
router_pid=$!
deadline="$((SECONDS + 300))"
models_url="http://127.0.0.1:8000/v1/models"
models_pending="${OUTPUT_DIR}/results/models.pending.json"
while true; do
  kill -0 "${router_pid}"
  if curl --fail --silent --max-time 2 "${models_url}" >"${models_pending}" && jq -e '.data | map(.id) | index("qwen") != null' "${models_pending}" >/dev/null; then mv "${models_pending}" "${OUTPUT_DIR}/results/models.json"; break; fi
  (( SECONDS < deadline ))
  sleep 2
done
"${script_path}" aiperf >"${OUTPUT_DIR}/logs/aiperf.log" 2>&1
if [[ "${PROFILE}" == true ]]; then [[ "$(find "${OUTPUT_DIR}/profile" -maxdepth 1 -type f -name 'profiler-invocation-*.argv' | wc -l)" -eq "$(wc -l <"${OUTPUT_DIR}/source/worker-endpoints.tsv")" ]]; fi
touch "${OUTPUT_DIR}/run-success"
