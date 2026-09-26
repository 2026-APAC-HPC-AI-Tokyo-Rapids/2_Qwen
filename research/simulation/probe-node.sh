#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?runtime directory required}"
exec >"${RUNTIME_DIR}/logs/bootstrap-$(hostname -s).log" 2>&1
trap 'printf "probe_bootstrap_failed line=%s status=%s\n" "${LINENO}" "$?" >&2' ERR
source "${HOME}/.local/.setenv"
: "${PBS_JOBID:?PBS allocation required}"
: "${RUNTIME_DIR:?runtime directory required}"
mapfile -t nodes <"${RUNTIME_DIR}/logs/hosts.txt"
[[ "${#nodes[@]}" -eq 2 ]]
rank="$(printf '%s\n' "${nodes[@]}" | awk -v h="$(hostname -s)" '$0==h {print NR-1}')"
[[ "${rank}" == 0 || "${rank}" == 1 ]]
node_tmp="$(mktemp -d "${TMPDIR:-/tmp}/qwen-calibration.XXXXXXXXXX")"
trap 'rm -rf -- "${node_tmp:?}"' EXIT
export ENROOT_LIBRARY_PATH="${RUNTIME_DIR}/enroot-library" ENROOT_SYSCONF_PATH="${RUNTIME_DIR}/enroot-system" ENROOT_MOUNT_HOME=no
export ENROOT_CACHE_PATH="${node_tmp}/cache" ENROOT_DATA_PATH="${node_tmp}/data" ENROOT_RUNTIME_PATH="${node_tmp}/runtime" ENROOT_TEMP_PATH="${node_tmp}/temp"
install -d -m 0700 "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}"
nvidia-smi -q -x >"${RUNTIME_DIR}/results/node-${rank}-gpu.xml"
nvidia-smi topo -m >"${RUNTIME_DIR}/results/node-${rank}-topology.txt"
{ hostname; date -u +%FT%TZ; df -Th "${RUNTIME_DIR}" "${node_tmp}"; for port in /sys/class/infiniband/*/ports/*; do for field in state rate link_layer; do if [[ -r "${port}/${field}" ]]; then printf '%s=' "${port}/${field}"; cat "${port}/${field}"; fi; done; done; } >"${RUNTIME_DIR}/results/node-${rank}-host.txt"
master="$(getent ahostsv4 "${nodes[0]}" | awk '$2=="STREAM" {print $1; exit}')"
hca_devices=()
for state_file in /sys/class/infiniband/*/ports/1/state; do if grep -q ACTIVE "${state_file}"; then device="${state_file#/sys/class/infiniband/}"; hca_devices+=("${device%%/*}:1"); fi; done
[[ "${#hca_devices[@]}" -gt 0 ]]
ucx_devices="$(IFS=,; printf '%s' "${hca_devices[*]}")"
printf 'ucx_net_devices=%s\n' "${ucx_devices}" >>"${RUNTIME_DIR}/results/node-${rank}-host.txt"
enroot start -e "UCX_NET_DEVICES=${ucx_devices}" -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_DISABLE_REQUIRE=1 -e MELLANOX_VISIBLE_DEVICES=all -e MELLANOX_MOUNT_DRIVER=yes -e "PBS_JOBID=${PBS_JOBID}" -e HOME=/tmp -m "${RUNTIME_DIR}:/calibration" "${RUNTIME_DIR}/sglang.sqsh" /usr/local/bin/uv run --offline --no-project --python /opt/sglang/venv/bin/python python -m torch.distributed.run --nnodes=2 --nproc-per-node=8 --node-rank="${rank}" --master-addr="${master}" --master-port=29573 /calibration/scripts/probe.py /calibration/results >"${RUNTIME_DIR}/logs/node-${rank}.log" 2>&1
