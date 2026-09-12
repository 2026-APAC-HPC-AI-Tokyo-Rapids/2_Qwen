#!/usr/bin/env bash
set -euo pipefail

: "${OUTPUT_DIR:?profile output directory is required}"
: "${NODE_RANK:?node rank is required}"

cpu_output="${OUTPUT_DIR}/cpu-node-${NODE_RANK}.csv"
network_output="${OUTPUT_DIR}/network-node-${NODE_RANK}.csv"
infiniband_output="${OUTPUT_DIR}/infiniband-node-${NODE_RANK}.csv"
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
