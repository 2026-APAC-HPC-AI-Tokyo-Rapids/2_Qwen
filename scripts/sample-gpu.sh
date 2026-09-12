#!/usr/bin/env bash
set -euo pipefail

: "${OUTPUT:?output path is required}"
while true; do
  timestamp="$(date -u +%FT%T.%NZ)"
  nvidia-smi --query-gpu=index,uuid,timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem --format=csv,noheader,nounits | while IFS= read -r row; do printf '%s, %s\n' "${timestamp}" "${row}"; done
  sleep 1
done >>"${OUTPUT}"
