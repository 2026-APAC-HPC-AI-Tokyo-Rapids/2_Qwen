#!/usr/bin/env bash
# Sequential per-worker measurements; different workers occupy disjoint GPUs.
set -uo pipefail
root=$1
worker=$2
phase=$3
case "${phase}" in
  screen)
    control=$4
    bash "${root}/scripts/select-firmus-prefill-kernels.sh" "${root}" "${worker}" --names n192-direct-o-varlen-pipe-diag --control "${control}"
    timeout --kill-after=5 175 /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${root}/scripts/select-firmus-prefill-megamoe.py" --unroll16 --output "${root}/results/prefill-unroll16/${worker}"
    ;;
  endurance)
    timeout --kill-after=5 60 /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${root}/scripts/select-firmus-prefill-fa3.py" --endurance "${root}/results/prefill-selection" --output "${root}/results/prefill-fa3-endurance/${worker}"
    timeout --kill-after=5 60 /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${root}/scripts/select-firmus-prefill-megamoe.py" --unroll16 --endurance "${root}/results/prefill-unroll16" --output "${root}/results/prefill-unroll16-endurance/${worker}"
    ;;
  *) printf 'Unknown PP4 bundle phase: %s\n' "${phase}" >&2; exit 2 ;;
esac
