#!/usr/bin/env bash
set -euo pipefail
root=${1:?runtime directory}
worker=${2:?worker}
run_id=${3:?run id}
image=${4:?image sha256}
model=${5:?checkpoint directory}
started=${6:?aggregate start epoch}
set -a
source "${root}/source/prefill-kernels.env"
set +a
cache=$(mktemp -d /tmp/prefill-dense-r44.XXXXXXXX)
trap 'rm -rf "${cache}"' EXIT
stat -f -c 'dense_jit_cache_filesystem=%T' "${cache}"
/opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${root}/scripts/probe-firmus-prefill-dense.py" --output "${root}/results/prefill-dense-r44/worker-${worker}" --worker "${worker}" --model "${model}" --run-id "${run_id}" --image-sha256 "${image}" --started "${started}" --cache "${cache}" --torchrun-started "$(date +%s.%N)"
