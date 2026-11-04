#!/usr/bin/env bash
set -euo pipefail
root=${1:?runtime directory}
worker=${2:?worker}
run_id=${3:?run id}
image=${4:?image sha256}
# Exactly the serving Prefill settings. The diagnostic has no environment flag.
set -a
source "${root}/source/prefill-kernels.env"
set +a
export DG_SM90_PREFILL_LOCAL_EP1=1 DG_SM90_PREFILL_2WG=1
exec /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${root}/scripts/probe-firmus-megamoe-rescale.py" --output "${root}/results/megamoe-rescale-probe/worker-${worker}" --worker "${worker}" --routing "${root}/source/megamoe-local-ep1-routing.npz" --run-id "${run_id}" --image-sha256 "${image}"
