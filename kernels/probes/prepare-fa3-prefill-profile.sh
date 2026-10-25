#!/usr/bin/env bash
# Control-only N224 fused profile, one calibration trial on one GPU.
set -euo pipefail
bash "${GITHUB_WORKSPACE}/kernels/probes/prepare-fa3-prefill.sh" --profile
cd "${RUNTIME_DIR}/source"
cp "${GITHUB_WORKSPACE}/kernels/probes/run-fa3-prefill-profile.sh" probe
chmod +x probe
jq '.kernel="FA3 Prefill M2 N224 fused shape4096" | .calibration=true' provenance.json > provenance.tmp
mv provenance.tmp provenance.json
