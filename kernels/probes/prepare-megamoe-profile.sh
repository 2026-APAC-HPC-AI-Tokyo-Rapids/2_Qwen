#!/usr/bin/env bash
# Instruction-level FC2 stall samples, one GPU, no model.
set -euo pipefail
bash "${GITHUB_WORKSPACE}/kernels/probes/prepare-megamoe-math.sh"
cd "${RUNTIME_DIR}/source"
mv probe math-probe
cat > probe <<'PROBE'
#!/usr/bin/env bash
set -euo pipefail
# Normal timings come from the unprofiled probe. These runs only diagnose stalls.
ncu --kernel-name regex:baseline_fc2 --launch-skip 3 --launch-count 1 --cache-control none --clock-control none --section SourceCounters --page source --print-source sass --csv "${RUNTIME_DIR}/source/math-probe"
PROBE
chmod +x probe
