#!/usr/bin/env bash
# Hardware counters for the existing FC1/FC2 math probe, one GPU, no model.
set -euo pipefail
[[ -z "${CANDIDATE_PATCH:-}" ]]
bash "${GITHUB_WORKSPACE}/kernels/probes/prepare-megamoe-math.sh"
cd "${RUNTIME_DIR}/source"
mv probe math-probe
cat > probe <<'PROBE'
#!/usr/bin/env bash
set -euo pipefail
# Normal timings come from the unprofiled probe. These runs only diagnose stalls.
ncu --filter-mode per-kernel --kernel-name 'regex:baseline_fc[12]' --launch-skip 3 --launch-count 1 --cache-control none --clock-control none --section SpeedOfLight --section ComputeWorkloadAnalysis --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis_Tables --section LaunchStats --page raw --csv --print-metric-name name "${RUNTIME_DIR}/source/math-probe"
PROBE
chmod +x probe
