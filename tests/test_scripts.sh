#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
bash -n "$root/scripts/run-experiment.sh" "$root/jobs/genkai-qwen.pjm"
bash -n "$root/jobs/genkai-qwen35-offline-oci.pjm" "$root/jobs/templates/genkai.pjm.in"
grep -F "contains(github.event.head_commit.message, '[run-qwen]')" "$root/.github/workflows/genkai.yml" >/dev/null
grep -F '# pjsub -x' "$root/.github/workflows/genkai.yml" >/dev/null
grep -F 'HF_HUB_OFFLINE=1' "$root/scripts/run-experiment.sh" >/dev/null
job="$(python3 "$root/scripts/render-job.py" "$root/experiments/qwen35-fp8-baseline.env")"
bash -n "$job"
grep -F '#PJM -L gpu=1' "$job" >/dev/null
python3 "$root/scripts/summarize.py" "$root/tests/fixtures" --output "$root/summary.csv"
grep -F '11.0,105.0,21.0,126.0' "$root/summary.csv" >/dev/null
python3 "$root/scripts/compare.py" "$root/summary.csv" --baseline fixtures --candidate fixtures | grep -F 'total_throughput,126,126,0.000' >/dev/null
rm -f "$root/summary.csv"
echo ok
