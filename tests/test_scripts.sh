#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
bash -n "$root/scripts/run-experiment.sh" "$root/jobs/genkai-qwen.pjm"
grep -F "contains(github.event.head_commit.message, '[run-qwen]')" "$root/.github/workflows/genkai.yml" >/dev/null
grep -F '# pjsub -x' "$root/.github/workflows/genkai.yml" >/dev/null
grep -F 'HF_HUB_OFFLINE=1' "$root/scripts/run-experiment.sh" >/dev/null
echo ok
