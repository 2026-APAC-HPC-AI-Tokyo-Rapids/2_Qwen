#!/usr/bin/env bash
# Profiling hook. Keep this transparent until a profiler is selected.
set -euo pipefail

: "${1:?a command is required}"
"$@"
