#!/usr/bin/env bash
set -euo pipefail

export GITHUB_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${GITHUB_WORKSPACE}/scripts/build-vllm-container.sh"
