#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${BUILD_DIR:?temporary build directory is required}"
: "${CONTAINER_ARCHIVE_DIR:?SquashFS output directory is required}"
: "${GITHUB_RUN_ATTEMPT:?unique build attempt is required}"
: "${GITHUB_RUN_ID:?unique build ID is required}"
: "${GITHUB_SHA:?Qwen repository commit is required}"
# Run submission/build/sglang-mirror.sh first to populate GIT_MIRROR_DIR from public sources.
: "${GIT_MIRROR_DIR:?directory containing 5_SGLang_Src.git is required}"

export GITHUB_WORKSPACE="${repository}"
exec "${repository}/scripts/build-sglang-container.sh"
