#!/usr/bin/env bash
# Build the Bonus serving image as SquashFS on a root Linux build host (Buildah chroot, Skopeo, Umoci, SquashFS tools).
#
# Run submission/build/sglang-mirror.sh first. Environment:
#   BUILD_DIR              empty temporary build directory
#   CONTAINER_ARCHIVE_DIR  output directory for BUILD_ID.sqsh and BUILD_ID.container-sidecar.json
#   GITHUB_RUN_ID          numeric build ID (BUILD_ID), GITHUB_RUN_ATTEMPT its attempt number
#   GITHUB_SHA             the checked-out commit of this repository
#   GIT_MIRROR_DIR         the directory sglang-mirror.sh filled (holds 5_SGLang_Src.git)
#   BUILD_CACHE_REPOSITORY optional writable OCI registry for Buildah layer caching
set -euo pipefail

GITHUB_WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export GITHUB_WORKSPACE
exec "${GITHUB_WORKSPACE}/scripts/build-sglang-container.sh"
