#!/usr/bin/env bash
# Build the Bonus serving image as SquashFS on a root Linux build host (Buildah chroot, Skopeo, Umoci, SquashFS tools).
#
# Run build/sglang-mirror.sh first. Environment:
#   BUILD_DIR              empty temporary build directory
#   CONTAINER_ARCHIVE_DIR  output directory for BUILD_ID.sqsh and BUILD_ID.container-sidecar.json
#   GITHUB_RUN_ID          numeric build ID (BUILD_ID), GITHUB_RUN_ATTEMPT its attempt number
#   GIT_MIRROR_DIR         the directory sglang-mirror.sh filled (holds 5_SGLang_Src.git)
#   BUILD_CACHE_REPOSITORY optional writable OCI registry for Buildah layer caching
#
# build/source/ holds the build inputs (containers/, kernels/, patches/, scripts/, configs/) as byte-identical
# copies of the team repository at Qwen 3f9124fd8afb, the commit the measured image was built from.
set -euo pipefail

GITHUB_WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")/source" && pwd)
# The image records the commit its build inputs come from.
GITHUB_SHA=3f9124fd8afb28e34c71288b6dc31e5c490c7f60
export GITHUB_WORKSPACE GITHUB_SHA
# The build script reads the object format and file blob IDs with git, so the input directory is a Git work tree.
git init -q "${GITHUB_WORKSPACE}"
exec "${GITHUB_WORKSPACE}/scripts/build-sglang-container.sh"
