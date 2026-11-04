#!/usr/bin/env bash
# Build the Basic image (official SGLang 0.5.23 plus a separate AIPerf environment) as SquashFS on a root Linux
# build host (Buildah chroot, Skopeo, Umoci, SquashFS tools).
#
# Environment:
#   BUILD_DIR              empty temporary build directory
#   CONTAINER_ARCHIVE_DIR  output directory for BUILD_ID.sqsh and BUILD_ID.container-sidecar.json
#   GITHUB_RUN_ID          numeric build ID (BUILD_ID), GITHUB_RUN_ATTEMPT its attempt number
#   BUILD_CACHE_REPOSITORY optional writable OCI registry for Buildah layer caching
#
# basic/build/source/ holds the build inputs as byte-identical copies of the team repository. The Dockerfile and image
# check are those of Qwen aa48ceeca33a, the commit the measured image was built from; the build script differs from
# that commit only in making the registry cache optional.
set -euo pipefail

GITHUB_WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")/source" && pwd)
# The image records the commit its build inputs come from.
GITHUB_SHA=aa48ceeca33afc27c2995713d55d4a4c65d02fc1
export GITHUB_WORKSPACE GITHUB_SHA
# The build script reads the object format and file blob IDs with git, so the input directory is a Git work tree.
git init -q "${GITHUB_WORKSPACE}"
exec "${GITHUB_WORKSPACE}/scripts/build-basic-container.sh"
