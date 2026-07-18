#!/usr/bin/env bash
set -euo pipefail

exec /usr/sbin/mksquashfs "$@" -Xcompression-level 1
