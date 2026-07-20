#!/usr/bin/env bash
#PJM -L rscgrp=a-batch
#PJM -L vnode-core=60
#PJM -L elapse=02:00:00
#PJM -j
#PJM -N sglang_sif_build

set -euo pipefail
: "${ROOTFS_TAR:?ROOTFS_TAR is required}"
: "${ROOTFS_SQSH:?ROOTFS_SQSH is required}"
: "${BUILD_DIR:?BUILD_DIR is required}"
: "${SQFSTAR:?SQFSTAR is required}"

"${SQFSTAR}" -root-uid 0 -root-gid 0 -root-mode 0755 -no-xattrs -processors 60 -comp gzip -Xcompression-level 1 "${ROOTFS_SQSH}" < "${ROOTFS_TAR}"
touch "${BUILD_DIR}/build-success"
