#!/bin/bash
set -euo pipefail
# Torch library registration and resource queries below do not initialize CUDA.
flash_ops=/opt/sglang/venv/lib/python3.12/site-packages/sgl_kernel/flash_ops.abi3.so
before=$(sha256sum "${flash_ops}")
/opt/sglang/venv/bin/python /opt/apac/fa3-verify/build.py
[[ "$(sha256sum "${flash_ops}")" == "${before}" ]]
