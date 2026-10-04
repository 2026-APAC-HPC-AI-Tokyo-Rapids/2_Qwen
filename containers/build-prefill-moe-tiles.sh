#!/bin/bash
set -euxo pipefail
cutlass_revision=57e3cfb47a2d9e0d46eb6335c3dc411498efa198
install -d /opt/apac/cutlass
curl -fLSs "https://github.com/NVIDIA/cutlass/archive/${cutlass_revision}.tar.gz" | tar -xz --strip-components=1 -C /opt/apac/cutlass
/opt/sglang/venv/bin/python /opt/apac/prefill-moe-tiles/build.py
/opt/sglang/venv/bin/python -c 'import torch, apac_prefill_moe_tiles; print(apac_prefill_moe_tiles.__file__)'
cp /opt/apac/cutlass/LICENSE.txt /opt/apac/cutlass-LICENSE.txt
find /opt/apac/cutlass -depth -delete
find /opt/apac/prefill-moe-tiles -maxdepth 1 -type f -name '*.o' -delete
