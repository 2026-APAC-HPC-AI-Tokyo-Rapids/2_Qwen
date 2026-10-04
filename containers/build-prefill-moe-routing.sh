#!/bin/bash
set -euxo pipefail
/opt/sglang/venv/bin/python /opt/apac/prefill-moe-routing/build.py
/opt/sglang/venv/bin/python -c 'import torch, apac_prefill_moe_routing; print(apac_prefill_moe_routing.__file__)'
find /opt/apac/prefill-moe-routing -maxdepth 1 -type f -name '*.o' -delete
