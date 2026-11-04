#!/usr/bin/env bash
set -euxo pipefail

export SGLANG_BUILD_RUST_EXTS=none
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:/usr/local/bin:${PATH}
export UV_CACHE_DIR=/opt/sglang/uv-cache
export UV_PROJECT_ENVIRONMENT=/opt/sglang/venv

git -C /opt/sglang/source hash-object python/sglang/srt/server_args.py | tee /opt/sglang/server-args-source-blob-oid.txt

cd /opt/sglang/source/python
uv pip install --python /opt/sglang/venv/bin/python --no-deps -e .
cd /opt/sglang/source
/opt/sglang/venv/bin/kernels lock python
/opt/sglang/venv/bin/kernels download python
uv pip freeze --python /opt/sglang/venv/bin/python > /opt/sglang/requirements-frozen.txt
/opt/sglang/venv/bin/python -m compileall -q /opt/sglang/source/python/sglang

rm -rf /root/.cache /opt/sglang/source/.git /opt/sglang/uv-cache /opt/DeepEP/.git /var/tmp/sglang-build
install -d -m 0755 /root/.cache
