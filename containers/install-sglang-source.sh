#!/usr/bin/env bash
set -euxo pipefail

export CARGO_HOME=/opt/cargo
export RUSTUP_HOME=/opt/rustup
export CARGO_PROFILE_RELEASE_DEBUG=2
export CARGO_PROFILE_RELEASE_STRIP=none
export CUDA_HOME=/usr/local/cuda
export PATH=/opt/cargo/bin:/usr/local/cuda/bin:/usr/local/bin:${PATH}
export UV_CACHE_DIR=/opt/sglang/uv-cache
export UV_PROJECT_ENVIRONMENT=/opt/sglang/venv

while IFS= read -r patch_file; do patch -d /opt/sglang/source -p1 < "${patch_file}"; done < <(find /opt/sglang-patches -type f -name '*.patch' -print | sort)
git -C /opt/sglang/source hash-object python/sglang/benchmark/offline_throughput.py | tee /opt/sglang/offline-throughput-source-blob-oid.txt
git -C /opt/sglang/source hash-object python/sglang/srt/server_args.py | tee /opt/sglang/server-args-source-blob-oid.txt
printf '2\n' > /opt/sglang/qwen-experiment-metrics-hook-version.txt

cd /opt/sglang/source/python
uv pip install --python /opt/sglang/venv/bin/python --no-deps -e .
cd /opt/sglang/source
/opt/sglang/venv/bin/kernels lock python
/opt/sglang/venv/bin/kernels download python
uv pip freeze --python /opt/sglang/venv/bin/python > /opt/sglang/requirements-frozen.txt
/opt/sglang/venv/bin/python -m compileall -q /opt/sglang/source/python/sglang

rm -rf /root/.cache /opt/sglang/source/.git /opt/sglang-patches /opt/sglang/uv-cache /opt/cargo /opt/rustup /opt/DeepEP/.git /var/tmp/sglang-build
install -d -m 0755 /root/.cache
