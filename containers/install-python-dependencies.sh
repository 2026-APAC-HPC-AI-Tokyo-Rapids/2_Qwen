#!/usr/bin/env bash
set -euxo pipefail

export UV_CACHE_DIR=/opt/sglang/uv-cache
export UV_PROJECT_ENVIRONMENT=/opt/sglang/venv

uv sync --project /opt/sglang-project-metadata/python --no-install-project
uv pip install --python /opt/sglang/venv/bin/python setuptools wheel
uv pip install --python /opt/sglang/venv/bin/python "aiperf==0.12.0" "sglang-router==0.3.2"
flashinfer_version="$(/opt/sglang/venv/bin/python -c 'from importlib.metadata import version; print(version("flashinfer-python"))')"
uv pip install --python /opt/sglang/venv/bin/python --upgrade --index-url https://flashinfer.ai/whl "flashinfer-cubin==${flashinfer_version}"
uv pip install --python /opt/sglang/venv/bin/python --upgrade --index-url https://flashinfer.ai/whl/cu130 "flashinfer-jit-cache==${flashinfer_version}"

rm -rf /opt/sglang-project-metadata /opt/sglang/uv-cache
