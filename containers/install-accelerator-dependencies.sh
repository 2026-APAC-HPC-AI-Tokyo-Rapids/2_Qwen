#!/usr/bin/env bash
set -euxo pipefail

export CUDA_HOME=/usr/local/cuda
export TMPDIR=/var/tmp/sglang-build

deepep_commit=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
curl -fsSL -o "${TMPDIR}/deepep.zip" "https://github.com/deepseek-ai/DeepEP/archive/${deepep_commit}.zip"
unzip -q "${TMPDIR}/deepep.zip" -d "${TMPDIR}"
mv "${TMPDIR}/DeepEP-${deepep_commit}" /opt/DeepEP
sed -i "/^    include_dirs = \['csrc\/'\]/a\\    include_dirs.append('${CUDA_HOME}/include/cccl')" /opt/DeepEP/setup.py
TORCH_CUDA_ARCH_LIST=9.0 uv pip install --python /opt/sglang/venv/bin/python --no-build-isolation /opt/DeepEP
rm "${TMPDIR}/deepep.zip"

uv pip install --python /opt/sglang/venv/bin/python "mooncake-transfer-engine-cuda13==0.3.11.post1"
uv pip install --python /opt/sglang/venv/bin/python --no-deps nixl nixl-cu13
git clone --depth=1 --branch sglang-v0.9.1 https://github.com/microsoft/mscclpp.git "${TMPDIR}/mscclpp"
export CMAKE_ARGS="-DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_CUDA=ON -DMSCCLPP_GPU_ARCHS=90"
uv pip install --python /opt/sglang/venv/bin/python "${TMPDIR}/mscclpp[cuda13]"
unset CMAKE_ARGS
rm -rf "${TMPDIR}/mscclpp"
