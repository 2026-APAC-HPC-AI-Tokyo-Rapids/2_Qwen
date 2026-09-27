FROM docker.io/nvidia/cuda@sha256:5a2d3b02eb7412847d051d0f2b0f0a5031057a0172d9ca78743cc41cfc5d037f

LABEL org.opencontainers.image.title="Source-built SGLang runtime"
LABEL org.opencontainers.image.description="SGLang main built with uv for rootless OCI and Enroot execution"
LABEL org.opencontainers.image.source="https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src"
LABEL ai.sglang.nccl-tests-source="https://github.com/NVIDIA/nccl-tests"
LABEL ai.sglang.nccl-tests-ref="717b68318278e93f371d8ffb46b076069d7c7851"
LABEL ai.sglang.deepep-commit="9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee"
LABEL ai.sglang.mooncake-version="0.3.11.post1"
LABEL ai.sglang.mscclpp-version="sglang-v0.9.1"

COPY containers/install-system-toolchain.sh /opt/apac/install-system-toolchain.sh
RUN /opt/apac/install-system-toolchain.sh && rm /opt/apac/install-system-toolchain.sh

COPY containers/build-nccl-tests.sh /opt/apac/build-nccl-tests.sh
RUN /opt/apac/build-nccl-tests.sh && rm /opt/apac/build-nccl-tests.sh

COPY sglang-project-metadata /opt/sglang-project-metadata
COPY containers/install-python-dependencies.sh /opt/apac/install-python-dependencies.sh
RUN /opt/apac/install-python-dependencies.sh && rm /opt/apac/install-python-dependencies.sh

COPY containers/install-accelerator-dependencies.sh /opt/apac/install-accelerator-dependencies.sh
RUN /opt/apac/install-accelerator-dependencies.sh && rm /opt/apac/install-accelerator-dependencies.sh

COPY sglang-source /opt/sglang/source
COPY sglang-commit.txt /opt/sglang/sglang-commit.txt
COPY containers/install-sglang-source.sh /opt/apac/install-sglang-source.sh
RUN /opt/apac/install-sglang-source.sh && rm /opt/apac/install-sglang-source.sh

RUN uv venv --python /opt/sglang/venv/bin/python /opt/trtllm/venv && uv pip install --python /opt/trtllm/venv/bin/python --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match 'tensorrt-llm @ https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.3.0rc20-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=cd7387185d8c2a5ab47d426fde17cd6590ebc346fe82ab58847c491243e1811a' 'torch==2.10.0+cu130' 'nixl-cu13==1.0.1' 'nixl==1.0.1' && /opt/trtllm/venv/bin/python -c "from importlib.metadata import version; assert version('tensorrt-llm') == '1.3.0rc20'" && uv cache clean

# DeepGEMM discovers its bundled headers by running pip show tensorrt_llm.
RUN uv pip install --python /opt/trtllm/venv/bin/python "pip==25.3" && /opt/trtllm/venv/bin/pip show tensorrt_llm && /opt/trtllm/venv/bin/python -c "from importlib.metadata import distribution; from pathlib import Path; p = Path(distribution('tensorrt-llm').locate_file('tensorrt_llm/include')); assert (p / 'deep_gemm/fp8_gemm_impl.cuh').is_file(), p; assert (p / 'cuda_fp8.h').is_file(), p" && uv cache clean

COPY containers/validate-trtllm-jit.py /opt/apac/validate-trtllm-jit.py
RUN PATH=/opt/trtllm/venv/bin:$PATH /opt/trtllm/venv/bin/python /opt/apac/validate-trtllm-jit.py && rm /opt/apac/validate-trtllm-jit.py

COPY apac-manifest.json /opt/apac/manifest.json

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/sglang/venv/bin:/opt/nccl-tests/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
