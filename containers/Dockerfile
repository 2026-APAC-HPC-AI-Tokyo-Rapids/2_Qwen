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

RUN uv venv --python /opt/sglang/venv/bin/python /opt/vllm/venv && uv pip install --python /opt/vllm/venv/bin/python 'vllm==0.24.0' 'nixl==1.2.0' 'nixl-cu13==1.2.0' && /opt/vllm/venv/bin/python -c "import vllm, nixl; print(vllm.__version__)" && uv cache clean

RUN curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/ee0da84ab9e04ac7610e28580af62c365e898389/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py -o /opt/vllm/proxy.py && echo '0ba88d1e77d0bd5f4f2f6e69f1bf6e29fbc258e64dbed695c0d968ff14720fd8  /opt/vllm/proxy.py' | sha256sum -c -

# The pinned toy proxy labels SSE as JSON; AIPerf selects its decoder from Content-Type.
RUN sed -i 's/media_type="application\/json"/media_type="text\/event-stream" if req_data.get("stream", False) else "application\/json"/' /opt/vllm/proxy.py

COPY apac-manifest.json /opt/apac/manifest.json

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/sglang/venv/bin:/opt/nccl-tests/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
