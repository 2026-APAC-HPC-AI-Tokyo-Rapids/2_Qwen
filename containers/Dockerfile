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

COPY prefill-load-lifetime.patch sglang-router.Cargo.lock /opt/apac/
COPY containers/build-sglang-router.sh /opt/apac/build-sglang-router.sh
RUN /opt/apac/build-sglang-router.sh && rm /opt/apac/build-sglang-router.sh

COPY containers/install-accelerator-dependencies.sh /opt/apac/install-accelerator-dependencies.sh
RUN /opt/apac/install-accelerator-dependencies.sh && rm /opt/apac/install-accelerator-dependencies.sh

COPY sm90-late-topk-fp32.patch /opt/apac/sm90-late-topk-fp32.patch
COPY sm90-fc2-pipeline.patch /opt/apac/sm90-fc2-pipeline.patch
COPY sm90-prefill-rescale.patch /opt/apac/sm90-prefill-rescale.patch
COPY containers/build-sm90-megamoe.py /opt/apac/build-sm90-megamoe.py
RUN /opt/sglang/venv/bin/python /opt/apac/build-sm90-megamoe.py

COPY sglang-source /opt/sglang/source
COPY sglang-commit.txt /opt/sglang/sglang-commit.txt
COPY containers/install-sglang-source.sh /opt/apac/install-sglang-source.sh
RUN /opt/apac/install-sglang-source.sh && rm /opt/apac/install-sglang-source.sh
COPY containers/check-decode-kernel-build.py /opt/apac/check-decode-kernel-build.py
RUN /opt/sglang/venv/bin/python /opt/apac/check-decode-kernel-build.py
COPY containers/check-rope-kernel-build.py /opt/apac/check-rope-kernel-build.py
RUN /opt/sglang/venv/bin/python /opt/apac/check-rope-kernel-build.py

COPY prefill-moe-routing /opt/apac/prefill-moe-routing
COPY containers/build-prefill-moe-routing.py /opt/apac/prefill-moe-routing/build.py
COPY containers/build-prefill-moe-routing.sh /opt/apac/build-prefill-moe-routing.sh
RUN /opt/apac/build-prefill-moe-routing.sh && rm /opt/apac/build-prefill-moe-routing.sh

COPY apac-manifest.json /opt/apac/manifest.json

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/sglang/venv/bin:/opt/nccl-tests/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
