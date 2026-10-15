# vllm/vllm-openai:v0.30.0, the official image used by the Firmus baseline.
FROM docker.io/vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90

LABEL org.opencontainers.image.title="APAC vLLM runtime"
LABEL org.opencontainers.image.source="https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen"
LABEL ai.vllm.version="0.30.0"

# Prefer the CUDA 13.0 compiler matching the runtime headers over wheel nvcc.
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# SquashFS has no OCI environment metadata; Enroot reads /etc/environment.
RUN sed -i '/^PATH=/d; /^NVIDIA_VISIBLE_DEVICES=/d; /^NVIDIA_DRIVER_CAPABILITIES=/d' /etc/environment && printf 'PATH="%s"\nNVIDIA_VISIBLE_DEVICES=all\nNVIDIA_DRIVER_CAPABILITIES=compute,utility\n' "$PATH" >>/etc/environment

COPY apac-manifest.json /opt/apac/manifest.json
COPY containers/check-vllm-image.py /opt/apac/check-vllm-image.py
COPY containers/fix-vllm-dense-draft-ep.py /opt/apac/fix-vllm-dense-draft-ep.py
RUN python3 /opt/apac/fix-vllm-dense-draft-ep.py /usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py >/opt/apac/dense-draft-ep-repair.json

# CPU-only native-ABI gate for the future decoder adapter. No serving backend is changed.
COPY containers/firmus-sm90-megamoe /opt/apac/firmus-sm90-megamoe
COPY sm90-provider /opt/apac/sm90-provider
RUN bash /opt/apac/firmus-sm90-megamoe/check_cpu_image.sh

# Keep the official OpenAI server entrypoint for docker run.
