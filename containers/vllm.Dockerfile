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

# Keep the official OpenAI server entrypoint for docker run.
