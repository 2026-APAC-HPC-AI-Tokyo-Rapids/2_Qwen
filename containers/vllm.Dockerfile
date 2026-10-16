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

COPY containers/fix-vllm-dense-draft-ep.py /opt/apac/fix-vllm-dense-draft-ep.py
RUN python3 /opt/apac/fix-vllm-dense-draft-ep.py /usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py >/opt/apac/dense-draft-ep-repair.json

# Private SM90 provider and explicit opt-in Qwen3 DP4/EP4 serving backend.
COPY containers/firmus-sm90-megamoe /opt/apac/firmus-sm90-megamoe
COPY sm90-provider /opt/apac/sm90-provider
COPY sm90-native-source /tmp/apac-sm90-native-source
RUN python3 /opt/apac/firmus-sm90-megamoe/install_vllm_adapter.py /usr/local/lib/python3.12/dist-packages/vllm >/opt/apac/vllm-sm90-adapter.json && bash /opt/apac/firmus-sm90-megamoe/check_cpu_image.sh && rm -rf /tmp/apac-sm90-native-source

COPY containers/fix-vllm-prefill-handoff.py /opt/apac/fix-vllm-prefill-handoff.py
RUN python3 /opt/apac/fix-vllm-prefill-handoff.py /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py >/opt/apac/prefill-handoff-repair.json

# Build identity and the final checker do not invalidate native compile layers.
COPY containers/check-vllm-image.py /opt/apac/check-vllm-image.py
COPY apac-manifest.json /opt/apac/manifest.json

# Keep the official OpenAI server entrypoint for docker run.
