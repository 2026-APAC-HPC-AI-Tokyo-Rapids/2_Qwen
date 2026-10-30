# Official lmsysorg/sglang:v0.5.23, linux/amd64; upstream commit 297e54fd105e974cdf401e5b4d20253a22c16f90.
FROM docker.io/lmsysorg/sglang@sha256:f7876bef696ccf9cf75a2bf7ccb82970f5fac577f082fb8ca95c9a59d2160eb5

LABEL org.opencontainers.image.title="Qwen Basic: unmodified SGLang 0.5.23"
LABEL ai.qwen.tier="basic"

# Keep benchmark and gateway dependencies out of the upstream inference environment.
RUN uv venv --python /opt/sglang/bin/python /opt/aiperf && uv pip install --python /opt/aiperf/bin/python aiperf==0.12.0 sglang-router==0.3.2 && uv pip freeze --python /opt/sglang/bin/python > /tmp/basic-inference-packages.txt
COPY apac-manifest.json /opt/apac/manifest.json
COPY containers/check-basic-image.py /opt/apac/check-basic-image.py
RUN mkdir -p /opt/apac && mv /tmp/basic-inference-packages.txt /opt/apac/inference-packages.txt && uv pip freeze --python /opt/aiperf/bin/python > /opt/apac/client-packages.txt

# Enroot imports a filesystem, so persist the paths rather than relying on OCI ENV.
ENV PATH=/opt/sglang/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN sed -i '/^PATH=/d; /^CUDA_HOME=/d; /^LD_LIBRARY_PATH=/d' /etc/environment && printf 'PATH="%s"\nCUDA_HOME=/usr/local/cuda\nLD_LIBRARY_PATH="%s"\n' "$PATH" "$LD_LIBRARY_PATH" >> /etc/environment
