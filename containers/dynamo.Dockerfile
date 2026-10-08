ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="APAC Dynamo PD runtime"
LABEL org.opencontainers.image.source="https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Enroot consumes /etc/environment, while SquashFS drops OCI environment metadata.
RUN python3 -c 'import os; from pathlib import Path; keys=("PATH","LD_LIBRARY_PATH","CUDA_HOME","NIXL_PREFIX","NIXL_LIB_DIR","NIXL_PLUGIN_DIR","IMAGEIO_FFMPEG_EXE","DYNAMO_COMMIT_SHA","NVIDIA_VISIBLE_DEVICES","NVIDIA_DRIVER_CAPABILITIES"); path=Path("/etc/environment"); previous=path.read_text().splitlines() if path.exists() else []; path.write_text("\n".join([line for line in previous if line.partition("=")[0] not in keys] + [key+"=\""+os.environ[key]+"\"" for key in keys if key in os.environ])+"\n")'
COPY apac-manifest.json /opt/apac/manifest.json
COPY configs/firmus-dynamo.json /opt/apac/firmus-dynamo.json
COPY containers/check-dynamo-image.py /opt/apac/check-dynamo-image.py
