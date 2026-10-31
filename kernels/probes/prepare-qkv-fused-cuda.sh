#!/usr/bin/env bash
# Fixed CUDA 13.4.3 compiler/runtime subset, shared by both probe variants.
# The Genkai login host lacks /usr/local/cuda-13.4; do not change the image.
set -euo pipefail
: "${RUNTIME_DIR:?}" "${GITHUB_WORKSPACE:?}"
install -d "${RUNTIME_DIR}/cuda-13.4" "${RUNTIME_DIR}/cuda-downloads"
while IFS=$'\t' read -r path sha256; do
  archive="${RUNTIME_DIR}/cuda-downloads/$(basename "${path}")"
  curl --fail --location --show-error --max-time 300 "https://developer.download.nvidia.com/compute/cuda/redist/${path}" -o "${archive}"
  printf '%s  %s\n' "${sha256}" "${archive}" | sha256sum --check
  tar -xJf "${archive}" --strip-components=1 -C "${RUNTIME_DIR}/cuda-13.4"
done < <(jq -r '.components[] | [.path,.sha256] | @tsv' "${GITHUB_WORKSPACE}/kernels/probes/qkv-fused-cuda13.4.json")
ln -s lib "${RUNTIME_DIR}/cuda-13.4/lib64"
rm -rf -- "${RUNTIME_DIR:?}/cuda-downloads"
"${RUNTIME_DIR}/cuda-13.4/bin/nvcc" --version
