#!/usr/bin/env bash
# CPU builds use the CUDA toolkit selectively extracted from the pinned image.
export CUDA_HOME="${RUNTIME_DIR}/toolchain/usr/local/cuda-13.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CXX=/usr/bin/g++
