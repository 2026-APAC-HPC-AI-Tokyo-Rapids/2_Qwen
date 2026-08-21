#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:/usr/local/bin:${PATH}

apt-get -o APT::Sandbox::User=root update
apt-get -o APT::Sandbox::User=root install -y --no-install-recommends --allow-change-held-packages build-essential ca-certificates cmake curl ffmpeg git git-lfs ibverbs-providers libc6-dbg libboost-all-dev libcurl4-openssl-dev libczmq-dev libfabric-dev libgoogle-glog-dev libgrpc++-dev libhiredis-dev libhwloc-dev libibumad3 libibverbs-dev libnccl-dev libnccl2 libnl-3-dev libnl-route-3-dev libnuma-dev libopenmpi-dev libpmix-dev libprotobuf-dev librdmacm-dev libssl-dev libucx-dev libunwind-dev linux-libc-dev ninja-build numactl openmpi-bin patch patchelf perl pkg-config protobuf-compiler protobuf-compiler-grpc pybind11-dev python3.12-dev python3.12-full rdma-core ucx-utils unzip wget
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2
update-alternatives --set python3 /usr/bin/python3.12
ln -sf /usr/bin/python3.12 /usr/bin/python
curl -LsSf https://astral.sh/uv/install.sh | sh
install -m 0755 /root/.local/bin/uv /root/.local/bin/uvx /usr/local/bin/

export RUSTUP_HOME=/opt/rustup
export CARGO_HOME=/opt/cargo
curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0

rm -rf /var/lib/apt/lists/* /root/.cache
install -d -m 0755 /root/.cache /var/tmp/sglang-build /opt/sglang
