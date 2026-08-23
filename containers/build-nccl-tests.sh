#!/usr/bin/env bash
set -euxo pipefail

export TMPDIR=/var/tmp/sglang-build
export MAX_JOBS=40

git clone https://github.com/NVIDIA/nccl-tests.git "${TMPDIR}/nccl-tests"
git -C "${TMPDIR}/nccl-tests" checkout --detach 717b68318278e93f371d8ffb46b076069d7c7851
git -C "${TMPDIR}/nccl-tests" rev-parse HEAD | tee /opt/sglang/nccl-tests-commit.txt
make -C "${TMPDIR}/nccl-tests" -j "${MAX_JOBS}" CC=/usr/bin/gcc CXX=/usr/bin/g++ MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90'
install -d -m 0755 /opt/nccl-tests/bin
find "${TMPDIR}/nccl-tests/build" -maxdepth 1 -type f -name '*_perf' -exec install -m 0755 '{}' /opt/nccl-tests/bin/ ';'
install -m 0644 "${TMPDIR}/nccl-tests/LICENSE.txt" /opt/nccl-tests/LICENSE.txt
test "$(find /opt/nccl-tests/bin -maxdepth 1 -type f -name '*_perf' | wc -l)" -eq 11
rm -rf "${TMPDIR}/nccl-tests"
