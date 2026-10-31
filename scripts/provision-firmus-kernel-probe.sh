#!/usr/bin/env bash
# CPU-only provisioning. The workflow owns Slurm submission and cleanup.
set -euo pipefail
: "${GITHUB_WORKSPACE:?}" "${GITHUB_RUN_ID:?}" "${GITHUB_RUN_ATTEMPT:?}" "${GITHUB_SHA:?}" "${GITHUB_OUTPUT:?}" "${RUNTIME_ROOT:?}"
benchmark=${BENCHMARK:-kernels/probes/prepare-fa3-prefill.sh}
case "${benchmark}" in kernels/probes/prepare-fa3-prefill.sh|kernels/probes/prepare-fa3-prefill-profile.sh|kernels/probes/prepare-dense-decode.sh|kernels/probes/prepare-firmus-eagle-accept.sh|kernels/probes/prepare-prefill-memory.sh|kernels/probes/prepare-fa3-exp.sh|kernels/probes/prepare-fa3-split-pv.sh) ;; *) printf 'Unsupported Firmus probe: %s\n' "${benchmark}" >&2; exit 64 ;; esac
[[ -z "${CANDIDATE_PATCH:-}" ]]
config="${GITHUB_WORKSPACE}/configs/firmus-framework.json"
image=$(jq -er .images.sglang.path "${config}")
[[ "$(stat -c '%s' "${image}")" == "$(jq -er .images.sglang.size_bytes "${config}")" ]]
[[ "$(jq -er .container_artifact.sha256 "${image%.sqsh}.container-sidecar.json")" == "$(jq -er .images.sglang.sha256 "${config}")" ]]
install -d -m 2775 "${RUNTIME_ROOT}"
export RUNTIME_DIR
RUNTIME_DIR=$(mktemp -d "${RUNTIME_ROOT}/kernel-probe-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.XXXXXXXX")
printf 'runtime_dir=%s\n' "${RUNTIME_DIR}" >>"${GITHUB_OUTPUT}"
install -d "${RUNTIME_DIR}/logs" "${RUNTIME_DIR}/source" "${RUNTIME_DIR}/results" "${RUNTIME_DIR}/scripts" "${RUNTIME_DIR}/home" "${RUNTIME_DIR}/bin" "${RUNTIME_DIR}/metadata"
ln -s "${image}" "${RUNTIME_DIR}/sglang.sqsh"
cp "${image%.sqsh}.container-sidecar.json" "${RUNTIME_DIR}/logs/container-sidecar.json"
sha256sum "${image}" >"${RUNTIME_DIR}/logs/image.sha256"
[[ "$(cut -d ' ' -f1 "${RUNTIME_DIR}/logs/image.sha256")" == "$(jq -er .images.sglang.sha256 "${config}")" ]]
for script in build-env.sh run-container.sh run-trials.sh run.sbatch; do install -m 0755 "${GITHUB_WORKSPACE}/platforms/firmus/kernel-probe/${script}" "${RUNTIME_DIR}/scripts/${script}"; done
if [[ "${benchmark}" == kernels/probes/prepare-firmus-eagle-accept.sh ]]; then
    bash "${GITHUB_WORKSPACE}/${benchmark}"
    sed -i 's/00:05:00/00:17:00/' "${RUNTIME_DIR}/scripts/run.sbatch"
elif [[ "${benchmark}" == kernels/probes/prepare-prefill-memory.sh ]]; then
    bash "${GITHUB_WORKSPACE}/${benchmark}"
    install -m 0644 "${RUNTIME_DIR}/source/fa3-verify-candidates.json" "${RUNTIME_DIR}/metadata/candidates.json"
else
    # The login container has no CUDA module. Extract compiler, headers and static runtime.
    unsquashfs -no-progress -no-xattrs -processors 1 -d "${RUNTIME_DIR}/toolchain" "${image}" usr/local/cuda-13.0/bin usr/local/cuda-13.0/nvvm usr/local/cuda-13.0/include usr/local/cuda-13.0/lib64 usr/local/cuda-13.0/targets/x86_64-linux/include usr/local/cuda-13.0/targets/x86_64-linux/lib/libcudart_static.a usr/local/cuda-13.0/targets/x86_64-linux/lib/libculibos.a usr/local/cuda-13.0/targets/x86_64-linux/lib/libcudadevrt.a >"${RUNTIME_DIR}/logs/extract-cuda.log" 2>&1
    chmod -R u+w "${RUNTIME_DIR}/toolchain"
    if [[ "${benchmark}" != kernels/probes/prepare-fa3-exp.sh && "${benchmark}" != kernels/probes/prepare-fa3-split-pv.sh ]]; then
        unsquashfs -no-progress -no-xattrs -processors 1 -d "${RUNTIME_DIR}/profiler" "${image}" opt/nvidia/nsight-compute/2025.3.1 >"${RUNTIME_DIR}/logs/extract-ncu.log" 2>&1
        chmod -R u+w "${RUNTIME_DIR}/profiler"
        ln -s "${RUNTIME_DIR}/profiler/opt/nvidia/nsight-compute/2025.3.1/ncu" "${RUNTIME_DIR}/bin/ncu"
        "${RUNTIME_DIR}/bin/ncu" --version >"${RUNTIME_DIR}/logs/ncu-version.txt"
    fi
    source "${RUNTIME_DIR}/scripts/build-env.sh"
    nvcc --version >"${RUNTIME_DIR}/logs/nvcc-version.txt"
    "${CXX}" --version >"${RUNTIME_DIR}/logs/cxx-version.txt"
    if [[ "${benchmark}" != kernels/probes/prepare-fa3-exp.sh && "${benchmark}" != kernels/probes/prepare-fa3-split-pv.sh ]]; then
        install -m 0644 "${GITHUB_WORKSPACE}/kernels/probes/ncu-permission.cu" "${RUNTIME_DIR}/source/ncu-permission.cu"
        nvcc -ccbin "${CXX}" -gencode arch=compute_90a,code=sm_90a --cudart static "${RUNTIME_DIR}/source/ncu-permission.cu" -o "${RUNTIME_DIR}/source/ncu-permission" >"${RUNTIME_DIR}/logs/build-ncu-permission.log" 2>&1
    fi
    bash "${GITHUB_WORKSPACE}/${benchmark}"
    test -x "${RUNTIME_DIR}/source/probe"
    if [[ "${benchmark}" != kernels/probes/prepare-fa3-exp.sh && "${benchmark}" != kernels/probes/prepare-fa3-split-pv.sh ]]; then install -m 0644 "${RUNTIME_DIR}/source/fa3-verify-candidates.json" "${RUNTIME_DIR}/metadata/candidates.json"; fi
    if [[ "${benchmark}" == kernels/probes/prepare-dense-decode.sh ]]; then install -m 0644 "${RUNTIME_DIR}/source/dense-candidates.json" "${RUNTIME_DIR}/metadata/dense-candidates.json"; fi
fi
test -x "${RUNTIME_DIR}/source/probe"
probe_gpus=1
if [[ "${benchmark}" == kernels/probes/prepare-firmus-eagle-accept.sh ]]; then probe_gpus=4; fi
jq --argjson probe_gpus "${probe_gpus}" --arg benchmark "${benchmark}" --arg image "${image}" '. + {cluster:"firmus",benchmark:$benchmark,image:$image,cuda:"13.0",cxx:"/usr/bin/g++",allocated_gpus:8,probe_gpus:$probe_gpus}' "${RUNTIME_DIR}/source/provenance.json" >"${RUNTIME_DIR}/source/provenance.tmp"
mv "${RUNTIME_DIR}/source/provenance.tmp" "${RUNTIME_DIR}/source/provenance.json"
cp "${RUNTIME_DIR}/source/provenance.json" "${RUNTIME_DIR}/logs/provenance.json"

if [[ "${benchmark}" == kernels/probes/prepare-prefill-memory.sh || "${benchmark}" == kernels/probes/prepare-fa3-exp.sh || "${benchmark}" == kernels/probes/prepare-fa3-split-pv.sh ]]; then cp "${RUNTIME_DIR}/source/provenance.json" "${RUNTIME_DIR}/results/source/provenance.json"; fi
