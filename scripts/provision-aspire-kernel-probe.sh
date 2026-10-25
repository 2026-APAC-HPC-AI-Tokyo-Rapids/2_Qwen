#!/usr/bin/env bash
# CPU-only provisioning; submission belongs exclusively to the workflow.
set -euo pipefail
: "${GITHUB_WORKSPACE:?}" "${GITHUB_RUN_ID:?}" "${GITHUB_RUN_ATTEMPT:?}" "${GITHUB_OUTPUT:?}" "${RUNTIME_ROOT:?}"
benchmark=${BENCHMARK:-kernels/probes/prepare-fa3-prefill.sh}
# Only the bounded single-GPU FA3 protocols are currently validated here.
case "${benchmark}" in kernels/probes/prepare-fa3-prefill.sh|kernels/probes/prepare-fa3-prefill-profile.sh) ;; *) printf 'Unsupported ASPIRE probe: %s\n' "${benchmark}" >&2; exit 64 ;; esac
[[ -z "${CANDIDATE_PATCH:-}" ]]
source "${HOME}/.local/.setenv"
config="${GITHUB_WORKSPACE}/configs/firmus-framework.json"
image="/scratch/users/industry/ai-hpc/ACCOUNT/sglang-containers/$(basename "$(jq -er .images.sglang.path "${config}")")"
[[ "$(stat -c '%s' "${image}")" == "$(jq -er .images.sglang.size_bytes "${config}")" ]]
[[ "$(jq -er .container_artifact.sha256 "${image%.sqsh}.container-sidecar.json")" == "$(jq -er .images.sglang.sha256 "${config}")" ]]
install -d -m 2775 "${RUNTIME_ROOT}"
export RUNTIME_DIR
RUNTIME_DIR=$(mktemp -d "${RUNTIME_ROOT}/kernel-probe-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.XXXXXXXX")
printf 'runtime_dir=%s\n' "${RUNTIME_DIR}" >>"${GITHUB_OUTPUT}"
install -d "${RUNTIME_DIR}/logs" "${RUNTIME_DIR}/source" "${RUNTIME_DIR}/results" "${RUNTIME_DIR}/scripts" "${RUNTIME_DIR}/enroot-library" "${RUNTIME_DIR}/enroot-system"
ln -s "${image}" "${RUNTIME_DIR}/sglang.sqsh"
cp "${image%.sqsh}.container-sidecar.json" "${RUNTIME_DIR}/logs/container-sidecar.json"
cp -a --no-preserve=ownership,xattr "${ENROOT_LIBRARY_PATH}/." "${RUNTIME_DIR}/enroot-library/"
cp -a --no-preserve=ownership,xattr "${ENROOT_SYSCONF_PATH}/." "${RUNTIME_DIR}/enroot-system/"
sed -i 's/local timeout=100 pid=-1 id=0/local timeout=5000 pid=-1 id=0/' "${RUNTIME_DIR}/enroot-library/runtime.sh"
sed -i 's%--ldconfig=@$(command -v ldconfig.real || command -v ldconfig)%--ldconfig=/sbin/ldconfig%' "${RUNTIME_DIR}/enroot-system/hooks.d/98-nvidia.sh"
install -m 0644 "${GITHUB_WORKSPACE}/platforms/aspire2ap/kernel-probe/build-env.sh" "${RUNTIME_DIR}/scripts/build-env.sh"
install -m 0755 "${GITHUB_WORKSPACE}/platforms/aspire2ap/kernel-probe/run.pbs" "${RUNTIME_DIR}/scripts/run.pbs"
if [[ "${benchmark}" == kernels/probes/prepare-fa3-prefill-profile.sh ]]; then
  # Pin the profiler to the same existing image, without unpacking the whole image.
  unsquashfs -no-progress -no-xattrs -processors 1 -d "${RUNTIME_DIR}/profiler" "${image}" opt/nvidia/nsight-compute/2025.3.1 >"${RUNTIME_DIR}/logs/extract-ncu.log" 2>&1
  chmod -R u+w "${RUNTIME_DIR}/profiler"
  install -d "${RUNTIME_DIR}/bin"
  ln -s "${RUNTIME_DIR}/profiler/opt/nvidia/nsight-compute/2025.3.1/ncu" "${RUNTIME_DIR}/bin/ncu"
  "${RUNTIME_DIR}/bin/ncu" --version >"${RUNTIME_DIR}/logs/ncu-version.txt"
fi
bash "${GITHUB_WORKSPACE}/${benchmark}"
test -x "${RUNTIME_DIR}/source/probe"
jq --arg benchmark "${benchmark}" --arg image "${image}" --arg cxx /app/apps/gcc/13.3.0/bin/g++ '. + {cluster:"aspire2ap",benchmark:$benchmark,image:$image,cuda:"12.6.2",cxx:$cxx}' "${RUNTIME_DIR}/source/provenance.json" >"${RUNTIME_DIR}/source/provenance.tmp"
mv "${RUNTIME_DIR}/source/provenance.tmp" "${RUNTIME_DIR}/source/provenance.json"
cp "${RUNTIME_DIR}/source/provenance.json" "${RUNTIME_DIR}/logs/provenance.json"
