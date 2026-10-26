#!/usr/bin/env bash
# Shared allocation/build entry point: the selected script creates source/probe.
set -euo pipefail
: "${GITHUB_WORKSPACE:?}" "${GITHUB_RUN_ID:?}" "${GITHUB_RUN_ATTEMPT:?}" "${GITHUB_OUTPUT:?}" "${RUNTIME_ROOT:?}"
benchmark=${BENCHMARK:-kernels/probes/prepare-megamoe-math.sh}
[[ "${benchmark}" =~ ^kernels/probes/[a-zA-Z0-9_./-]+\.sh$ && "${benchmark}" != *..* && -f "${GITHUB_WORKSPACE}/${benchmark}" ]]
install -d -m 2775 "${RUNTIME_ROOT}"
export RUNTIME_DIR
RUNTIME_DIR=$(mktemp -d "${RUNTIME_ROOT}/kernel-probe-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.XXXXXXXX")
printf 'runtime_dir=%s\n' "${RUNTIME_DIR}" >>"${GITHUB_OUTPUT}"
install -d "${RUNTIME_DIR}/logs" "${RUNTIME_DIR}/source" "${RUNTIME_DIR}/results" "${RUNTIME_DIR}/scripts"
install -m 0644 "${GITHUB_WORKSPACE}/platforms/genkai/kernel-probe/build-env.sh" "${RUNTIME_DIR}/scripts/build-env.sh"
bash "${GITHUB_WORKSPACE}/${benchmark}"
test -x "${RUNTIME_DIR}/source/probe"
install -m 0755 "${GITHUB_WORKSPACE}/platforms/genkai/kernel-probe/run.pjm" "${RUNTIME_DIR}/scripts/run.pjm"
