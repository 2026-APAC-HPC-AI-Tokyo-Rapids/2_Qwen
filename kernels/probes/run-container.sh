#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?}"
export ENROOT_LIBRARY_PATH="${RUNTIME_DIR}/enroot-library" ENROOT_SYSCONF_PATH="${RUNTIME_DIR}/enroot-system" ENROOT_MOUNT_HOME=no
node_tmp=$(mktemp -d "${TMPDIR:-/tmp}/sglang-probe.XXXXXXXX")
trap 'rm -rf -- "${node_tmp:?}"' EXIT
export ENROOT_CACHE_PATH="${node_tmp}/cache" ENROOT_DATA_PATH="${node_tmp}/data" ENROOT_RUNTIME_PATH="${node_tmp}/runtime" ENROOT_TEMP_PATH="${node_tmp}/temp"
install -d "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}" "${RUNTIME_DIR}/cache" "${RUNTIME_DIR}/home"
mounts=()
if [[ "$1" == /opt/fa3-ncu/ncu ]]; then
  ncu_root=$(dirname "$(readlink -f "$(command -v ncu)")")
  mounts+=(-m "${ncu_root}:/opt/fa3-ncu")
fi
enroot start "${mounts[@]}" -m "${RUNTIME_DIR}:${RUNTIME_DIR}" -m "${RUNTIME_DIR}/home:${HOME}" -m "${RUNTIME_DIR}/source/fa3-verify-candidates.json:/opt/apac/fa3-verify/candidates.json:none:x-create=file,bind,ro,nosuid,nodev" -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_DISABLE_REQUIRE=1 -e "PYTHONPATH=${RUNTIME_DIR}/source/python" -e "TVM_FFI_CACHE_DIR=${RUNTIME_DIR}/cache/tvm" -e "TRITON_CACHE_DIR=${RUNTIME_DIR}/cache/triton" "${RUNTIME_DIR}/sglang.sqsh" "$@"
