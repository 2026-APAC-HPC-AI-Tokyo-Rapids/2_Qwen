#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_IMAGE:?Enroot image is required}"
: "${RUNTIME_DIR:?runtime directory is required}"

export ENROOT_LIBRARY_PATH="${RUNTIME_DIR}/enroot-library"
export ENROOT_SYSCONF_PATH="${RUNTIME_DIR}/enroot-system"
export ENROOT_MOUNT_HOME=no
node_temporary_root="${TMPDIR:-${TEMPDIR:-${TEMP:-/tmp}}}"
[[ "${node_temporary_root}" == /* && -d "${node_temporary_root}" && -w "${node_temporary_root}" ]]
node_temporary="$(mktemp -d "${node_temporary_root%/}/qwen-check-env.XXXXXXXXXX")"
export ENROOT_CACHE_PATH="${node_temporary}/cache"
export ENROOT_DATA_PATH="${node_temporary}/data"
export ENROOT_RUNTIME_PATH="${node_temporary}/runtime"
export ENROOT_TEMP_PATH="${node_temporary}/temp"
install -d -m 0700 "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}"
cleanup() {
  status=$?
  trap - EXIT
  rm -rf -- "${node_temporary:?}"
  exit "${status}"
}
trap cleanup EXIT

{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'job_id=%s\n' "${PJM_JOBID:-${PBS_JOBID:-${SLURM_JOB_ID:-unknown}}}"
  printf 'enroot_version=%s\n' "$(enroot version)"
  printf 'squashfuse=%s\n' "$(command -v squashfuse)"
  printf 'fusermount3=%s\n' "$(command -v fusermount3)"
  printf 'container_image=%s\n' "${CONTAINER_IMAGE}"
  printf 'container_size=%s\n' "$(stat -c '%s' "${CONTAINER_IMAGE}")"
} >"${RUNTIME_DIR}/logs/environment.log"

enroot start -e NVIDIA_VISIBLE_DEVICES=void -e NVIDIA_DRIVER_CAPABILITIES=void "${CONTAINER_IMAGE}" /opt/sglang/venv/bin/python -m sglang.check_env 2>&1 | tee "${RUNTIME_DIR}/logs/check-env.log"
grep -q '^Python:' "${RUNTIME_DIR}/logs/check-env.log"
grep -q '^PyTorch:' "${RUNTIME_DIR}/logs/check-env.log"
grep -q '^sglang:' "${RUNTIME_DIR}/logs/check-env.log"
touch "${RUNTIME_DIR}/run-success"
