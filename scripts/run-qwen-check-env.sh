#!/usr/bin/env bash
set -euo pipefail

: "${RUNTIME_DIR:?runtime directory is required}"

rank="${CHECK_ENV_RANK:-${PMI_RANK:-0}}"
[[ "${rank}" =~ ^[0-9]+$ ]]
if [[ "${CHECK_ENV_DEFER_AGGREGATE:-0}" == 1 ]]; then
  rank_output="${RUNTIME_DIR}/logs/rank-${rank}"
  environment_log="${rank_output}/environment.log"
  check_log="${rank_output}/check-env.log"
else
  rank_output="${RUNTIME_DIR}/logs"
  environment_log="${rank_output}/environment-rank-${rank}.log"
  check_log="${rank_output}/check-env-rank-${rank}.log"
fi
install -d -m 0755 "${rank_output}"

container_image="${RUNTIME_DIR}/sglang.sqsh"
[[ -f "${container_image}" ]]

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
  printf 'rank=%s\n' "${rank}"
  printf 'job_id=%s\n' "${PJM_JOBID:-${PBS_JOBID:-${SLURM_JOB_ID:-unknown}}}"
  printf 'enroot_version=%s\n' "$(enroot version)"
  printf 'squashfuse=%s\n' "$(command -v squashfuse)"
  printf 'fusermount3=%s\n' "$(command -v fusermount3)"
  printf 'container_image=%s\n' "${container_image}"
  printf 'container_size=%s\n' "$(stat -c '%s' "${container_image}")"
  nvidia-smi -L
} >"${environment_log}"

{
  enroot start -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_DISABLE_REQUIRE=1 "${container_image}" /opt/sglang/venv/bin/python -c 'from importlib.metadata import version; print("NIXL: nixl={} nixl-cu13={}".format(version("nixl"), version("nixl-cu13")))'
  enroot start -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_DISABLE_REQUIRE=1 "${container_image}" /opt/sglang/venv/bin/python -m sglang.check_env
} 2>&1 | tee "${check_log}"
grep -q '^NIXL: nixl=' "${check_log}"
grep -q '^Python:' "${check_log}"
grep -q '^PyTorch:' "${check_log}"
grep -q '^sglang:' "${check_log}"
grep -q '^CUDA available: True$' "${check_log}"
grep -Eq '^GPU [0-9,]*0[0-9,]*:' "${check_log}"
grep -q '^NVIDIA Topology:' "${check_log}"
if [[ "${CHECK_ENV_DEFER_AGGREGATE:-0}" == 1 ]]; then
  touch "${rank_output}/run-success"
elif [[ -n "${PMI_RANK:-}" ]]; then
  touch "${RUNTIME_DIR}/logs/run-success-rank-${rank}"
else
  touch "${RUNTIME_DIR}/run-success"
fi
