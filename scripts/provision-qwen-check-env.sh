#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?AWS region is required}"
: "${CLUSTER_NAME:?cluster name is required}"
: "${CONTAINER_BUCKET:?container bucket is required}"
: "${CONTAINER_KEY:?container key is required}"
: "${CONTAINER_SHA256:?container SHA-256 is required}"
: "${CONTAINER_SIZE:?container size is required}"
: "${ENROOT_LIBRARY_PATH:?Enroot library path is required}"
: "${ENROOT_SYSCONF_PATH:?Enroot system configuration path is required}"
: "${GITHUB_RUN_ATTEMPT:?workflow run attempt is required}"
: "${GITHUB_RUN_ID:?workflow run ID is required}"
: "${GITHUB_WORKSPACE:?workflow workspace is required}"
: "${PROVISION_OUTPUT:?GitHub output file is required}"
: "${RUNNER_TEMP:?GitHub runner temporary directory is required}"
: "${RUNTIME_ROOT:?runtime root is required}"
: "${SIDECAR_KEY:?container sidecar key is required}"

[[ "${CONTAINER_KEY}" == containers/*/sglang.sqsh && "${SIDECAR_KEY}" == containers/*/container-sidecar.json ]]
[[ "${CONTAINER_SHA256}" =~ ^[0-9a-f]{64}$ && "${CONTAINER_SIZE}" =~ ^[1-9][0-9]*$ ]]
[[ "${CONTAINER_KEY%/sglang.sqsh}" == "${SIDECAR_KEY%/container-sidecar.json}" ]]

export UV_CACHE_DIR="${RUNNER_TEMP}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${RUNNER_TEMP}/uv-python"
install -d -m 0700 "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"
install -d -m 2775 "${RUNTIME_ROOT}"
runtime_dir="$(mktemp -d "${RUNTIME_ROOT}/check-env-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.XXXXXXXXXX")"
printf 'runtime_dir=%s\n' "${runtime_dir}" >>"${PROVISION_OUTPUT}"
install -d -m 0755 "${runtime_dir}/enroot-library" "${runtime_dir}/enroot-system" "${runtime_dir}/logs" "${runtime_dir}/scripts" "${runtime_dir}/source"

download() {
  uvx --from 'boto3>=1.40,<2' python "${GITHUB_WORKSPACE}/scripts/s3-object-transfer.py" download "${CONTAINER_BUCKET}" "$1" "$2"
}

download "${SIDECAR_KEY}" "${runtime_dir}/source/container-sidecar.json"
expected_size="$(jq -er 'select(.schema_version == "container-sidecar/v1") | .container_artifact.size_bytes' "${runtime_dir}/source/container-sidecar.json")"
expected_sha256="$(jq -er '.container_artifact.sha256' "${runtime_dir}/source/container-sidecar.json")"
[[ "${expected_size}" == "${CONTAINER_SIZE}" && "${expected_sha256}" == "${CONTAINER_SHA256}" ]]
download "${CONTAINER_KEY}" "${runtime_dir}/.sglang.sqsh.incoming"
[[ "$(stat -c '%s' "${runtime_dir}/.sglang.sqsh.incoming")" == "${CONTAINER_SIZE}" ]]
[[ "$(sha256sum "${runtime_dir}/.sglang.sqsh.incoming" | awk '{print $1}')" == "${CONTAINER_SHA256}" ]]
mv -- "${runtime_dir}/.sglang.sqsh.incoming" "${runtime_dir}/sglang.sqsh"

cp -a --no-preserve=ownership,xattr "${ENROOT_SYSCONF_PATH}/." "${runtime_dir}/enroot-system/"
cp -a --no-preserve=ownership,xattr "${ENROOT_LIBRARY_PATH}/." "${runtime_dir}/enroot-library/"
runtime_library="${runtime_dir}/enroot-library/runtime.sh"
[[ "$(grep -Fxc '    local timeout=100 pid=-1 id=0' "${runtime_library}")" -eq 1 ]]
sed -i 's/local timeout=100 pid=-1 id=0/local timeout=5000 pid=-1 id=0/' "${runtime_library}"
grep -Fqx '    local timeout=5000 pid=-1 id=0' "${runtime_library}"

nvidia_hook="${runtime_dir}/enroot-system/hooks.d/98-nvidia.sh"
if [[ -x "${nvidia_hook}" ]]; then sed -i 's%--ldconfig=@$(command -v ldconfig.real || command -v ldconfig)%--ldconfig=/sbin/ldconfig%' "${nvidia_hook}"; fi
if [[ "${CLUSTER_NAME}" == genkai && -e "${runtime_dir}/enroot-system/hooks.d/10-cgroups.sh" ]]; then chmod -x "${runtime_dir}/enroot-system/hooks.d/10-cgroups.sh"; fi

install -m 0755 "${GITHUB_WORKSPACE}/scripts/run-qwen-check-env.sh" "${runtime_dir}/scripts/run-qwen-check-env.sh"
jq -n --arg cluster "${CLUSTER_NAME}" --arg bucket "${CONTAINER_BUCKET}" --arg container_key "${CONTAINER_KEY}" --arg sidecar_key "${SIDECAR_KEY}" --arg sha256 "${CONTAINER_SHA256}" --argjson size_bytes "${CONTAINER_SIZE}" '{schema_version:"qwen-check-env-source/v1",cluster:$cluster,s3:{bucket:$bucket,container_key:$container_key,sidecar_key:$sidecar_key},container:{size_bytes:$size_bytes,sha256:$sha256},downloaded:true,enroot_direct_squashfs:true}' >"${runtime_dir}/source/check-env-source.json"
