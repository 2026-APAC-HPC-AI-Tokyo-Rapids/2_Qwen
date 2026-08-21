#!/usr/bin/env bash
set -euo pipefail

: "${CLUSTER_NAME:?cluster name is required}"
: "${CONTAINER_ARCHIVE:?container archive is required}"
: "${CONTAINER_SIDECAR:?container sidecar is required}"
: "${ENROOT_SYSCONF_PATH:?Enroot system configuration path is required}"
: "${GITHUB_SHA:?workflow commit is required}"
: "${GITHUB_WORKSPACE:?workflow workspace is required}"
: "${GPUS_PER_NODE:?GPU count per node is required}"
: "${MODEL_CACHE_ROOT:?model metadata cache root is required}"
: "${NODE_COUNT:?node count is required}"
: "${PROVISION_OUTPUT:?GitHub output file is required}"
: "${RESOURCE_GROUP:?scheduler resource group is required}"
: "${RUNNER_TEMP:?GitHub runner temporary directory is required}"
: "${RUN_PURPOSE:?run purpose is required}"
: "${RUNTIME_ROOT:?runtime root is required}"

export UV_CACHE_DIR="${RUNNER_TEMP}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${RUNNER_TEMP}/uv-python"
install -d -m 0700 "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

model_repository=Qwen/Qwen3.5-397B-A17B-FP8
model_revision="$(curl --fail --location --silent --show-error "https://huggingface.co/api/models/${model_repository}" | jq -er '.sha | select(test("^[0-9a-f]{40,64}$"))')"
model_path="${MODEL_CACHE_ROOT}/${model_revision}"
install -d -m 2775 "${model_path}" "${RUNTIME_ROOT}"
uvx --from 'huggingface_hub>=1.0' hf download "${model_repository}" --revision "${model_revision}" --include '*.json' --include '*.txt' --include '*.model' --include '*.tiktoken' --exclude '*.safetensors' --exclude '*.bin' --local-dir "${model_path}"
test -s "${model_path}/config.json"
test -s "${model_path}/tokenizer_config.json"

expected_size="$(jq -er 'select(.schema_version == "container-sidecar/v1") | .container_artifact.size_bytes' "${CONTAINER_SIDECAR}")"
expected_sha256="$(jq -er '.container_artifact.sha256' "${CONTAINER_SIDECAR}")"
test "$(stat -c '%s' "${CONTAINER_ARCHIVE}")" = "${expected_size}"
test "$(sha256sum "${CONTAINER_ARCHIVE}" | awk '{print $1}')" = "${expected_sha256}"

runtime_dir="$(mktemp -d "${RUNTIME_ROOT}/run-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.XXXXXXXXXX")"
printf 'runtime_dir=%s\n' "${runtime_dir}" >>"${PROVISION_OUTPUT}"
mkdir -p "${runtime_dir}/control" "${runtime_dir}/domain/fragments" "${runtime_dir}/enroot-system" "${runtime_dir}/logs" "${runtime_dir}/profile/raw" "${runtime_dir}/results" "${runtime_dir}/scripts" "${runtime_dir}/source"
ln -s "${CONTAINER_ARCHIVE}" "${runtime_dir}/sglang.sqsh"
install -m 0644 "${CONTAINER_SIDECAR}" "${runtime_dir}/source/container-sidecar.json"
install -m 0755 "${GITHUB_WORKSPACE}/scripts/run-qwen35-job.sh" "${GITHUB_WORKSPACE}/scripts/run-qwen35-node.sh" "${GITHUB_WORKSPACE}/scripts/exec-profiler.sh" "${GITHUB_WORKSPACE}/scripts/qwen-profile-wrapper.py" "${GITHUB_WORKSPACE}/scripts/send-qwen-fragments.sh" "${GITHUB_WORKSPACE}/scripts/package-qwen-runtime.sh" "${runtime_dir}/scripts/"
cp -a --no-preserve=ownership,xattr "${GITHUB_WORKSPACE}/qwen-experiment-metrics" "${runtime_dir}/qwen-experiment-metrics"
cp -a --no-preserve=ownership,xattr "${ENROOT_SYSCONF_PATH}/." "${runtime_dir}/enroot-system/"
nvidia_hook="${runtime_dir}/enroot-system/hooks.d/98-nvidia.sh"
test -x "${nvidia_hook}"
sed -i 's%--ldconfig=@$(command -v ldconfig.real || command -v ldconfig)%--ldconfig=/sbin/ldconfig%' "${nvidia_hook}"
grep -Fq -- '--ldconfig=/sbin/ldconfig' "${nvidia_hook}"
if [[ "${CLUSTER_NAME}" == genkai && -e "${runtime_dir}/enroot-system/hooks.d/10-cgroups.sh" ]]; then chmod -x "${runtime_dir}/enroot-system/hooks.d/10-cgroups.sh"; fi
mapfile -t runtime_source_oids < <(ENROOT_SYSCONF_PATH="${runtime_dir}/enroot-system" enroot start -e NVIDIA_VISIBLE_DEVICES=void -e MELLANOX_VISIBLE_DEVICES=void "${CONTAINER_ARCHIVE}" cat /opt/sglang/offline-throughput-source-blob-oid.txt /opt/sglang/server-args-source-blob-oid.txt)
[[ "${#runtime_source_oids[@]}" -eq 2 ]]
[[ "${runtime_source_oids[0]}" =~ ^[0-9a-f]{40,64}$ && "${runtime_source_oids[1]}" =~ ^[0-9a-f]{40,64}$ ]]
jq -n --arg bench_args_blob_oid "${runtime_source_oids[0]}" --arg server_args_blob_oid "${runtime_source_oids[1]}" '{schema_version:"qwen-runtime-source-oids/v1",bench_args_blob_oid:$bench_args_blob_oid,server_args_blob_oid:$server_args_blob_oid}' >"${runtime_dir}/source/runtime-source-oids.json"
export UV_PYTHON_INSTALL_DIR="${runtime_dir}/qwen-experiment-metrics/.uv-python"
install -d -m 0700 "${UV_PYTHON_INSTALL_DIR}"
uv sync --project "${runtime_dir}/qwen-experiment-metrics" --managed-python --frozen --no-dev --extra publish

tensor_parallel_size="$(( NODE_COUNT * GPUS_PER_NODE ))"
jq -n --arg purpose "${RUN_PURPOSE}" --argjson nodes "${NODE_COUNT}" --argjson gpus_per_node "${GPUS_PER_NODE}" --argjson tensor_parallel_size "${tensor_parallel_size}" --arg resource_group "${RESOURCE_GROUP}" --arg transport_policy "${CLUSTER_NAME}-host-mpi-enroot" '{schema_version:"qwen-execution-settings/v1",run_purpose:$purpose,nodes:$nodes,mpi_processes:$nodes,gpus_per_node:$gpus_per_node,tensor_parallel_size:$tensor_parallel_size,resource_group:$resource_group,transport_policy:$transport_policy,advertise_source:"mpi-rank-zero-hostname",sglang_port:25000,node_agent_port:25001,rendezvous_timeout_sec:300,ready_timeout_sec:300}' >"${runtime_dir}/control/execution-settings-input.json"
jq -n --arg repository "${model_repository}" --arg revision "${model_revision}" --arg path "${model_path}" '{schema_version:"model-metadata/v1",repository:$repository,revision:$revision,path:$path,weights_present:false,load_format:"dummy"}' >"${runtime_dir}/source/model-metadata.json"
dataset_fixture="${runtime_dir}/source/dummy-sharegpt.json"
jq -n '[range(8) as $index | {conversations:[{from:"human",value:("deterministic dummy prompt " + ($index | tostring))},{from:"gpt",value:"deterministic dummy response"}]}]' >"${dataset_fixture}"
jq -e 'type == "array" and length == 8 and all(.[]; (.conversations | type == "array" and length == 2 and all(.[]; (.value | type == "string" and length > 0))))' "${dataset_fixture}" >/dev/null
dataset_fixture_sha256="$(sha256sum "${dataset_fixture}" | awk '{print $1}')"
jq -n --arg fixture_sha256 "${dataset_fixture_sha256}" '{schema_version:"dummy-random-dataset/v1",dataset_name:"random",seed:42,num_prompts:8,input_length:1024,output_length:128,range_ratio:1,source:{kind:"generated_sharegpt_fixture",relative_path:"source/dummy-sharegpt.json",sha256:$fixture_sha256}}' >"${runtime_dir}/source/dataset-manifest.json"

object_hash_algorithm="$(git -C "${GITHUB_WORKSPACE}" rev-parse --show-object-format)"
base_commit="$(git -C "${GITHUB_WORKSPACE}" rev-parse "${GITHUB_SHA}^" 2>/dev/null || printf '%s' "${GITHUB_SHA}")"
control_plane_sha256="$(git -C "${GITHUB_WORKSPACE}" ls-tree -r --full-tree "${GITHUB_SHA}" -- .github/workflows scripts platforms containers | LC_ALL=C sort | sha256sum | awk '{print $1}')"
instrumentation_bundle_sha256="$(git -C "${GITHUB_WORKSPACE}" ls-tree -r --full-tree "${GITHUB_SHA}" -- qwen-experiment-metrics patches | LC_ALL=C sort | sha256sum | awk '{print $1}')"
if [[ "${RUN_PURPOSE}" == profile ]]; then base_role=profile_base; else base_role=score_reference; fi
"${runtime_dir}/qwen-experiment-metrics/.venv/bin/qwen-experiment-metrics" init-run --runtime-dir "${runtime_dir}" --purpose "${RUN_PURPOSE}" --object-hash-algorithm "${object_hash_algorithm}" --base-role "${base_role}" --base-commit "${base_commit}" --head-commit "${GITHUB_SHA}" --control-plane-sha256 "${control_plane_sha256}" --instrumentation-bundle-sha256 "${instrumentation_bundle_sha256}" --execution-settings "${runtime_dir}/control/execution-settings-input.json" --repository "${GITHUB_WORKSPACE}"
"${runtime_dir}/qwen-experiment-metrics/.venv/bin/qwen-experiment-metrics" record-provenance --manifest "${runtime_dir}/control/run-manifest.json" --runtime-dir "${runtime_dir}"
run_id="$(jq -er '.run_id' "${runtime_dir}/control/run-manifest.json")"
printf 'run_id=%s\nmodel_path=%s\n' "${run_id}" "${model_path}" >>"${PROVISION_OUTPUT}"
