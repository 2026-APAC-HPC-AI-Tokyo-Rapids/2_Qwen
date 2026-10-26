#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?}"
[[ "${GPU_COUNT:-}" == 4 ]]
started=$(date +%s)
export ENROOT_LIBRARY_PATH="${RUNTIME_DIR}/enroot-library" ENROOT_SYSCONF_PATH="${RUNTIME_DIR}/enroot-system" ENROOT_MOUNT_HOME=no
node_tmp=$(mktemp -d "${TMPDIR:-/tmp}/eagle-accept.XXXXXXXX")
trap 'chmod -R u+w "${node_tmp:?}"; rm -rf -- "${node_tmp:?}"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export ENROOT_CACHE_PATH="${node_tmp}/cache" ENROOT_DATA_PATH="${node_tmp}/data" ENROOT_RUNTIME_PATH="${node_tmp}/runtime" ENROOT_TEMP_PATH="${node_tmp}/temp"
install -d "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}" "${node_tmp}/jit" "${RUNTIME_DIR}/home"
printf 'budget_s startup=450 aiperf_setup=45 measurement=20 grace=30 collection_shutdown=15 total=560 timeout=570 pjm=600\n'
enroot start -m "${RUNTIME_DIR}:${RUNTIME_DIR}" -m "${RUNTIME_DIR}/home:${HOME}" -m /fast/PROJECT/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8:/model:ro -m "${node_tmp}/jit:/probe-cache" -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_DISABLE_REQUIRE=1 -e PYTHONPATH= -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e "HF_HOME=${RUNTIME_DIR}/tokenizer-cache" -e TRITON_CACHE_DIR=/probe-cache/triton -e TORCHINDUCTOR_CACHE_DIR=/probe-cache/inductor -e TVM_FFI_CACHE_DIR=/probe-cache/tvm -e SGLANG_DG_CACHE_DIR=/probe-cache/deep-gemm -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 "${RUNTIME_DIR}/sglang.sqsh" /opt/sglang/venv/bin/python -u "${RUNTIME_DIR}/source/eagle-accept.py" "${RUNTIME_DIR}" "${started}"
