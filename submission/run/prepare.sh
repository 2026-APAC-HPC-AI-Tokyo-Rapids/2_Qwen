#!/usr/bin/env bash
# Prepare a run directory for the Bonus job on the Firmus login node (no GPU needed).
#
# usage: prepare.sh RUN_DIR IMAGE MODEL TOKENIZER_SNAPSHOT
#   RUN_DIR             new directory on storage both compute nodes can read
#   IMAGE               the SquashFS serving image recorded in config/serving.json
#   MODEL               Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 at MODEL_REVISION (all 24 shards)
#   TOKENIZER_SNAPSHOT  Qwen/Qwen3-235B-A22B-Instruct-2507 tokenizer files at TOKENIZER_REVISION
#
# Then: cd RUN_DIR && sbatch scripts/submit.sbatch
set -euo pipefail

MODEL_REVISION=e156cb4efae43fbee1a1ab073f946a1377e6b969
TOKENIZER_REPOSITORY=Qwen/Qwen3-235B-A22B-Instruct-2507
TOKENIZER_REVISION=ac9c66cc9b46af7306746a9250f23d47083d689e
TOKENIZER_FILES=(config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json)

package=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${package}/config/serving.json
run_dir=$(realpath -m "$1")
image=$(realpath "$2")
model=$(realpath "$3")
tokenizer=$(realpath "$4")

mkdir "${run_dir}"
install -d -m 0755 "${run_dir}/logs" "${run_dir}/results" "${run_dir}/scripts" "${run_dir}/source" "${run_dir}/home"
install -m 0644 "${config}" "${run_dir}/source/firmus-framework.json"

# ---- Image, model and tokenizer ----

sha256sum "${image}" >"${run_dir}/source/sglang-image.sha256"
ln -s "${image}" "${run_dir}/sglang.sqsh"
jq -n --arg model_path "${model}" --arg model_revision "${MODEL_REVISION}" --arg tokenizer_repository "${TOKENIZER_REPOSITORY}" --arg tokenizer_revision "${TOKENIZER_REVISION}" '{model:{repository:"Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",revision:$model_revision,path:$model_path,precision:"fp8",shards:24},tokenizer:{repository:$tokenizer_repository,revision:$tokenizer_revision}}' >"${run_dir}/source/model.json"

# AIPerf reads the tokenizer from an offline Hugging Face cache.
sha256sum "${tokenizer}"/* >"${run_dir}/source/tokenizer.sha256"
tokenizer_cache=${run_dir}/tokenizer-cache/hub/models--${TOKENIZER_REPOSITORY/\//--}
install -d -m 0755 "${tokenizer_cache}/refs" "${tokenizer_cache}/snapshots/${TOKENIZER_REVISION}"
for file in "${TOKENIZER_FILES[@]}"; do cp -L "${tokenizer}/${file}" "${tokenizer_cache}/snapshots/${TOKENIZER_REVISION}/${file}"; done
printf '%s' "${TOKENIZER_REVISION}" >"${tokenizer_cache}/refs/main"

# ---- Draft: download, then declare the SGLang architecture and dtype in its config ----

export HF_HUB_DISABLE_XET=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1
uvx --from 'huggingface_hub==0.36.0' hf download "$(jq -er .speculation.draft_repository "${config}")" config.json README.md model.safetensors --revision "$(jq -er .speculation.draft_revision "${config}")" --local-dir "${run_dir}/draft" --max-workers 1
python3 "${package}/run/prepare_eagle.py" "${run_dir}/draft" "${run_dir}/source"

# ---- Patched packages: copy each package out of the image and apply its numbered overlays in order ----

# stage_overlay NAME IMAGE_PATH PACKAGE_PATH
#   patches/overlay/NAME/NN-*.patch are relative to PACKAGE_PATH; submit.sbatch mounts the result over IMAGE_PATH.
stage_overlay() {
  local name=$1 image_path=$2 package_path=$3 patch
  local root=${run_dir}/overlay/${name} extract=${run_dir}/overlay/extract-${name}
  local patches=("${package}/patches/overlay/${name}"/[0-9][0-9]-*.patch)
  install -d -m 0755 "${run_dir}/overlay"
  unsquashfs -no-progress -no-xattrs -processors 1 -d "${extract}" "${run_dir}/sglang.sqsh" "${image_path}"
  chmod -R u+w "${extract}"
  install -d -m 0755 "$(dirname "${root}/${package_path}")" "${run_dir}/source/${name}-overlay"
  mv "${extract}/${image_path}" "${root}/${package_path}"
  rm -rf -- "${extract}"
  for patch in "${patches[@]}"; do
    (cd "${root}" && git apply "${patch}")
    cp "${patch}" "${run_dir}/source/${name}-overlay/"
  done
  # Record what was applied, in order, with hashes.
  for patch in "${patches[@]}"; do jq -n --arg path "${patch#"${package}/"}" --arg sha "$(sha256sum "${patch}" | awk '{print $1}')" '{path:$path,sha256:$sha}'; done | jq -s --arg image "$(jq -er .images.sglang.sha256 "${config}")" --arg mount "/${image_path}" '{base_image_sha256:$image,mount:$mount,patches:.}' >"${run_dir}/source/${name}-overlay.json"
}
stage_overlay sglang opt/sglang/source/python/sglang python/sglang
stage_overlay deep-gemm opt/sglang/venv/lib/python3.12/site-packages/deep_gemm deep_gemm

# ---- Job inputs and scripts ----

# Kernel choices adopted by the in-job gates of Actions run 41767886175, and the GPU clock table worker.sh
# uses to place each Prefill worker's last pipeline stage.
install -m 0644 "${package}/config/prefill-kernels.env" "${package}/config/decode-kernels.env" "${package}/config/firmus-gpu-sm-clock.json" "${run_dir}/source/"
install -m 0755 "${package}/run/submit.sbatch" "${package}/run/worker.sh" "${package}/run/official-aiperf.sh" "${run_dir}/scripts/"
install -m 0644 "${package}/run/parse-aiperf-result.py" "${package}/run/order-firmus-pp4-gpus.py" "${package}/run/greedy-probe.py" "${run_dir}/scripts/"
# Keep the exact scripts this run used with its records.
cp "${package}/run/prepare.sh" "${package}/run/prepare_eagle.py" "${run_dir}/scripts/"* "${run_dir}/source/"

printf 'Prepared %s\n' "${run_dir}"
printf 'cd %q && sbatch scripts/submit.sbatch\n' "${run_dir}"
