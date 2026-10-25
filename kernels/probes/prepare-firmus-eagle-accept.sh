#!/usr/bin/env bash
# Login-node only: pinned downloads, config adaptation, diagnostic source overlay.
set -euo pipefail
: "${RUNTIME_DIR:?}" "${GITHUB_WORKSPACE:?}" "${GITHUB_SHA:?}"
config="${GITHUB_WORKSPACE}/configs/firmus-framework.json"
model=/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8
[[ "$(head -n 1 "${model}/.cache/huggingface/download/config.json.metadata")" == e156cb4efae43fbee1a1ab073f946a1377e6b969 ]]
jq -e '.model_type == "qwen3_moe" and .quantization_config.quant_method == "fp8" and .quantization_config.fmt == "e4m3"' "${model}/config.json" >/dev/null
[[ "$(jq -r '[.weight_map[]] | unique | length' "${model}/model.safetensors.index.json")" == 24 ]]
while read -r shard; do test -s "${model}/${shard}"; done < <(jq -r '[.weight_map[]] | unique[]' "${model}/model.safetensors.index.json")
jq -e '.speculation.algorithm == "EAGLE3" and .speculation.steps == 3 and .speculation.draft_tokens == 4 and .speculation.draft_repository == "nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507" and .speculation.draft_revision == "ff0f0c0999f6c86b21dfd2f5803eebc4ac712224"' "${config}" >/dev/null
install -d "${RUNTIME_DIR}/source" "${RUNTIME_DIR}/logs" "${RUNTIME_DIR}/results"
export HF_HUB_DISABLE_XET=1 HF_HOME="${RUNTIME_DIR}/tokenizer-cache" UV_CACHE_DIR="${RUNTIME_DIR}/uv-cache"
export UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1
uvx --from 'huggingface_hub==0.36.0' hf download "$(jq -er .speculation.draft_repository "${config}")" config.json README.md model.safetensors --revision "$(jq -er .speculation.draft_revision "${config}")" --local-dir "${RUNTIME_DIR}/draft" --max-workers 1
uv run --no-project --python 3.12 "${GITHUB_WORKSPACE}/research/simulation/prepare_eagle.py" "${RUNTIME_DIR}/draft" "${RUNTIME_DIR}/source"
jq -e '.weight_sha256 == "bf226a966868fe413a22e62e471aa1130bd94f17fd2c40d322b1c8b4def1ae86" and .weight_bytes == 1145943008 and .precision == "BF16" and .runtime_config_sha256 == "85dc5a49b8644d070c396b5b993a4ec9d0ec19aaf41dfa5025500c0f5578329c"' "${RUNTIME_DIR}/source/draft-provenance.json" >/dev/null
tokenizer_revision=ac9c66cc9b46af7306746a9250f23d47083d689e
uvx --from 'huggingface_hub==0.36.0' hf download Qwen/Qwen3-235B-A22B-Instruct-2507 config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json --max-workers 1 --revision "${tokenizer_revision}"
tokenizer_cache="${HF_HOME}/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507"
for file in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do test -s "${tokenizer_cache}/snapshots/${tokenizer_revision}/${file}"; done
install -d "${tokenizer_cache}/refs"
printf '%s' "${tokenizer_revision}" >"${tokenizer_cache}/refs/main"
image=$(jq -er .images.sglang.path "${config}")
unsquashfs -no-progress -no-xattrs -processors 1 -d "${RUNTIME_DIR}/diagnostic" "${image}" opt/sglang/source/python/sglang/srt/managers/scheduler_components/batch_result_processor.py >"${RUNTIME_DIR}/logs/extract-diagnostic.log" 2>&1
python3 "${GITHUB_WORKSPACE}/kernels/probes/instrument-eagle-accept.py" "${RUNTIME_DIR}/diagnostic/opt/sglang/source/python/sglang/srt/managers/scheduler_components/batch_result_processor.py"
install -m 0644 "${GITHUB_WORKSPACE}/kernels/probes/eagle-accept.py" "${GITHUB_WORKSPACE}/kernels/probes/eagle-english-prompts.json" "${RUNTIME_DIR}/source/"
install -m 0644 "${GITHUB_WORKSPACE}/scripts/collect-eagle-acceptance.py" "${RUNTIME_DIR}/source/eagle_metrics.py"
install -m 0755 "${GITHUB_WORKSPACE}/kernels/probes/run-firmus-eagle-accept.sh" "${RUNTIME_DIR}/source/probe"
jq -n --arg commit "${GITHUB_SHA}" --slurpfile config "${config}" '{schema:"firmus-eagle-accept/v1",calibration:true,commit:$commit,image:$config[0].images.sglang,speculation:$config[0].speculation,model_revision:"e156cb4efae43fbee1a1ab073f946a1377e6b969",tokenizer_revision:"ac9c66cc9b46af7306746a9250f23d47083d689e",scope:"single stock TP4 server; no PD, no MegaMoE, no CUDA graphs; acceptance diagnostic only",timing:{startup_limit_s:600,synthetic_setup_measure_grace_s:95,english_limit_s:180,cleanup_s:15,process_limit_s:960,slurm_limit_s:1020},startup_reference:{run:38030890137,target_load_s:[27.41,28.49],draft_load_s:0.35,server_start_to_post_graph_s:172},draft_kv_bf16:{enabled:false,reason:"No independent EAGLE draft KV dtype CLI in pinned image; fa4 compute-dtype override is DFLASH-only; source adaptation would require a separate server run"}}' >"${RUNTIME_DIR}/source/provenance.json"
