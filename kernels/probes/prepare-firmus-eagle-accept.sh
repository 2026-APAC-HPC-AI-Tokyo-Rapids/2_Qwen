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
install -m 0644 "${GITHUB_WORKSPACE}/kernels/probes/eagle-drafts.json" "${RUNTIME_DIR}/source/eagle-drafts.json"
while IFS=$'\t' read -r name repository revision; do
    install -d "${RUNTIME_DIR}/drafts/${name}" "${RUNTIME_DIR}/results/${name}" "${RUNTIME_DIR}/logs/${name}"
    uvx --from 'huggingface_hub==0.36.0' hf download "${repository}" config.json README.md model.safetensors --revision "${revision}" --local-dir "${RUNTIME_DIR}/drafts/${name}" --max-workers 1
    uv run --no-project --python 3.12 "${GITHUB_WORKSPACE}/kernels/probes/prepare-eagle-drafts.py" "${RUNTIME_DIR}/drafts/${name}" "${RUNTIME_DIR}/results/${name}" "${RUNTIME_DIR}/source/eagle-drafts.json" "${name}"
done < <(jq -r '.[] | select(.enabled) | [.name,.repository,.revision] | @tsv' "${RUNTIME_DIR}/source/eagle-drafts.json")
install -d "${RUNTIME_DIR}/results/redhat"
jq '.[] | select(.name == "redhat") | {status:"excluded",repository,revision,reason,accept_length_including_bonus:null,accepted_draft_length:null,draft_precision:null,verify_steps:null}' "${RUNTIME_DIR}/source/eagle-drafts.json" >"${RUNTIME_DIR}/results/redhat/eagle-acceptance.json"
tokenizer_revision=ac9c66cc9b46af7306746a9250f23d47083d689e
uvx --from 'huggingface_hub==0.36.0' hf download Qwen/Qwen3-235B-A22B-Instruct-2507 config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json --max-workers 1 --revision "${tokenizer_revision}"
tokenizer_cache="${HF_HOME}/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507"
for file in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do test -s "${tokenizer_cache}/snapshots/${tokenizer_revision}/${file}"; done
install -d "${tokenizer_cache}/refs"
printf '%s' "${tokenizer_revision}" >"${tokenizer_cache}/refs/main"
image=$(jq -er .images.sglang.path "${config}")
unsquashfs -no-progress -no-xattrs -processors 1 -d "${RUNTIME_DIR}/diagnostic" "${image}" opt/sglang/source/python/sglang/srt/managers/scheduler_components/batch_result_processor.py >"${RUNTIME_DIR}/logs/extract-diagnostic.log" 2>&1
python3 "${GITHUB_WORKSPACE}/kernels/probes/instrument-eagle-accept.py" "${RUNTIME_DIR}/diagnostic/opt/sglang/source/python/sglang/srt/managers/scheduler_components/batch_result_processor.py"
install -m 0644 "${GITHUB_WORKSPACE}/kernels/probes/eagle-accept.py" "${RUNTIME_DIR}/source/"
install -m 0644 "${GITHUB_WORKSPACE}/scripts/collect-eagle-acceptance.py" "${RUNTIME_DIR}/source/eagle_metrics.py"
install -m 0755 "${GITHUB_WORKSPACE}/kernels/probes/run-firmus-eagle-accept.sh" "${RUNTIME_DIR}/source/probe"
jq -n --arg commit "${GITHUB_SHA}" --slurpfile config "${config}" --slurpfile drafts "${RUNTIME_DIR}/source/eagle-drafts.json" '{schema:"firmus-eagle-drafts/v1",calibration:true,commit:$commit,image:$config[0].images.sglang,speculation:$config[0].speculation,drafts:$drafts[0],model_revision:"e156cb4efae43fbee1a1ab073f946a1377e6b969",tokenizer_revision:"ac9c66cc9b46af7306746a9250f23d47083d689e",scope:"two stock TP4 servers in parallel; no PD, no MegaMoE, no CUDA graphs; FP8 target/draft/KV; acceptance diagnostic only",timing:{startup_limit_s:600,client_duration_limit_s:180,client_grace_s:30,worker_cleanup_at_s:900,parent_cleanup_at_s:915,process_limit_s:960,slurm_limit_s:1020},startup_reference:{run:40621030371,target_load_s:[19.38,21.15],draft_load_s:[0.95,0.96],server_start_to_graph_begin_s:60,target_graph_s:99.7,draft_graph_s:24.9,ready_s:194,graph_free_subtraction_estimate_s:69.4,prefill_graph_disabled_ready_s:165,prefill_pd_warmup_s:78,client_setup_s:49.4,client_export_s:16.1}}' >"${RUNTIME_DIR}/source/provenance.json"
