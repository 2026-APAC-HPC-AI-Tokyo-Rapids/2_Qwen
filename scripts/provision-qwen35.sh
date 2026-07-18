#!/usr/bin/env bash
# Validate and warm the non-GPU parts of the Qwen 3.5 benchmark runtime.
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATASET_PATH:?DATASET_PATH is required}"
: "${SIF_ARCHIVE_DIR:?SIF_ARCHIVE_DIR is required}"
: "${GITHUB_RUN_NUMBER:?GITHUB_RUN_NUMBER is required}"

sif_path="${SIF_ARCHIVE_DIR}/${GITHUB_RUN_NUMBER}.sif"
jit_cache="${HOME}/.cache/sglang-jit"

module purge
module load python/3.12.11 singularity-ce/4.1.3
python - "${MODEL_PATH}/config.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
model_type = str(config.get("model_type", ""))
architectures = " ".join(config.get("architectures") or [])
if not (model_type.startswith("qwen3_5") and "Qwen3_5" in architectures):
    raise SystemExit(
        "Not a Qwen 3.5 checkpoint: {!r}, {!r}".format(model_type, architectures)
    )
PY

singularity exec --containall --cleanenv "${sif_path}" /opt/sglang/venv/bin/python -c 'import sglang; print(sglang.__file__)'

# Complete model-dependent initialization that does not require a GPU before
# entering the measured compute job.
mkdir -p "${jit_cache}/deep-gemm" "${jit_cache}/torchinductor" "${jit_cache}/triton"
singularity exec --containall --cleanenv \
  --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_DATASETS_OFFLINE=1 \
  --env SGLANG_DG_CACHE_DIR=/jit-cache/deep-gemm,TORCHINDUCTOR_CACHE_DIR=/jit-cache/torchinductor,TRITON_CACHE_DIR=/jit-cache/triton \
  --bind "${MODEL_PATH}:/model:ro,${DATASET_PATH}:/dataset/ShareGPT_V3_unfiltered_cleaned_split.json:ro,${jit_cache}:/jit-cache" \
  "${sif_path}" /opt/sglang/venv/bin/python - <<'PY'
import random

from sglang.benchmark.utils import get_tokenizer
from sglang.benchmark.datasets.sharegpt import sample_sharegpt_requests
from sglang.srt.configs.model_config import ModelConfig
import sglang.benchmark.offline_throughput

model_config = ModelConfig("/model")
tokenizer = get_tokenizer("/model")
tokenizer("SGLang provision check")
random.seed(42)
requests = sample_sharegpt_requests(
    "/dataset/ShareGPT_V3_unfiltered_cleaned_split.json",
    32,
    tokenizer,
    fixed_output_len=256,
    context_len=262144,
)
if len(requests) != 32:
    raise SystemExit(f"ShareGPT yielded only {len(requests)} requests")
print(model_config.hf_config.model_type, type(tokenizer).__name__, len(requests))
PY
