#!/usr/bin/env bash
set -euo pipefail
mode="${1:-throughput}"
root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
source "${root}/configs/default.env"
if [[ -n "${EXPERIMENT_FILE:-}" ]]; then
  test -f "$EXPERIMENT_FILE" || { echo "Missing experiment file: $EXPERIMENT_FILE" >&2; exit 2; }
  source "$EXPERIMENT_FILE"
fi
: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen 3.5 checkpoint/config directory}"
: "${LOAD_FORMAT:=dummy}" "${NUM_PROMPTS:=128}" "${RANDOM_INPUT_LEN:=1024}"
: "${RANDOM_OUTPUT_LEN:=256}" "${TP_SIZE:=1}" "${TRIALS:=3}"
: "${MEM_FRACTION_STATIC:=0.85}" "${PROFILE_STEPS:=5}" "${SEED:=42}"
test -f "${MODEL_PATH}/config.json" || { echo "Missing ${MODEL_PATH}/config.json" >&2; exit 2; }
python - "$MODEL_PATH/config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
model_type = str(config.get("model_type", ""))
architectures = " ".join(config.get("architectures") or [])
if not (model_type.startswith("qwen3_5") and "Qwen3_5" in architectures):
    raise SystemExit(f"Only Qwen 3.5 is supported: model_type={model_type!r}, architectures={architectures!r}")
PY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
sha="$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo nogit)"
out="${RESULTS_ROOT:-${root}/results}/${stamp}-${sha}-${mode}"
mkdir -p "$out"
common=(python -m sglang.benchmark.offline_throughput
  --model-path "$MODEL_PATH" --load-format "$LOAD_FORMAT" --tp-size "$TP_SIZE"
  --dataset-name random --random-input-len "$RANDOM_INPUT_LEN"
  --random-output-len "$RANDOM_OUTPUT_LEN" --seed "$SEED"
  --mem-fraction-static "$MEM_FRACTION_STATIC" --result-filename "$out/results.jsonl")
if [[ -n "${EXTRA_ARGS:-}" ]]; then read -r -a extra <<<"$EXTRA_ARGS"; common+=("${extra[@]}"); fi
{
  echo "mode=$mode"; echo "git_sha=$(git -C "$root" rev-parse HEAD 2>/dev/null || true)"
  echo "git_dirty=$(test -n "$(git -C "$root" status --porcelain 2>/dev/null)" && echo true || echo false)"
  echo "model_path=$MODEL_PATH"; echo "load_format=$LOAD_FORMAT"; printf 'command='; printf '%q ' "${common[@]}"; echo
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv || true
  python -c 'import platform, sglang, torch; print("platform="+platform.platform()); print("sglang="+getattr(sglang,"__version__","unknown")); print("torch="+torch.__version__)'
} >"$out/metadata.txt" 2>&1
case "$mode" in
  smoke) "${common[@]}" --num-prompts 2 2>&1 | tee "$out/smoke.log" ;;
  throughput)
    # SGLang performs its own warmup inside each offline benchmark invocation.
    for ((i=1; i<=TRIALS; i++)); do "${common[@]}" --num-prompts "$NUM_PROMPTS" 2>&1 | tee "$out/trial-${i}.log"; done ;;
  torch-profile)
    export SGLANG_TORCH_PROFILER_DIR="$out/torch-profiler"; mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
    "${common[@]}" --num-prompts "$NUM_PROMPTS" --profile --profile-steps "$PROFILE_STEPS" 2>&1 | tee "$out/profile.log" ;;
  nsys)
    command -v nsys >/dev/null || { echo "nsys is not available" >&2; exit 3; }
    nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true -o "$out/nsys" "${common[@]}" --num-prompts "$NUM_PROMPTS" 2>&1 | tee "$out/nsys.log" ;;
  *) echo "Unknown mode: $mode (smoke|throughput|torch-profile|nsys)" >&2; exit 2 ;;
esac
echo "Results: $out"
