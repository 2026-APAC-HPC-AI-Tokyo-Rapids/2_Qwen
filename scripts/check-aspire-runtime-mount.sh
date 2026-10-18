#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(realpath "${1:?runtime directory required}")"
model_path="${2:?model path required}"
preflight_dir="$(mktemp -d "${TMPDIR:-/tmp}/aspire-mount-preflight.XXXXXXXXXX")"
trap 'rm -rf -- "${preflight_dir:?}"' EXIT
export ENROOT_LIBRARY_PATH="${runtime_dir}/enroot-library" ENROOT_SYSCONF_PATH="${runtime_dir}/enroot-system" ENROOT_MOUNT_HOME=no
export ENROOT_CACHE_PATH="${preflight_dir}/cache" ENROOT_DATA_PATH="${preflight_dir}/data" ENROOT_RUNTIME_PATH="${preflight_dir}/runtime" ENROOT_TEMP_PATH="${preflight_dir}/temp"
install -d -m 0700 "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}" "${preflight_dir}/home"
expected="$(sha256sum "${runtime_dir}/source/prefill-expert-location.json" | cut -d ' ' -f 1)"
# Exercise the actual image and directory mounts before qsub, without loading weights or using GPUs.
timeout 180 enroot start -e NVIDIA_VISIBLE_DEVICES=none -e HOME=/runtime-home -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -m "${model_path}:/model" -m "${preflight_dir}/home:/runtime-home" -m "${runtime_dir}/draft:/draft" -m "${runtime_dir}:/runtime:ro" "${runtime_dir}/sglang.sqsh" /opt/sglang/venv/bin/python - "${expected}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

placement = Path("/runtime/source/prefill-expert-location.json").read_bytes()
assert hashlib.sha256(placement).hexdigest() == sys.argv[1], "expert map changed across the bind mount"
mapping = json.loads(placement)["physical_to_logical_map"]
model = json.loads(Path("/model/config.json").read_text())
assert len(mapping) == model["num_hidden_layers"], "expert map layer count differs from model"
assert all(sorted(layer) == list(range(model["num_experts"])) for layer in mapping), "expert map must permute every expert exactly once"
config = json.loads(Path("/runtime/source/aspire-serving.json").read_text())
assert model["num_experts"] % config["dp_size"] == 0
draft = json.loads(Path("/draft/config.json").read_text())
assert draft["architectures"] == ["LlamaForCausalLMEagle3"] and draft["torch_dtype"] == "bfloat16"
assert Path("/draft/model.safetensors").stat().st_size > 0
assert Path("/runtime-home").is_dir()
print(f"aspire_runtime_mount_ok layers={len(mapping)} experts={model['num_experts']} placement_sha256={sys.argv[1]} gpu_used=false", flush=True)
PY
