#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(realpath "${1:?runtime directory required}")"
model_path="${2:?model path required}"
preflight_dir="$(mktemp -d "${TMPDIR:-/tmp}/aspire-mount-preflight.XXXXXXXXXX")"
trap 'rm -rf -- "${preflight_dir:?}"' EXIT
export ENROOT_LIBRARY_PATH="${runtime_dir}/enroot-library" ENROOT_SYSCONF_PATH="${runtime_dir}/enroot-system" ENROOT_MOUNT_HOME=no
# `none` still loads NVML; the login CPU has no NVIDIA driver (Actions 39097518209).
export NVIDIA_VISIBLE_DEVICES=void
export ENROOT_CACHE_PATH="${preflight_dir}/cache" ENROOT_DATA_PATH="${preflight_dir}/data" ENROOT_RUNTIME_PATH="${preflight_dir}/runtime" ENROOT_TEMP_PATH="${preflight_dir}/temp"
install -d -m 0700 "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}" "${preflight_dir}/home"
expected="$(sha256sum "${runtime_dir}/source/prefill-expert-location.json" | cut -d ' ' -f 1)"
printf 'aspire_mount_preflight_host=%s enroot_version=%s\n' "$(hostname -s)" "$(enroot version)"
overlay_mounts=(-m "${runtime_dir}/overlay/sglang/python/sglang:/opt/sglang/source/python/sglang:none:bind,ro,x-create=dir" -m "${runtime_dir}/overlay/deep-gemm/deep_gemm:/opt/sglang/venv/lib/python3.12/site-packages/deep_gemm:none:bind,ro,x-create=dir")
# Exercise the actual image and directory mounts before qsub, without loading weights or using GPUs.
timeout 180 enroot start -e NVIDIA_VISIBLE_DEVICES=void -e HOME=/runtime-home -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -m "${model_path}:/model" -m "${preflight_dir}/home:/runtime-home" -m "${runtime_dir}/draft:/draft" -m "${runtime_dir}:/runtime:none:bind,ro,x-create=dir" "${overlay_mounts[@]}" "${runtime_dir}/sglang.sqsh" /opt/sglang/venv/bin/python - "${expected}" <<'PY'
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path

placement = Path("/runtime/source/prefill-expert-location.json").read_bytes()
assert hashlib.sha256(placement).hexdigest() == sys.argv[1], "expert map changed across the bind mount"
mapping = json.loads(placement)["physical_to_logical_map"]
model = json.loads(Path("/model/config.json").read_text())
assert len(mapping) == model["num_hidden_layers"], "expert map layer count differs from model"
config = json.loads(Path("/runtime/source/aspire-serving.json").read_text())
redundant = config["prefill_ep_num_redundant_experts"]
assert redundant == 12 and config["dp_size"] == 4, "replica overlay requires hot-4 on EP4"
for layer in mapping:
    assert len(layer) == model["num_experts"] + redundant, "physical expert count differs from configuration"
    counts = Counter(layer)
    assert sorted(counts) == list(range(model["num_experts"])), "missing or invalid logical expert"
    assert sorted(counts.values()) == [1] * (model["num_experts"] - 4) + [4] * 4, "expected exactly four hot replicas per rank"
    width = len(layer) // config["dp_size"]
    hot = {expert for expert, count in counts.items() if count == 4}
    for rank in range(config["dp_size"]):
        local = layer[rank * width:(rank + 1) * width]
        assert len(set(local)) == width and hot <= set(local), "each rank must own one copy of every hot expert"
assert model["num_experts"] % config["dp_size"] == 0
for name, prefix in (("sglang", "python/sglang"), ("deep-gemm", "deep_gemm")):
    record = json.loads(Path(f"/runtime/source/{name}-overlay.json").read_text())
    assert record["base_image_sha256"] == config["container_sha256"]
    assert [patch["path"] for patch in record["patches"]] == config[f"{name.replace('-', '_')}_overlay_patches"]
    for patch in record["patches"]:
        content = Path(f"/runtime/source/{name}-overlay/{Path(patch['path']).name}").read_bytes()
        assert hashlib.sha256(content).hexdigest() == patch["sha256"]
        for line in content.decode().splitlines():
            if line.startswith(f"+++ b/{prefix}/"):
                relative = line.removeprefix(f"+++ b/{prefix}/")
                assert Path(f"/runtime/overlay/{name}/{prefix}/{relative}").read_bytes() == Path(f"{record['mount']}/{relative}").read_bytes(), "patched package differs across bind mount"
draft = json.loads(Path("/draft/config.json").read_text())
assert draft["architectures"] == ["LlamaForCausalLMEagle3"] and draft["torch_dtype"] == "bfloat16"
assert Path("/draft/model.safetensors").stat().st_size > 0
assert Path("/runtime-home").is_dir()
print(f"aspire_runtime_mount_ok layers={len(mapping)} experts={model['num_experts']} placement_sha256={sys.argv[1]} gpu_used=false", flush=True)
PY
