"""Make the downloaded EAGLE3 draft loadable by SGLang and record its provenance.

The pinned draft (nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507) declares LlamaForCausalLM. SGLang loads it with
the LlamaForCausalLMEagle3 architecture, and the dtype is declared as bfloat16. The weights are not changed.

usage: prepare_eagle.py DRAFT_DIR RECORD_DIR
"""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    draft, records = map(Path, sys.argv[1:])
    config_path = draft / "config.json"
    weights = draft / "model.safetensors"

    original = config_path.read_bytes()
    config = json.loads(original)
    config["architectures"] = ["LlamaForCausalLMEagle3"]
    config["torch_dtype"] = "bfloat16"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    # The safetensors header: an 8-byte little-endian length, then the tensor table as JSON.
    with weights.open("rb") as stream:
        header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
    tensors = {name: tensor for name, tensor in header.items() if name != "__metadata__"}

    records.mkdir(parents=True, exist_ok=True)
    (records / "draft-original-config.json").write_bytes(original)
    (records / "draft-runtime-config.json").write_bytes(config_path.read_bytes())
    (records / "draft-README.md").write_bytes((draft / "README.md").read_bytes())
    provenance = {
        "repository": "nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507",
        "revision": "ff0f0c0999f6c86b21dfd2f5803eebc4ac712224",
        "license": "CC-BY-4.0",
        "weight_sha256": sha256(weights),
        "weight_bytes": weights.stat().st_size,
        "precision": "BF16",
        "parameter_count": sum(math.prod(tensor["shape"]) for tensor in tensors.values() if tensor["dtype"] == "BF16"),
        "tensors": tensors,
        "adaptation": "Native SGLang EAGLE3 architecture selector and explicit BF16 dtype only; weights unchanged",
        "original_config_sha256": hashlib.sha256(original).hexdigest(),
        "runtime_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    (records / "draft-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
