"""Install explicit Qwen3 MegaMoE backend into exact pinned vLLM source."""

import ast
import hashlib
import json
import sys
from pathlib import Path

PATCHES = {
    "config/kernel.py": (
        "27e23c584cf862f4091858db6dc85bbc2a01f4baf9dca3f232ae6f3bac02390e",
        "MoEBackend = Literal[\n",
        'MoEBackend = Literal[\n    "apac_sm90_megamoe",\n',
    ),
    "model_executor/layers/quantization/fp8.py": (
        "947540b443c15e9e5b4c08ff07f7a7c0e8d8f0d8494f3964c0172a772a779fe5",
        "            if self.is_checkpoint_fp8_serialized:\n                return Fp8MoEMethod(self, layer)\n",
        '            if self.is_checkpoint_fp8_serialized:\n                if layer.moe_config.moe_backend == "apac_sm90_megamoe":\n                    from vllm.model_executor.layers.quantization.apac_sm90 import Sm90MegaMoEMethod\n                    return Sm90MegaMoEMethod(self, layer)\n                return Fp8MoEMethod(self, layer)\n',
    ),
    "model_executor/layers/fused_moe/fused_moe_method_base.py": (
        "18dcfc469d3252749801bbf943857f58da85de1f2db8fdff5a9f4986a925dc68",
        "    @property\n    def supports_internal_mk(self) -> bool:\n",
        "    @property\n    def output_is_reduced(self) -> bool:\n        return self.moe_kernel is not None and self.moe_kernel.output_is_reduced()\n\n    @property\n    def supports_internal_mk(self) -> bool:\n",
    ),
    "model_executor/layers/fused_moe/runner/moe_runner.py": (
        "05893188402fe37229bf72d75088d66193c417d5d502daafd3a55e1bfedaa60c",
        "        return (\n            self._quant_method.moe_kernel is not None\n            and self._quant_method.moe_kernel.output_is_reduced()\n        )\n",
        "        return self._quant_method.output_is_reduced\n",
    ),
}


def patch(source, name):
    expected, before, after = PATCHES[name]
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("Unsupported pinned vLLM adapter source: " + name)
    text = source.decode()
    if text.count(before) != 1:
        raise ValueError("Nonunique adapter anchor: " + name)
    result = text.replace(before, after).encode()
    ast.parse(result)
    return result


def install(root):
    outputs = {name: patch((root / name).read_bytes(), name) for name in PATCHES}
    adapter = Path(__file__).with_name("vllm_method.py").read_bytes()
    ast.parse(adapter)
    target = root / "model_executor/layers/quantization/apac_sm90.py"
    if target.exists():
        raise ValueError("Adapter destination already exists")
    for name, data in outputs.items():
        (root / name).write_bytes(data)
    target.write_bytes(adapter)
    return {
        "schema": "apac-vllm-sm90-adapter/v1",
        "source_sha256": {n: x[0] for n, x in PATCHES.items()},
        "patched_sha256": {
            n: hashlib.sha256(b).hexdigest() for n, b in outputs.items()
        },
        "adapter_sha256": hashlib.sha256(adapter).hexdigest(),
        "backend": "apac_sm90_megamoe",
    }


if __name__ == "__main__":
    print(json.dumps(install(Path(sys.argv[1])), indent=2))
