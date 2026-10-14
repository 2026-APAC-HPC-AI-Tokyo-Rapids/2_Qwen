"""Audit exact-image Triton EP4 workspace shapes without CUDA execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from audit_pinned_moe import EXPECTED_SOURCE_HASHES, functions

PREFIX = "model_executor__layers__fused_moe__"


def audit(source: Path) -> dict:
    names = [
        PREFIX + "experts__triton_moe.py",
        PREFIX + "modular_kernel.py",
        PREFIX + "fused_moe.py",
        PREFIX + "prepare_finalize__no_dp_ep.py",
    ]
    mapping_name = PREFIX + "expert_map_manager.py"
    names.append(mapping_name)
    expected_hashes = {
        **EXPECTED_SOURCE_HASHES,
        mapping_name: "308c00f63fcb6f44518ec3a56d8f902c7bba75928c2e3aea30fbe1f3f473973b",
    }
    hashes = {}
    for name in names:
        digest = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if digest != expected_hashes[name]:
            raise ValueError(f"pinned source mismatch: {name}")
        hashes[name] = digest
    namespace = {}
    # Execute only selected native CPU shape functions after exact hash checks.
    exec(  # noqa: S102
        functions(
            source / names[1], {"adjust_N_for_activation"}, "FusedMoEExpertsModular"
        )["code"],
        namespace,
    )
    exec(  # noqa: S102
        functions(source / names[0], {"workspace_shapes"}, "TritonExperts")["code"],
        namespace,
    )
    mapping_namespace = {
        "torch": SimpleNamespace(
            full=lambda shape, value, **_kwargs: [value] * shape[0],
            arange=lambda *args, **_kwargs: list(range(*args)),
            int32="int32",
        )
    }
    exec(  # noqa: S102
        functions(source / mapping_name, {"determine_expert_map"})["code"],
        mapping_namespace,
    )
    mappings = []
    for rank in range(4):
        count, mapping, mask = mapping_namespace["determine_expert_map"](4, rank, 128)
        local = [expert for expert, index in enumerate(mapping) if index >= 0]
        assert (
            count == 32
            and mask is None
            and local == list(range(rank * 32, (rank + 1) * 32))
        )
        assert [mapping[expert] for expert in local] == list(range(32))
        assert mapping.count(-1) == 96
        mappings.append(
            {"ep_rank": rank, "local_global_experts": local, "nonlocal_entries": 96}
        )
    native_adjust = namespace["adjust_N_for_activation"]
    shape_self = SimpleNamespace(adjust_N_for_activation=native_adjust)
    activation = SimpleNamespace(is_gated=True)
    rows = []
    for tokens in [128, 256, 288, 384]:
        layouts = {}
        for label, experts, intermediate in [
            ("tp4_ep1", 128, 384),
            ("attention_tp4_moe_tp1_ep4", 32, 1536),
        ]:
            gate_up = 2 * intermediate
            shapes = namespace["workspace_shapes"](
                shape_self, tokens, gate_up, 4096, 8, 128, experts, None, activation
            )
            layouts[label] = {
                "local_experts": experts,
                "intermediate_width": intermediate,
                "gate_up_cache_shape": [tokens, 8, gate_up],
                "activation_cache_shape": [tokens * 8, intermediate],
                "down_cache_shape": [tokens, 8, 4096],
                "native_workspace_shapes": shapes,
                "fp8_expert_weight_elements_per_layer_per_rank": experts
                * 3
                * intermediate
                * 4096,
                "gate_up_cache_elements": tokens * 8 * gate_up,
                "activation_cache_elements": tokens * 8 * intermediate,
            }
        old, new = layouts.values()
        assert old["native_workspace_shapes"] == new["native_workspace_shapes"]
        assert (
            old["fp8_expert_weight_elements_per_layer_per_rank"]
            == new["fp8_expert_weight_elements_per_layer_per_rank"]
        )
        assert new["gate_up_cache_elements"] == 4 * old["gate_up_cache_elements"]
        assert new["activation_cache_elements"] == 4 * old["activation_cache_elements"]
        rows.append({"illustrative_tokens_not_measured": tokens, "layouts": layouts})
    return {
        "serving_image_sha256": "2012b52332deaecdab9a7a397a35412b7e098c40308f93560c7a419519ebfb0e",
        "native_source_sha256": hashes,
        "source_evidence": {
            "prepare": "no_dp_ep.py:58-81 quantizes the original a1 and returns no compact expert metadata",
            "assignment": "triton_moe.py:332-345 uses global expert alignment; fused_moe.py:1544-1590 defaults ignore_invalid_experts=False",
            "nonlocal": "fused_moe.py:423-440 writes zero outputs and returns for expert_id=-1",
            "activation": "triton_moe.py:475-485 invokes SiLU/block quantization over the full gate-up view in E8M0-off FP8 mode",
            "combine": "triton_moe.py:576-580 reduces full M/topk/K output",
            "alignment_wrapper": {
                "path": "model_executor/layers/fused_moe/moe_align_block_size.py",
                "sha256": "c3f7fc2087836f0160a32ab99ceb1d8f6e87c793679da34dd672e225cb7fa31b",
                "lines": "23-45,90-103: default aligns global routes then maps nonlocal experts to -1; compact invalid-expert filtering exists but is not requested by this call",
            },
        },
        "native_linear_expert_mappings": mappings,
        "shape_checks": rows,
        "decision": "Native EP4 is shape-compatible but is not a compact-local-route optimization. Keep candidate held pending profile/route evidence; do not assume lower activation/quantization traffic or4x speedup.",
        "limitations": "Selected exact native Python CPU shape/placement functions with minimal CPU array stand-ins only. No CUDA/C++ alignment execution, weight loading, numerical validation, actual route histogram, memory peak, capture or speed measured. Illustrative token shapes are not observed batch samples. Wider GEMMs may help despite full-route overhead; net effect is unknown.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.source)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["decision"])


if __name__ == "__main__":
    main()
