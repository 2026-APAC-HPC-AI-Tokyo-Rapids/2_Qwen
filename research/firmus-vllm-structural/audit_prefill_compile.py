"""Exact-image CPU audit of prefill compile fusion boundaries, not a GPU test."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from audit_pinned_moe import functions

HASHES = {
    "config__compilation.py": "c9cec5c7200e8e559810ec8c30113ad61dab6780f9f7fb3c116bd0d9b5a43065",
    "compilation__breakable_cudagraph.py": "3cc427612a08e2b9b3fee47548026400c1d0776e2d4747535e59ef5512bdf1e8",
    "compilation__passes__fusion__allreduce_rms_fusion.py": "0af193505880cec0f6ffa224c688bce3c9ddc318dccdf04a696b4e4b4506a963",
    "compilation__passes__fusion__qk_norm_rope_fusion.py": "aebc7d1a5189b65ac665afcc169e7baebd53efe719c48e35e7c5babe85af0d39",
    "config__vllm.py": "956b812e5a719bcbfa3a3958801361b38b9c928c96b8c073311bbb96376dfb7a",
    "distributed__device_communicators__flashinfer_all_reduce.py": "b4359760e9b55963676109a5c3a015fb82e190d4df1d021de9ee01e7814cbdda",
}


def literal_assignment(path, name):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "frozenset"
            ):
                value = value.args[0]
            return ast.literal_eval(value)
    raise ValueError(f"missing native constant: {name}")


def audit(directory):
    for name, expected in HASHES.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"pinned source mismatch: {name}")
    ar = directory / "compilation__passes__fusion__allreduce_rms_fusion.py"
    maximum = literal_assignment(ar, "FI_ALLREDUCE_FUSION_MAX_SIZE_MB")[90][2] * 1024**2
    ns = {
        "MiB": 1024**2,
        "_FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB": literal_assignment(
            ar, "_FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB"
        ),
    }
    exec(functions(ar, {"_select_flashinfer_allreduce_use_oneshot"})["code"], ns)  # noqa: S102
    rows = []
    for tokens in [4096, 8192, 16384]:
        size = tokens * 4096 * 2
        rows.append(
            {
                "physical_tokens": tokens,
                "bf16_hidden4096_bytes": size,
                "within_tp2_fusion_size_limit": size <= maximum,
                "native_trtllm_oneshot_selector": ns[
                    "_select_flashinfer_allreduce_use_oneshot"
                ]("trtllm", 90, 2, size),
            }
        )
    assert [r["within_tp2_fusion_size_limit"] for r in rows] == [True, True, False]
    assert [r["native_trtllm_oneshot_selector"] for r in rows] == [True, False, False]
    defaults = literal_assignment(
        directory / "config__vllm.py", "DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES"
    )
    assert "Qwen3MoeForCausalLM" not in defaults
    return {
        "schema": "pinned-prefill-compile-cpu-audit/v1",
        "image_sha256": "2012b52332deaecdab9a7a397a35412b7e098c40308f93560c7a419519ebfb0e",
        "source_sha256": HASHES,
        "attention_tp": 2,
        "hidden": 4096,
        "bf16_element_bytes": 2,
        "maximum_ar_fusion_bytes": maximum,
        "batch_boundaries": rows,
        "qwen3_moe_auto_breakable_default": False,
        "findings": [
            "Breakable graph enabled forces CompilationMode.NONE in config/vllm.py:756-774.",
            "Removing/explicitly disabling breakable plus mode3 makes compile passes eligible, not guaranteed to match an actual FX graph.",
            "VLLM_ALLREDUCE_USE_FLASHINFER=0 does not itself disable the fusion workspace: get_fi_ar_workspace checks it only for expanding the standalone-AR allocation.",
            "AR fusion pass also requires FlashInfer comm/workspace initialization, dtype-matched patterns and discovered model graph nodes.",
            "A1 compile and A2 chunk16384 interact: BF16 hidden4096 batches over8192 tokens exceed the Hopper TP2 64MiB fusion limit. Cannot add their estimated gains or assume fusion persists on full16k batches.",
            "QK norm/RoPE pass requires BF16/FP16, discovered attention layers and supported head dimension; graph match and actual fused runtime kernel remain unverified.",
        ],
        "limitations": "Selected exact source constants and pure CPU one-shot function only. No CUDA import, FlashInfer workspace, graph-pattern matching, capture/memory/startup test, performance or model accuracy measured. The full16k size is a boundary case, not an observed physical batch. Do not infer a hard throughput ceiling from the team analytical model.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.source)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["batch_boundaries"]))


if __name__ == "__main__":
    main()
