"""Check native BF16 RMSNorm/group-FP8 fusion before compiled prefill serving.

This bounded operator check does not establish full-model accuracy or speed.
The source image, published scales and private MegaMoE remain unchanged.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def run(output):
    os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "0"
    import torch
    import vllm
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    assert vllm.__version__ == "0.30.0"
    props = torch.cuda.get_device_properties(0)
    assert "H200" in props.name and (props.major, props.minor) == (9, 0)
    torch.manual_seed(2026)
    records = []
    for rows in (1, 128, 4096):
        for residual_enabled in (False, True):
            for pattern in ("random", "zero", "outlier"):
                x = torch.randn((rows, 4096), device="cuda", dtype=torch.bfloat16)
                residual = torch.randn_like(x) if residual_enabled else None
                if pattern == "zero":
                    x.zero_()
                    if residual is not None:
                        residual.zero_()
                elif pattern == "outlier":
                    x[:, ::128] *= 32
                weight = (1 + 0.1 * torch.randn(4096, device="cuda")).bfloat16()
                reference_x = x.clone()
                reference_residual = residual.clone() if residual is not None else None
                if reference_residual is None:
                    norm = torch.empty_like(x)
                    ops.rms_norm(norm, reference_x, weight, 1e-6)
                else:
                    ops.fused_add_rms_norm(reference_x, reference_residual, weight, 1e-6)
                    norm = reference_x
                reference_q, reference_scale = per_token_group_quant_fp8(norm, 128, column_major_scales=True, tma_aligned_scales=True, use_ue8m0=False)
                fused_q, fused_scale = ops.rms_norm_per_block_quant(x, weight, 1e-6, torch.float8_e4m3fn, [1, 128], residual=residual, is_scale_transposed=True, tma_alignment=4)
                if residual is not None:
                    torch.testing.assert_close(residual, reference_residual, rtol=0, atol=0)
                reference = reference_q.float() * reference_scale.repeat_interleave(128, dim=-1)
                actual = fused_q.float() * fused_scale.repeat_interleave(128, dim=-1)
                assert torch.isfinite(actual).all() and torch.isfinite(fused_scale).all()
                difference = actual - reference
                denominator = reference.norm().item()
                relative_l2 = difference.norm().item() / denominator if denominator else difference.norm().item()
                # BF16 rounding order and FP8 bin boundaries can differ. Bound
                # dequantized operator error; never call this bitwise equality.
                assert relative_l2 <= 0.025, (rows, residual_enabled, pattern, relative_l2)
                records.append({"rows": rows, "residual": residual_enabled, "pattern": pattern, "relative_l2": relative_l2, "max_abs_error": difference.abs().max().item()})
    torch.cuda.synchronize()
    result = {"schema": "vllm-native-norm-quant-gate/v1", "completed": True, "passed": True, "vllm": vllm.__version__, "torch": torch.__version__, "device": props.name, "group_size": 128, "hidden_size": 4096, "epsilon": 1e-6, "scale_dtype": "float32", "e8m0": False, "relative_l2_limit": 0.025, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "records": records, "limitations": "Direct native operator comparison only; not actual FX matching, full-model accuracy, capture acceptance or performance."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "records": len(records), "max_relative_l2": max(record["relative_l2"] for record in records)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
