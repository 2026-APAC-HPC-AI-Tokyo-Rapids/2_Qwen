"""Check the new FC2 activation boundary in the existing numeric-gate step.

No scheduler submission. Compare against the actual unfused CUDA operations
and CUDA Graph replay with changed inputs. Bit equality is diagnostic only;
the tolerance below is a local gross-error screen, not a competition metric.
"""
import json

import torch
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.attention.rope import (
    apply_qknorm_rope_fp8_q,
    apply_rope_fp8_q,
    apply_rope_inplace,
)
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from sglang.kernels.ops.moe.decode_silu_quant import decode_silu_quant
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8


def check_numerics(actual, expected, label):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, label
    bitwise = torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    a, b = actual.float(), expected.float()
    assert torch.equal(a.isnan(), b.isnan()), (label, "NaN locations")
    assert torch.equal(a.isinf(), b.isinf()), (label, "infinity locations")
    finite = b.isfinite()
    a, b = a[finite], b[finite]
    if b.numel():
        error = a - b
        relative_l2 = (error.norm() / b.norm().clamp_min(1e-12)).item()
        max_scaled = (error.abs().amax() / b.abs().amax().clamp_min(1e-12)).item()
    else:
        relative_l2 = max_scaled = 0.0
    print(json.dumps({"check": label, "bitwise_equal": bitwise, "relative_l2": relative_l2, "max_scaled_error": max_scaled}), flush=True)
    # Allow rounding differences. These bounds reject broken indexing/scaling;
    # they do not establish end-to-end model quality or official eligibility.
    assert relative_l2 <= 0.05 and max_scaled <= 0.25, label


def check_decode_silu_quant():
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.manual_seed(381108)
    for rows in (0, 1, 7, 8, 31, 32, 257, 1024, 1536, 4096, 6144):
        x = torch.randn(rows, 768, device="cuda", dtype=torch.bfloat16) * 4
        if rows:
            x[0].zero_()
        expected_activation = torch.empty((rows, 384), device="cuda", dtype=torch.bfloat16)
        if rows:
            silu_and_mul(x, expected_activation)
        expected, scales = sglang_per_token_group_quant_fp8(expected_activation, 128)
        actual, actual_scales = decode_silu_quant(x)
        check_numerics(actual, expected, f"silu_quant_{rows}")
        check_numerics(actual_scales, scales, f"silu_scales_{rows}")

    x = torch.randn(1024, 768, device="cuda", dtype=torch.bfloat16)
    # Warmup on a side stream before graph capture; every buffer is graph-owned.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode_silu_quant(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, actual_scales = decode_silu_quant(x)
    x.copy_(torch.randn_like(x) * 8)
    graph.replay()
    expected, scales = sglang_per_token_group_quant_fp8(silu_and_mul(x), 128)
    check_numerics(actual, expected, "silu_quant_graph")
    check_numerics(actual_scales, scales, "silu_scales_graph")
    print("PASS: Decode SiLU/group128 FP8 numeric screen; empty/tail rows; graph replay", flush=True)


def check_rope_fp8_q(with_norm=False):
    torch.manual_seed(6300)
    device = "cuda"
    angles = torch.arange(8192, device=device).float()[:, None] * (1000000.0 ** (-torch.arange(0, 128, 2, device=device).float() / 128))
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()

    q_weight = (torch.randn(128, device=device) * 0.25 + 1).bfloat16()
    k_weight = (torch.randn(128, device=device) * 0.25 + 1).bfloat16()
    eps = 1e-6

    def forward(q, k, positions, actual):
        if with_norm:
            apply_qknorm_rope_fp8_q(q, k, q_weight, k_weight, cache, positions, eps, actual)
        else:
            apply_rope_fp8_q(q, k, cache, positions, actual)

    def reference(q, k, positions):
        expected_q, expected_k = q.clone(), k.clone()
        if q.shape[0]:
            if with_norm:
                fused_inplace_qknorm(expected_q, expected_k, q_weight, k_weight, eps=eps, head_dim=128)
            apply_rope_inplace(expected_q, expected_k, cache, positions, is_neox=True)
        return expected_q.to(torch.float8_e4m3fn), expected_k

    def compare(q, k, actual, expected, expected_k):
        check_numerics(actual, expected, f"rope_q_norm{with_norm}_{q.shape[0]}")
        check_numerics(k, expected_k, f"rope_k_norm{with_norm}_{k.shape[0]}")

    for rows in (0, 1, 4, 31, 116, 128, 4058, 4096):
        for id_dtype in (torch.int32, torch.int64):
            # Packed QKV views preserve the real token stride, unlike a dense clone.
            packed = torch.randn(rows, 72, 128, device=device, dtype=torch.bfloat16) * 4
            if rows:
                packed[0].zero_()
            q, k = packed[:, :64], packed[:, 64:68]
            if id_dtype == torch.int64:
                q, k = q.contiguous(), k.contiguous()
            original_q = q.clone()
            positions = torch.randint(0, 8192, (rows,), device=device, dtype=id_dtype)
            expected, expected_k = reference(q, k, positions)
            actual = torch.empty(q.shape, device=device, dtype=torch.float8_e4m3fn)
            forward(q, k, positions, actual)
            compare(q, k, actual, expected, expected_k)
            assert torch.equal(q.view(torch.uint8), original_q.view(torch.uint8)), "Q input was modified"

    if not with_norm:
        # All 65,536 BF16 bit patterns include FP8 ties, subnormals, signed zeros,
        # saturation boundaries, overflow, infinities, and NaNs.
        q = torch.arange(65536, device=device, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).reshape(8, 64, 128)
        k = torch.zeros(8, 4, 128, device=device, dtype=torch.bfloat16)
        positions = torch.zeros(8, device=device, dtype=torch.int64)
        expected, expected_k = reference(q, k, positions)
        actual = torch.empty(q.shape, device=device, dtype=torch.float8_e4m3fn)
        forward(q, k, positions, actual)
        compare(q, k, actual, expected, expected_k)

    q = torch.randn(116, 64, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(116, 4, 128, device=device, dtype=torch.bfloat16)
    actual = torch.empty(q.shape, device=device, dtype=torch.float8_e4m3fn)
    positions = torch.arange(116, device=device, dtype=torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward(q, k, positions, actual)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward(q, k, positions, actual)
    for replay in range(3):
        q.copy_(torch.randn_like(q) * (replay + 1))
        k.copy_(torch.randn_like(k))
        positions.add_(997)
        expected, expected_k = reference(q, k, positions)
        graph.replay()
        compare(q, k, actual, expected, expected_k)
    print(f"PASS: RoPE (QK norm={with_norm}) FP8 Q/BF16 K numeric screen; packed/contiguous strides; graph replay", flush=True)


if __name__ == "__main__":
    check_decode_silu_quant()
    check_rope_fp8_q()
    check_rope_fp8_q(with_norm=True)
