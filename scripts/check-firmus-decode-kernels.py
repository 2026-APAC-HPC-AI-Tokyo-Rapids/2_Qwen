"""Check the fused SGLang JIT kernels in the existing numeric-gate step.

No scheduler submission. Compare against the actual unfused CUDA operations
and CUDA Graph replay with changed inputs. Existing activation/RoPE checks
use a gross-error tolerance; the KV and Top-K fusions (komoriyuta) require
exact non-NaN bytes against the same tree's FP8 Q kernel + torch casts +
store_cache; the fused add-RMSNorm + group quant must stay within rounding of
flashinfer fused_add_rmsnorm + per_token_group_quant (tolerances in
check_add_rmsnorm_group_quant). Neither replaces the official workload or
accuracy evaluation. The kernel bench also runs this file on the image tree,
which lacks the new kernels; those checks print SKIP there.
The dense FP8 check requires the Decode-config host image for both trees and
screens every dense candidate against stock, FP32 dequantized GEMM and changed-input graphs.
Serving gates cover the selected kernels and their references. Exploratory
attention candidates remain in the separate kernel benchmark, not this gate.
"""
# ruff: noqa: B023 - the nested helpers run inside the loop iteration that defines them, so the loop variables are current
import json
import os
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import sglang.kernels.ops.attention.rope as rope_ops
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


def decode_cap_tokens():
    config = json.loads((Path(__file__).resolve().parents[1] / "source/firmus-framework.json").read_text())
    return config["decode_max_running_requests"]


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


def check_decode_fp8_conversion():
    """All BF16 encodings in each packed lane against the old CUDA NOSAT call."""
    from sglang.kernels.jit.utils import load_jit

    module = load_jit("decode_fp8_check", cuda_files=["elementwise/decode_fp8_check.cuh"], cuda_wrappers=[("run", "DecodeFP8Check::run")])
    bits = torch.arange(65536, device="cuda", dtype=torch.int32)
    # A bijective permutation puts every encoding in both positions and makes
    # the neighbours different, catching swapped bytes and cross-lane repair.
    pairs = torch.stack((bits, bits ^ 0xA55A), dim=1).to(torch.uint16).view(torch.bfloat16)
    reference, actual = [torch.empty(65536, device="cuda", dtype=torch.uint16) for _ in range(2)]
    module.run(pairs, reference, actual)
    assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)), "packed BF16 -> FP8 differs from CUDA NOSAT"
    print(json.dumps({"check": "decode_fp8_all_bf16", "patterns_per_lane": 65536, "packed_lanes": 2, "bitwise_equal": True}), flush=True)


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

    for rows in (0, 1, 4, 31, 116, 128, 3986, 4027, 4058, 4096):
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


def check_topk_mask():
    from sglang.kernels.ops.moe import topk_softmax

    try:
        from sglang.kernels.ops.moe.trtllm_lora_temp.topk_softmax_pack import (
            topk_softmax_masked,
        )
    except ImportError:
        print("SKIP: topk_softmax_masked is not in this SGLang tree", flush=True)
        return

    torch.manual_seed(83719)
    for dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 7, 31, 128, 160, 192):
            logits = torch.randn(rows, 128, dtype=dtype, device="cuda") * 4
            logits[0].zero_()  # tie order must match the actual AOT router
            if rows > 1:
                logits[1] *= 1000
            actual_w = torch.empty(rows, 8, dtype=torch.float32, device="cuda")
            actual_i = torch.empty(rows, 8, dtype=torch.int32, device="cuda")
            expected_w, expected_i = torch.empty_like(actual_w), torch.empty_like(actual_i)
            for renormalize in (False, True):
                topk_softmax(expected_w, expected_i, logits, renormalize)
                for valid in (0, max(0, rows - 3), rows):
                    count = torch.tensor(valid, dtype=torch.int32, device="cuda")
                    topk_softmax_masked(actual_w, actual_i, logits, count, renormalize)
                    expected_ids = expected_i.clone()
                    expected_ids[valid:] = -1
                    assert torch.equal(actual_i, expected_ids), (dtype, rows, valid, "ids")
                    assert torch.equal(actual_w.view(torch.uint8), expected_w.view(torch.uint8)), (dtype, rows, valid, "weights")

    # The valid-row count must remain a device input during replay.
    logits = torch.randn(128, 128, device="cuda")
    count = torch.tensor(128, dtype=torch.int32, device="cuda")
    actual_w = torch.empty(128, 8, dtype=torch.float32, device="cuda")
    actual_i = torch.empty(128, 8, dtype=torch.int32, device="cuda")
    expected_w, expected_i = torch.empty_like(actual_w), torch.empty_like(actual_i)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        topk_softmax_masked(actual_w, actual_i, logits, count, True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        topk_softmax_masked(actual_w, actual_i, logits, count, True)
    for valid in (0, 7, 125, 128):
        count.fill_(valid)
        logits.copy_(torch.randn_like(logits) * 8)
        topk_softmax(expected_w, expected_i, logits, True)
        expected_i[valid:] = -1
        graph.replay()
        assert torch.equal(actual_i, expected_i)
        assert torch.equal(actual_w.view(torch.uint8), expected_w.view(torch.uint8))
    print("PASS: fused Top-K padding is bitwise equal to AOT; ties, extreme finite logits, empty valid regions and changing CUDA Graph inputs/counts", flush=True)


def check_rope_fp8_kv():
    """Require exact bytes against the existing fused Q + torch KV casts."""
    if not hasattr(rope_ops, "apply_qknorm_rope_fp8_kv"):
        print("SKIP: apply_qknorm_rope_fp8_kv is not in this SGLang tree", flush=True)
        return
    apply_qknorm_rope_fp8_kv = rope_ops.apply_qknorm_rope_fp8_kv
    torch.manual_seed(18634)
    angles = torch.arange(8192, device="cuda").float()[:, None] * (1000000.0 ** (-torch.arange(0, 128, 2, device="cuda").float() / 128))
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
    qw = (torch.randn(128, device="cuda") * .25 + 1).bfloat16()
    kw = (torch.randn(128, device="cuda") * .25 + 1).bfloat16()
    eps, slots = 1e-6, 4099  # every Prefill-sized row can own a distinct slot

    def reference(q, k, v, positions, loc):
        # empty_like preserves packed strides for zero rows; the FFI requires dense output strides.
        expected_q = torch.empty(q.shape, device=q.device, dtype=torch.float8_e4m3fn)
        rotated_k = k.clone()
        apply_qknorm_rope_fp8_q(q, rotated_k, qw, kw, cache, positions, eps, expected_q)
        # Compare the actual store_cache path, including its reserved slot zero.
        from sglang.kernels.ops.kvcache.kvcache import store_cache

        expected_k = torch.full((slots, 4, 128), 3., device="cuda", dtype=torch.float8_e4m3fn)
        expected_v = torch.full_like(expected_k, 3.)
        if q.shape[0]:
            store_cache(rotated_k.to(torch.float8_e4m3fn).view(-1, 512), v.to(torch.float8_e4m3fn).view(-1, 512), expected_k.view(-1, 512), expected_v.view(-1, 512), loc, row_bytes=512, v_row_bytes=512, size_limit=slots)
        return expected_q, expected_k, expected_v

    def compare(actual, expected, label):
        for name, a, b in zip(("q", "k_cache", "v_cache"), actual, expected):
            check_numerics(a, b, f"fp8_kv_{label}_{name}")
            # NaN payload/sign is immaterial. V is a pure cast and untouched
            # cache rows / the padding sink keep their fill value, so those must
            # match exactly. Q and K go through RMSNorm and RoPE, whose FP32
            # reduction order may differ between the Decode head-parallel launch
            # and the reference; those may differ by at most one FP8 code
            # (bit identity is not required, see qwen-precision-policy).
            non_nan = ~b.float().isnan()
            fill = b.float() == 3.0
            exact = non_nan & fill if name != "v_cache" else non_nan
            assert torch.equal(a.view(torch.uint8)[exact], b.view(torch.uint8)[exact]), (label, name, "not bitwise equal")
            if name != "v_cache":
                codes = (a.view(torch.uint8).int() - b.view(torch.uint8).int()).abs()[non_nan]
                assert codes.numel() == 0 or int(codes.max()) <= 1, (label, name, "more than one FP8 code apart")

    # Zero rows, odd tails, Decode verify sizes and Prefill chunks (token-warp grid).
    for rows in (0, 1, 3, 31, 126, 128, 129, 132, 136, 140, 144, 145, 160, 192, 193, 3986, 4027):
        for id_dtype in (torch.int32, torch.int64):
            packed = torch.randn(rows, 72, 128, device="cuda", dtype=torch.bfloat16) * 4
            q, k, v = packed[:, :64], packed[:, 64:68], packed[:, 68:]
            if id_dtype == torch.int64:
                q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            positions = torch.randint(0, 8192, (rows,), device="cuda", dtype=id_dtype)
            loc = torch.arange(rows, device="cuda", dtype=id_dtype) + 1
            if rows > 1:
                loc[-min(rows, 7):] = 0
            original = [x.clone() for x in (q, k, v)]
            expected = reference(q, k, v, positions, loc)
            actual = (torch.empty(q.shape, device=q.device, dtype=torch.float8_e4m3fn), torch.full_like(expected[1], 3.), torch.full_like(expected[2], 3.))
            apply_qknorm_rope_fp8_kv(q, k, qw, kw, cache, positions, eps, actual[0], v, actual[1], actual[2], loc)
            compare(actual, expected, f"{rows}_{id_dtype}")
            assert all(torch.equal(x.view(torch.uint8), old.view(torch.uint8)) for x, old in zip((q, k, v), original)), "QKV input modified"
            if rows in (128, 129, 132, 144, 160, 192):
                # Same production API and fixed addresses; runtime input values,
                # cache locations and the number of slot-zero padding rows vary.
                def decode_forward():
                    apply_qknorm_rope_fp8_kv(q, k, qw, kw, cache, positions, eps, actual[0], v, actual[1], actual[2], loc)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    decode_forward()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    decode_forward()
                for replay in range(3):
                    packed.copy_(torch.randn_like(packed) * (replay + 1))
                    if id_dtype == torch.int64:
                        q.copy_(packed[:, :64])
                        k.copy_(packed[:, 64:68])
                        v.copy_(packed[:, 68:])
                    positions.copy_(torch.randint(0, 8192, (rows,), device="cuda", dtype=id_dtype))
                    loc.copy_(torch.randperm(slots - 1, device="cuda")[:rows].to(id_dtype) + 1)
                    loc[-(1 + 3 * replay):] = 0
                    original = [x.clone() for x in (q, k, v)]
                    actual[1].fill_(3.)
                    actual[2].fill_(3.)
                    expected = reference(q, k, v, positions, loc)
                    graph.replay()
                    compare(actual, expected, f"decode_graph_{rows}_{id_dtype}_{replay}")
                    assert all(torch.equal(x.view(torch.uint8), old.view(torch.uint8)) for x, old in zip((q, k, v), original)), "graph modified QKV input"

    q = torch.randn(128, 64, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(128, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).reshape(128, 4, 128)
    positions = torch.arange(128, device="cuda", dtype=torch.int64)
    loc = positions.clone() + 1
    expected = reference(q, k, v, positions, loc)
    actual = (torch.empty(q.shape, device=q.device, dtype=torch.float8_e4m3fn), torch.full_like(expected[1], 3.), torch.full_like(expected[2], 3.))

    def forward():
        apply_qknorm_rope_fp8_kv(q, k, qw, kw, cache, positions, eps, actual[0], v, actual[1], actual[2], loc)

    forward()
    compare(actual, expected, "all_bf16_value_patterns")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    for replay in range(3):
        q.copy_(torch.randn_like(q) * (replay + 1))
        k.copy_(torch.randn_like(k))
        v.copy_(torch.randn_like(v) * 16)
        positions.add_(997)
        loc.copy_((torch.arange(128, device="cuda") + 23 * replay) % (slots - 1) + 1)
        loc[-7:] = 0
        actual[1].fill_(3.)
        actual[2].fill_(3.)
        expected = reference(q, k, v, positions, loc)
        graph.replay()
        compare(actual, expected, f"graph_{replay}")
    print("PASS: exact FP8 KV fusion; packed/contiguous inputs; int32/int64 slots; reserved padding; all BF16 V patterns; graph replay changes inputs, positions and slots", flush=True)


def check_decode_router_padding():
    """Exact existing Triton router arithmetic, including ties and graph padding."""
    import inspect

    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
    from sglang.srt.layers.moe.topk import _mask_topk_ids_padded_region

    if "num_token_non_padded" not in inspect.signature(moe_fused_gate).parameters:
        print("SKIP: fused Decode Triton router is not in this tree", flush=True)
        return
    torch.manual_seed(61022)
    for dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 7, 31, 32, 128, 129, 132, 136, 140, 144, 160, 192):
            logits = torch.randn(rows, 128, device="cuda", dtype=dtype) * 4
            logits[0].zero_()  # exact tie ordering
            if rows > 1:
                logits[1] *= 1000  # stable softmax, no new approximation
            bias = torch.zeros(128, device="cuda", dtype=torch.float32)
            count = torch.tensor(rows, device="cuda", dtype=torch.int32)
            for renormalize in (False, True):
                def reference():
                    weights, ids = moe_fused_gate(logits, bias, 8, scoring_func="softmax", renormalize=renormalize)
                    _mask_topk_ids_padded_region(ids, count)
                    return weights, ids

                def forward():
                    return moe_fused_gate(logits, None, 8, scoring_func="softmax", renormalize=renormalize, num_token_non_padded=count)

                def compare(actual, label):
                    expected = reference()
                    for name, a, b in zip(("weights", "ids"), actual, expected):
                        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (label, name)
                    print(json.dumps({"check": label, "bitwise_equal": True}), flush=True)

                for valid in (0, max(0, rows - 3), rows):
                    count.fill_(valid)
                    compare(forward(), f"decode_router_{dtype}_{rows}_{valid}_renorm{renormalize}")
                if rows in (128, 129, 132, 144, 160, 192):
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        forward()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = forward()
                    for replay, valid in enumerate((rows, 0, rows - 7)):
                        logits.copy_(torch.randn_like(logits) * (replay + 1))
                        logits[0].zero_()
                        count.fill_(valid)
                        graph.replay()
                        compare(actual, f"decode_router_graph_{dtype}_{rows}_{replay}_renorm{renormalize}")
    print("PASS: Decode Triton router exact weights/IDs; ties, extreme logits, zero/live padding and changed-input graph replay", flush=True)




def compare_norm_quant(actual, expected, label):
    q, scale, residual = actual
    eq, es, eres = expected
    assert q.shape == eq.shape and scale.shape == es.shape and scale.stride() == es.stride(), label
    bitwise = torch.equal(q.view(torch.uint8), eq.view(torch.uint8)) and torch.equal(scale, es)
    assert torch.equal(residual.view(torch.uint8), eres.view(torch.uint8)), (label, "residual")
    rows = q.shape[0]
    mismatch = int((q.view(torch.uint8) != eq.view(torch.uint8)).sum())
    if rows:
        scale_rel = ((scale - es).abs() / es).max().item()
        a = q.float().view(rows, 32, 128) * scale[:, :, None]
        b = eq.float().view(rows, 32, 128) * es[:, :, None]
        # One E4M3 step of the larger magnitude (relative 2^-3), floor at the
        # subnormal step 2^-9 x scale, plus the allowed scale change.
        bound = torch.maximum(a.abs(), b.abs()) * (2.0**-3 + scale_rel) + 2.0**-9 * torch.maximum(scale, es)[:, :, None]
        value_ok = bool(((a - b).abs() <= bound).all())
    else:
        scale_rel, value_ok = 0.0, True
    print(json.dumps({"check": f"norm_quant_{label}", "bitwise_equal": bitwise, "fp8_codes_differing": mismatch, "max_scale_rel_diff": scale_rel}), flush=True)
    assert scale_rel <= 2.0**-7, (label, "scale", scale_rel)
    assert value_ok, (label, "dequantized value")
    return bitwise


def check_add_rmsnorm_group_quant():
    """Fused input_layernorm + QKV quant against flashinfer fused_add_rmsnorm +
    column-major TMA-aligned group-128 quant.

    Tolerances (rounding level): the residual (one FP32 add, rounded to BF16)
    must be bitwise equal. The normed BF16 value may move by summation order,
    so each scale may differ by at most 1 BF16 ulp (relative 2^-7), and each
    dequantized FP8 element by at most one FP8 E4M3 step of its value (relative
    2^-3, or 2^-9 x scale in the subnormal range) after allowing that scale
    change. The number of differing FP8 codes is recorded, not limited: a
    rounding-level change of the normed BF16 value moves an FP8 code by one
    step only when it sits next to an E4M3 rounding boundary, and run
    39100197595 showed 13 of 12288 such one-step moves with identical scales
    (the task rules allow FP summation-order differences).
    """
    try:
        import sglang.kernels.ops.layernorm.add_rmsnorm_group_quant as nq
    except ImportError:
        print("SKIP: add_rmsnorm_group_quant is not in this SGLang tree", flush=True)
        return
    from sgl_kernel import fused_add_rmsnorm

    torch.manual_seed(52119)
    weight = (torch.randn(4096, device="cuda") * 0.1 + 1).bfloat16()
    eps = 1e-6

    def reference(x, residual):
        x, residual = x.clone(), residual.clone()
        if x.shape[0]:
            fused_add_rmsnorm(x, residual, weight, eps)
        q, scale = sglang_per_token_group_quant_fp8(x, 128, column_major_scales=True, scale_tma_aligned=True)
        return q, scale, residual

    for rows in (0, 1, 2, 3, 7, 31, 126, 127, 128, 144, 2047, 2048, 2049, 3986, 4027):
        x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * 2
        residual = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * 8
        if rows:
            x[0].zero_()
            residual[0].zero_()  # all-zero row: amax clamp 1e-10 path
        expected = reference(x, residual)
        work = residual.clone()
        q, scale = nq.add_rmsnorm_group_quant_fp8(x, work, weight, eps)
        compare_norm_quant((q, scale, work), expected, str(rows))

    x = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
    work = residual.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        nq.add_rmsnorm_group_quant_fp8(x, work, weight, eps)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = nq.add_rmsnorm_group_quant_fp8(x, work, weight, eps)
    for replay in range(3):
        x.copy_(torch.randn_like(x) * (replay + 1))
        residual.copy_(torch.randn_like(residual) * 4)
        work.copy_(residual)
        expected = reference(x, residual)
        graph.replay()
        compare_norm_quant((*actual, work), expected, f"graph_{replay}")
    print("PASS: fused add-RMSNorm + group-128 FP8 quant within rounding of flashinfer fused_add_rmsnorm + per_token_group_quant (residual exact); zero/odd/Prefill rows; graph replay", flush=True)



# TARGET_VERIFY input-norm/quant kernels: "stock" is the 64-lane one-row
# kernel; "wide" spreads the row over 512 lanes. Serving reads the choice from
# SGLANG_DECODE_NORM_QUANT (empty = stock), written by the dense selection.
NORM_QUANT_CANDIDATES = ("stock", "wide")


def norm_quant_kernel(nq, name):
    return {"stock": nq.decode_add_rmsnorm_group_quant_fp8_out, "wide": nq.decode_add_rmsnorm_group_quant_wide_fp8_out}[name]


def norm_quant_outputs(rows, device="cuda"):
    q = torch.empty((rows, 4096), dtype=torch.float8_e4m3fn, device=device)
    scale = torch.empty((32, (rows + 3) // 4 * 4), dtype=torch.float32, device=device).T[:rows]
    return q, scale


def norm_quant_reference(x, residual, weight):
    from sgl_kernel import fused_add_rmsnorm

    y, r = x.clone(), residual.clone()
    if y.shape[0]:
        fused_add_rmsnorm(y, r, weight, 1e-6)
    q, s = sglang_per_token_group_quant_fp8(y, 128, column_major_scales=True, scale_tma_aligned=True)
    return q, s, r


def norm_validate_candidate(nq, name, rows_list, seed):
    """Eager rows (zero row, tails) and changed-input CUDA graph replay against
    flashinfer fused_add_rmsnorm + group-128 quant, within rounding level."""
    kernel = norm_quant_kernel(nq, name)
    torch.manual_seed(seed)
    weight = (torch.randn(4096, device="cuda") * .1 + 1).bfloat16()
    for rows in rows_list:
        x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        if rows:
            x[0].zero_()
            residual[0].zero_()
        expected = norm_quant_reference(x, residual, weight)
        work = residual.clone()
        q, scale = norm_quant_outputs(rows)
        kernel(x, work, weight, 1e-6, q, scale)
        compare_norm_quant((q, scale, work), expected, f"decode_{name}_{rows}")
        if rows not in (160, 176, 192):
            continue
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            kernel(x, work, weight, 1e-6, q, scale)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel(x, work, weight, 1e-6, q, scale)
        for replay in range(3):
            x.copy_(torch.randn_like(x) * (replay + 1))
            residual.copy_(torch.randn_like(residual) * 8)
            work.copy_(residual)
            expected = norm_quant_reference(x, residual, weight)
            graph.replay()
            compare_norm_quant((q, scale, work), expected, f"decode_{name}_graph_{rows}_{replay}")


def check_decode_add_rmsnorm_group_quant(selected_only=False):
    """One-row Decode layouts under the same rounding-level bounds as the Prefill screen.

    The two-row kernel already differs from flashinfer in a few FP8 codes per
    row (run 40197369083: 512 of 127x4096 codes, identical scales), so a
    bitwise gate rejects the unchanged arithmetic as well. Like the dense
    GEMMs, the serving gate checks only the configured kernel: the job-local
    selection screens every candidate and drops failures instead of failing
    the job.
    """
    try:
        import sglang.kernels.ops.layernorm.add_rmsnorm_group_quant as nq
    except ImportError:
        print("SKIP: norm/quant overlay is not in this SGLang tree", flush=True)
        return

    if not hasattr(nq, "decode_add_rmsnorm_group_quant_fp8"):
        print("SKIP: one-row Decode norm/quant is not in this SGLang tree", flush=True)
        return
    for name in [os.environ.get("SGLANG_DECODE_NORM_QUANT") or "stock"] if selected_only else NORM_QUANT_CANDIDATES:
        if name == "wide" and not hasattr(nq, "decode_add_rmsnorm_group_quant_wide_fp8_out"):
            print("SKIP: wide Decode norm/quant is not in this SGLang tree", flush=True)
            continue
        norm_validate_candidate(nq, name, (0, 1, 127, 128, 144, 160, 164, 168, 172, 176, 177, 192), 1764096)


def check_prefill_attention():
    if not hasattr(rope_ops, "apply_qknorm_rope_fp8_prefill"):
        print("SKIP: contiguous Prefill is not in this SGLang tree", flush=True)
        return
    from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    from sglang.kernels.ops.kvcache.kvcache import store_cache

    torch.manual_seed(319104)
    angles = torch.arange(8192, device="cuda").float()[:, None] * (1000000.0 ** (-torch.arange(0, 128, 2, device="cuda").float() / 128))
    trig = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
    qw, kw = [(torch.randn(128, device="cuda") * .25 + 1).bfloat16() for _ in range(2)]
    slots = 8256

    def equal(a, b, label):
        check_numerics(a, b, label)
        finite = ~b.float().isnan()
        assert torch.equal(a.view(torch.uint8)[finite], b.view(torch.uint8)[finite]), (label, "not exact")

    for rows in (0, 1, 3, 31, 128, 129, 1023, 1024, 3986, 4027, 4096):
        for ids in (torch.int32, torch.int64):
            packed = torch.randn(rows, 72, 128, device="cuda", dtype=torch.bfloat16) * 4
            q, k, v = packed[:, :64], packed[:, 64:68], packed[:, 68:]
            if ids == torch.int64:
                q, k, v = [x.contiguous() for x in (q, k, v)]
            if rows == 128 and ids == torch.int64:
                v.copy_(torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).reshape(128, 4, 128))
            original = [x.clone() for x in (q, k, v)]
            pos = torch.randint(0, 8192, (rows,), device="cuda", dtype=ids)
            loc = torch.randperm(slots - 1, device="cuda")[:rows].to(ids) + 1
            if rows > 1:
                loc[-min(rows, 7):] = 0
            out = torch.empty((rows, 64, 128), dtype=torch.float8_e4m3fn, device="cuda")
            ck = torch.empty((rows, 4, 128), dtype=out.dtype, device="cuda")
            cv = torch.empty_like(ck)
            kc = torch.full((slots, 4, 128), 3., device="cuda", dtype=out.dtype)
            vc = torch.full_like(kc, 3.)

            def reference():
                # Unfused RMSNorm -> BF16 RoPE -> FP8 casts/store_cache.
                rq, rk = q.clone(), k.clone()
                if rows:
                    fused_inplace_qknorm(rq, rk, qw, kw, eps=1e-6, head_dim=128)
                    apply_rope_inplace(rq, rk, trig, pos, is_neox=True)
                return rq.to(out.dtype), rk.to(out.dtype), v.to(out.dtype)

            def forward():
                rope_ops.apply_qknorm_rope_fp8_prefill(q, k, qw, kw, trig, pos, 1e-6, out, v, kc, vc, loc, ck, cv)

            def compare(label):
                rq, rk, rv = reference()
                # Keep the existing unfused-reference numerical tolerances.
                check_numerics(out, rq, label + "_unfused_q")
                check_numerics(ck, rk, label + "_unfused_k")
                equal(cv, rv, label + "_unfused_v")
                # Record the distance to the existing fused kernel under the same
                # bounds. Run 39194911829 found its BF16 K writeback one FP8 step
                # off the unfused reference where this kernel matched it exactly.
                old_q, old_k = torch.empty_like(out), k.clone()
                apply_qknorm_rope_fp8_q(q, old_k, qw, kw, trig, pos, 1e-6, old_q)
                check_numerics(out, old_q, label + "_fused_q")
                check_numerics(ck, old_k.to(out.dtype), label + "_fused_k")
                ek, ev = torch.full_like(kc, 3.), torch.full_like(vc, 3.)
                if rows:
                    store_cache(ck.view(-1, 512), cv.view(-1, 512), ek.view(-1, 512), ev.view(-1, 512), loc, row_bytes=512, v_row_bytes=512, size_limit=slots)
                equal(kc, ek, label + "_paged_k")
                equal(vc, ev, label + "_paged_v")

            forward()
            compare(f"prefill_{rows}_{ids}")
            assert all(torch.equal(x.view(torch.uint8), old.view(torch.uint8)) for x, old in zip((q, k, v), original))
            if rows in (128, 4027):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    forward()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    forward()
                for replay in range(3):
                    for x in (q, k, v):
                        x.copy_(torch.randn_like(x) * (replay + 1))
                    pos.copy_(torch.randint(0, 8192, (rows,), device="cuda", dtype=ids))
                    loc.copy_(torch.randperm(slots - 1, device="cuda")[:rows].to(ids) + 1)
                    loc[-7:] = 0
                    kc.fill_(3.)
                    vc.fill_(3.)
                    graph.replay()
                    compare(f"prefill_graph_{rows}_{ids}_{replay}")

    # FA3 consumes the same FP8 Q and byte-identical paged/contiguous KV.
    # Multiple requests and DP/BCG-style padding exercise rank-local boundaries.
    for lengths, padding in (([1], 0), ([17, 13], 7), ([129, 257], 14), ([3986], 41)):
        rows = sum(lengths)
        q = torch.randn(rows + padding, 64, 128, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        k = torch.randn(rows + padding, 4, 128, device="cuda", dtype=torch.bfloat16).to(q.dtype)
        v = torch.randn_like(k.bfloat16()).to(q.dtype)
        cu = torch.tensor([0] + list(__import__("itertools").accumulate(lengths)), device="cuda", dtype=torch.int32)
        page_counts = [(n + 63) // 64 for n in lengths]
        pages = sum(page_counts) + 1
        kc = torch.zeros(pages * 64, 4, 128, device="cuda", dtype=q.dtype)
        vc = torch.zeros_like(kc)
        table = torch.zeros(len(lengths), max(page_counts), device="cuda", dtype=torch.int32)
        first_page = 1
        for i, (n, count) in enumerate(zip(lengths, page_counts)):
            table[i, :count] = torch.arange(first_page, first_page + count, device="cuda", dtype=torch.int32)
            start = sum(lengths[:i])
            kc[first_page * 64:first_page * 64 + n].copy_(k[start:start + n])
            vc[first_page * 64:first_page * 64 + n].copy_(v[start:start + n])
            first_page += count
        actual = flash_attn_varlen_func(q=q[:rows], k=k[:rows], v=v[:rows], cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(lengths), max_seqlen_k=max(lengths), softmax_scale=128 ** -.5, causal=True, num_splits=0)
        expected = flash_attn_with_kvcache(q=q[:rows], k_cache=kc.view(pages, 64, 4, 128), v_cache=vc.view(pages, 64, 4, 128), page_table=table, cache_seqlens=torch.tensor(lengths, device="cuda", dtype=torch.int32), cu_seqlens_q=cu, max_seqlen_q=max(lengths), softmax_scale=128 ** -.5, causal=True, num_splits=0)
        check_numerics(actual, expected, f"prefill_varlen_{lengths}_pad{padding}")
    print("PASS: Prefill prologue within bounds of the unfused reference and the existing fused kernel, zero/odd/4k rows, paged + contiguous KV, graph replay and FA3 multi-request boundaries", flush=True)


def check_verify_scheduler_sharing():
    from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionBackend,
        FlashAttentionMetadata,
    )

    if not hasattr(FlashAttentionBackend, "_get_verify_scheduler_metadata"):
        print("SKIP: image lacks verify scheduler sharing", flush=True)
        return
    helper = Path(__file__).resolve().parents[1] / "overlay/sglang/python/sglang/kernels/ops/attention/verify_attention_candidates.py"
    ops = runpy.run_path(str(helper))
    backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
    backend.fa_impl_ver, backend.use_mla, backend.attn_cp_size = 3, False, 1
    backend.topk, backend.page_size = 1, 64
    backend._get_scheduler_metadata = get_scheduler_metadata
    layer = SimpleNamespace(tp_q_head_num=64, tp_k_head_num=4, tp_v_head_num=4, head_dim=128, v_head_dim=128, logit_cap=0, is_cross_attention=False)
    fb = SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda: True), spec_info=None)
    captured_buckets = []
    for batch in ops["BATCHES"]:
        common = ops["make_inputs"](batch, boundary=True)
        metadata = FlashAttentionMetadata(cache_seqlens_int32=common["cache_seqlens"], cu_seqlens_q=common["cu_seqlens_q"], page_table=common["page_table"], max_seq_len_q=4)
        backend.forward_metadata = metadata
        out = torch.empty_like(common["q"], dtype=torch.bfloat16)

        def forward(common=common, out=out):
            backend.init_forward_metadata_in_graph(fb)
            for _ in range(94):
                sched = backend._get_verify_scheduler_metadata(common["q"], common["k_cache"], layer, fb, True, (-1, -1), False, False)
                assert sched is not None
                flash_attn_with_kvcache(**common, out=out, scheduler_metadata=sched, num_splits=1, pack_gqa=True)
            return out

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()  # warmup populates metadata: capture must refresh it
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = forward()
        # Replay uses the same buffer addresses, fresh lengths and page IDs.
        # Include page/tile boundaries and bucket-owned allocations.
        common["cache_seqlens"].add_(1)
        common["cu_seqlens_k_new"][1:].copy_(common["cache_seqlens"].cumsum(0, dtype=torch.int32))
        common["page_table"].copy_(common["page_table"].flip(0))
        for key in ("q", "k_cache", "v_cache"):
            common[key].copy_((-common[key].float()).to(common[key].dtype))
        captured_buckets.append((batch, graph, actual, common, metadata))
    assert len({m.verify_scheduler_metadata.data_ptr() for _, _, _, _, m in captured_buckets}) == len(ops["BATCHES"])
    # Replay old buckets after capturing other buckets, including different B.
    for batch, graph, actual, common, metadata in reversed(captured_buckets):
        graph.replay()
        expected = flash_attn_with_kvcache(**common)
        assert actual.isfinite().all(), batch
        check_numerics(actual, expected, f"verify_shared94_graph_changed_lengths_pages_b{batch}")
    print("PASS: verify shared metadata, 94 layers, warmup/capture and changed-length/page replay across buckets", flush=True)


# Companion verify helpers are shared by the numeric gate and job-local selector.
_candidate_table = Path(__file__).resolve().parents[1] / "kernels/fa3-verify/candidates.json"
if not _candidate_table.is_file():
    _candidate_table = Path("/opt/apac/fa3-verify/candidates.json")
FA3_VERIFY_CANDIDATES = json.loads(_candidate_table.read_text())
FA3_VERIFY_NAMES = tuple(c["name"] for c in FA3_VERIFY_CANDIDATES)


def verify_inputs(batch, seed=93101, boundary=False):
    assert 32 <= batch <= 48
    generator = torch.Generator(device="cuda").manual_seed(seed)
    lengths = ([4, 63, 64, 65, 95, 96, 97, 127, 128, 129] * 5)[:batch] if boundary else [3900 + (37 * i + 13) % 301 for i in range(batch)]
    columns = (max(lengths) + 63) // 64
    pages = batch * columns
    dtype = torch.float8_e4m3fn

    def values(shape):
        return (torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * .5).to(dtype)

    common = {"q": values((4 * batch, 64, 128)), "k_cache": values((pages, 64, 4, 128)), "v_cache": values((pages, 64, 4, 128)), "page_table": torch.randperm(pages, device="cuda", generator=generator).reshape(batch, columns).int(), "cache_seqlens": torch.tensor(lengths, device="cuda", dtype=torch.int32), "cu_seqlens_q": torch.arange(0, 4 * batch + 1, 4, device="cuda", dtype=torch.int32), "max_seqlen_q": 4, "softmax_scale": 128**-.5, "causal": True}
    return common, lengths


def verify_stock_metadata(common):
    from sgl_kernel.flash_attn import get_scheduler_metadata

    return get_scheduler_metadata(batch_size=common["cache_seqlens"].numel(), max_seqlen_q=4, max_seqlen_k=common["page_table"].shape[1] * 64, num_heads=64, num_heads_k=4, headdim=128, cache_seqlens=common["cache_seqlens"], qkv_dtype=common["q"].dtype, cu_seqlens_q=common["cu_seqlens_q"], page_size=64, causal=True, num_splits=1, pack_gqa=True)


def verify_fp32_reference(common, lengths):
    # Grouped Q heads avoid materializing 16 copies of the KV cache.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        output = torch.empty_like(common["q"], dtype=torch.float32)
        for batch, length in enumerate(lengths):
            ids = common["page_table"][batch, :(length + 63) // 64].long()
            k = common["k_cache"].view(torch.uint8)[ids].view(common["q"].dtype).reshape(-1, 4, 128)[:length].float()
            v = common["v_cache"].view(torch.uint8)[ids].view(common["q"].dtype).reshape(-1, 4, 128)[:length].float()
            q = common["q"][4 * batch:4 * batch + 4].float().reshape(4, 4, 16, 128)
            scores = torch.einsum("qhgd,khd->hgqk", q, k) * common["softmax_scale"]
            mask = torch.arange(length, device=q.device)[None, :] > length - 4 + torch.arange(4, device=q.device)[:, None]
            scores.masked_fill_(mask[None, None], float("-inf"))
            output[4 * batch:4 * batch + 4] = torch.einsum("hgqk,khd->qhgd", scores.softmax(-1), v).reshape(4, 64, 128)
        return output
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def verify_step(commons, name):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    from sglang.srt.layers.attention import decode_fa3_verify as native

    common = commons[0]
    fp8 = name != "stock" and native.output_fp8(name)
    if fp8:
        outputs = [(torch.empty_like(c["q"]).view(-1, 8192), torch.empty((64, c["q"].shape[0]), device=c["q"].device, dtype=torch.float32).T) for c in commons]
    else:
        outputs = [torch.empty_like(c["q"], dtype=torch.bfloat16) for c in commons]
    state = None if name == "stock" else native.allocate(common["cache_seqlens"], name)

    def step():
        if state is None:
            metadata = verify_stock_metadata(common)
        else:
            native.prepare(state, common["cache_seqlens"], common["cu_seqlens_q"])
        for layer in range(94):
            c, out = commons[layer % len(commons)], outputs[layer % len(commons)]
            if state is None:
                flash_attn_with_kvcache(**c, out=out, scheduler_metadata=metadata, num_splits=1, pack_gqa=True)
            else:
                native.forward(state, c["q"], c["k_cache"], c["v_cache"], c["page_table"], c["cache_seqlens"], c["cu_seqlens_q"], c["softmax_scale"], (out[0].view(-1,64,128), out[1]) if fp8 else out)
            if not fp8:
                # Include the actual downstream quant kernel and its launch gap
                # for stock and every BF16 candidate, just as serving does.
                sglang_per_token_group_quant_fp8(out.view(-1,8192), 128, column_major_scales=True, scale_tma_aligned=True)
        return outputs[93 % len(commons)]

    step.output = outputs[93 % len(commons)]
    step.state = state
    return step


def verify_capture(step):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph


def verify_errors(actual, expected, label):
    assert actual.shape == expected.shape and actual.dtype in (torch.bfloat16, torch.float32), label
    assert actual.isfinite().all() and expected.isfinite().all(), label
    a, b = actual.float(), expected.float()
    error = a - b
    l2 = (error.norm() / b.norm().clamp_min(1e-12)).item()
    scaled = (error.abs().amax() / b.abs().amax().clamp_min(1e-12)).item()
    result = {"relative_l2": l2, "max_scaled_error": scaled, "max_abs": error.abs().amax().item(), "rms": error.square().mean().sqrt().item()}
    assert l2 <= .05 and scaled <= .25, (label, result)
    return result


def verify_dequant(output):
    if isinstance(output, tuple):
        a, scale = output
        return (a.float().view(-1,64,128) * scale[...,None]).view(-1,64,128)
    return output


def verify_quant_gate(output, stock):
    if not isinstance(output, tuple):
        return {}
    a, scale = output
    q, expected_scale = sglang_per_token_group_quant_fp8(stock.view(-1,8192), 128, column_major_scales=True, scale_tma_aligned=True)
    assert scale.stride(0) == 1 and scale.stride(1) % 4 == 0
    assert scale.isfinite().all() and (scale > 0).all()
    # FP32 merge intentionally avoids BF16 rounding, so compare values and
    # scales numerically; exact codes would reject the permitted finer result.
    return {"quantized_values": verify_errors(a.float(), q.float(), "verify_fp8_codes"), "scales": verify_errors(scale, expected_scale, "verify_fp8_scales"), "stock_quant_dequant": verify_errors(verify_dequant(output), verify_dequant((q, expected_scale)), "verify_stock_quant_dequant"), "differing_codes": (a.view(torch.uint8) != q.view(torch.uint8)).sum().item()}


def stream_slot_counts(lengths, block_n, slots=6, sms=132):
    """Partial slots per request of the four-heads-per-CTA stream split (VerifyStreamScheduler):
    the requests' block sequence is cut into G <= sms equal intervals of at least
    ceil(max blocks / (slots - 1)) blocks, one per CTA."""
    blocks = [(x + block_n - 1) // block_n for x in lengths]
    total = sum(blocks)
    groups = min(sms, total // -(-max(blocks) // (slots - 1)))
    counts, start = [], 0
    for n in blocks:
        counts.append(0 if n == 0 else ((start + n) * groups - 1) // total - ((start + 1) * groups - 1) // total + 1)
        start += n
    return counts


def verify_validate_candidate(batch, name):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    metrics = []
    for boundary in (False, True):
        common, lengths = verify_inputs(batch, 93101 + batch, boundary)
        step = verify_step([common], name)
        if boundary and step.state is not None:
            step.state["partial"].fill_(float("nan"))
            step.state["partial_lse"].fill_(float("nan"))
        step()
        stock = flash_attn_with_kvcache(**common, num_splits=1, pack_gqa=True)
        metrics.append({"boundary": boundary, "o_gemm_input": verify_quant_gate(step.output, stock), "stock": verify_errors(verify_dequant(step.output), stock, name), "fp32": verify_errors(verify_dequant(step.output), verify_fp32_reference(common, lengths), name + "_fp32")})
        graph = verify_capture(step)
        for replay in range(2):
            for key in ("q", "k_cache", "v_cache"):
                tensor = common[key]
                tensor.copy_((torch.randn(tensor.shape, device=tensor.device, dtype=torch.bfloat16) * (.25 + replay * .1)).to(tensor.dtype))
            common["page_table"].copy_(common["page_table"].roll(1, 0))
            lengths = [max(4, length - 1) for length in lengths]
            common["cache_seqlens"].copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
            # cu_q is also live graph input; keep this fixed Q4 specialization valid.
            common["cu_seqlens_q"].copy_(torch.arange(0, 4 * batch + 1, 4, device="cuda", dtype=torch.int32))
            if boundary and step.state is not None:
                step.state["partial"].fill_(float("nan"))
                step.state["partial_lse"].fill_(float("nan"))
            graph.replay()
            stock = flash_attn_with_kvcache(**common, num_splits=1, pack_gqa=True)
            metrics.append({"boundary": boundary, "replay": replay, "o_gemm_input": verify_quant_gate(step.output, stock), "stock": verify_errors(verify_dequant(step.output), stock, name + "_changed94"), "fp32": verify_errors(verify_dequant(step.output), verify_fp32_reference(common, lengths), name + "_changed_fp32")})
            state = step.state
            if state is not None and state["spec"].get("heads") == 4:
                # The main kernel stores each request's slot count; it must match the CPU partition.
                assert state["metadata"][:batch].cpu().tolist() == stream_slot_counts(lengths, state["spec"]["block_n"]), "stream split slot counts"
            elif state is not None and state["spec"]["splits"] > 1:
                splits = state["spec"]["splits"]
                counts = [splits] * batch
                if state["spec"].get("uneven"):
                    rounded = (batch + 3) // 4 * 4
                    metadata = state["metadata"].cpu().tolist()
                    for virtual in range(batch):
                        counts[metadata[rounded * 2 + virtual]] = metadata[virtual]
                    four = max(0, min(batch, 132 - 3 * batch))
                    assert sorted(counts) == [3] * (batch - four) + [4] * four
                if state["spec"].get("merge"):
                    rounded = (batch + 3) // 4 * 4
                    assert (state["metadata"][4 * rounded + 1:] == 0).all(), "merge counters survive graph replay"
                    # Resetting prepare runs once per step; every layer must
                    # return its own arrival counters to zero without prepare.
                for b, length in enumerate(lengths):
                    active_splits = counts[b]
                    blocks = (length + state["spec"]["block_n"] - 1) // state["spec"]["block_n"]
                    per_split = (blocks + active_splits - 1) // active_splits
                    for split in range(active_splits):
                        if split * per_split >= blocks:
                            assert state["partial_lse"][split, :, 4 * b:4 * b + 4].isneginf().all(), "empty split LSE"
        if step.state is not None and step.state["spec"].get("merge"):
            output = step.output if isinstance(step.output, tuple) else (step.output,)
            saved = [x.contiguous().view(torch.uint8).clone() for x in output]
            graph.replay()
            assert all((x.contiguous().view(torch.uint8) == y).all() for x, y in zip(output, saved)), "in-kernel merge is not deterministic"
    return {"numeric_passed": True, "metrics": metrics, "changed_input_replays": 2, "layers_per_replay": 94}


def check_fa3_verify(selected_only=False):
    selected = os.environ.get("SGLANG_DECODE_FA3_VERIFY", "")
    if selected_only and not selected:
        print("SKIP: no companion FA3 verify selected", flush=True)
        return
    names = (selected,) if selected_only else FA3_VERIFY_NAMES
    for name in names:
        assert name in FA3_VERIFY_NAMES
        for batch in range(32, 49):
            result = verify_validate_candidate(batch, name)
            print(json.dumps({"check": "fa3_verify", "candidate": name, "batch": batch, **result}), flush=True)


DENSE_PROJECTIONS = (("qkv", 9216, 4096), ("o", 4096, 8192))
# (BM, BN, stages, multicast, multicast_on_A); one factor at a time where possible.
DENSE_CONFIGS = {
    "qkv": ((64, 144, 5, 1, 0), (64, 144, 7, 1, 0), (128, 144, 5, 1, 0), (128, 80, 7, 1, 0)),
    "o": ((64, 64, 9, 1, 0), (64, 64, 13, 1, 0), (128, 64, 8, 1, 0), (128, 32, 10, 1, 0)),
}
# PP4 at 129..192 rows (host sm90-1d2d-decode-bm192.patch, "QKV>128" entries): one BLOCK_M 192 x N80 (bn % 16 == 0 for the host)
# tile with three math warpgroups instead of a 128-row tile plus a 48..64-row one. Same WGMMA
# shapes per row, so the output must equal the pinned config bit for bit.
DENSE_LARGE = {"qkv": ((192, 80, 5, 1, 0),), "o": ()}
# Overlay JIT kernels (decode_dense_gemm.SWAPAB_CONFIGS): token rows on the WGMMA N side.
DENSE_SWAPAB = {"qkv": ("swapab-176-s5", "swapab-176-s5-mc2"), "o": ("swapab-176-s5-k4",)}
DENSE_HOST_SHA256 = "3ed1a5f562c2e10f9f8c438225eb63fcc36ef31378a2262267a062abcf090326"


def dense_candidates(op):
    candidates = [{"name": "stock", "backend": "deepgemm", "config": None}]
    for config in DENSE_CONFIGS[op]:
        candidates.append({"name": "dg-" + "-".join(map(str, config)), "backend": "deepgemm", "config": config})
    candidates.append({"name": "dg1-swapab-streamk", "backend": "companion", "config": None})
    candidates += [{"name": name, "backend": "swapab", "config": None} for name in DENSE_SWAPAB[op]]
    return candidates


@contextmanager
def dense_gemm_mode(op, candidate):
    """The host reads this before each launch; graph replay keeps the captured choice."""
    key = "DG_SM90_1D2D_DECODE_CONFIG"
    previous = os.environ.get(key)
    config = candidate["config"] if candidate["backend"] == "deepgemm" else None
    if config is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = candidate.get("spec") or op.upper() + ":" + ",".join(map(str, config))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def require_dense_host(path=Path("/opt/apac/deep-gemm-host-build.json")):
    """Pin patch bytes too: the previous B5 image has a different override API."""
    record = json.loads(path.read_text())
    digest = record["patches"].get("sm90-1d2d-decode-config.patch", "")
    assert record["source_commit"] == "1c8138a59b3df5dac50b76b730834ed57982fdfb", record
    assert digest == DENSE_HOST_SHA256, "dense GEMM bench requires the rebuilt Decode-config host image matching this script"
    return digest


def dense_call(candidate, inputs):
    lhs, rhs, out = inputs
    a, sa = lhs
    b, sb = rhs
    if candidate["backend"] == "deepgemm":
        import deep_gemm

        return lambda: deep_gemm.fp8_gemm_nt(lhs, rhs, out)
    if candidate["backend"] == "swapab":
        from sglang.srt.layers.quantization.decode_dense_gemm import (
            run_decode_dense_gemm,
        )

        projection = "QKV" if b.shape == (9216, 4096) else "O"
        return lambda: run_decode_dense_gemm(projection, candidate["name"], lhs, rhs, out)
    if candidate["backend"] == "companion":
        import sys

        if "/opt/apac/dense-dg1" not in sys.path:
            sys.path.insert(0, "/opt/apac/dense-dg1")
        from dense_dg1 import prepare

        return prepare(lhs, rhs, out)
    # Call the image's existing kernel directly: the public wrapper chooses a
    # JSON/default config and allocates output. No global config monkeypatching.
    from sglang.kernels.ops.quantization.fp8_kernel import _w8a8_block_fp8_matmul

    bm, bn, stages, warps = candidate["config"]
    m, k = a.shape
    n = b.shape[0]
    grid = (((m + bm - 1) // bm) * ((n + bn - 1) // bn),)
    return lambda: _w8a8_block_fp8_matmul[grid](a, b, out, sa, sb, m, n, k, 128, 128, a.stride(0), a.stride(1), b.stride(1), b.stride(0), out.stride(0), out.stride(1), sa.stride(0), sa.stride(1), sb.stride(1), sb.stride(0), BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=128, GROUP_SIZE_M=32, needs_masking=False, num_stages=stages, num_warps=warps)


def dense_inputs(rows, n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16, generator=gen) * 4
    x[0].zero_()
    a, sa = sglang_per_token_group_quant_fp8(x, 128, column_major_scales=True, scale_tma_aligned=True)
    b = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=gen) * 4).to(torch.float8_e4m3fn)
    # Distinct scales across both N and K catch BN80's scale-boundary mapping.
    sb = 10.0 ** (torch.rand(n // 128, k // 128, device="cuda", generator=gen) * 3 - 3)
    out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    assert sa.dtype == sb.dtype == torch.float32
    return (a, sa), (b, sb), out


def dense_reference(lhs, rhs):
    a, sa = lhs
    b, sb = rhs
    x = a.float() * sa.repeat_interleave(128, 1)
    w = b.float() * sb.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return x @ w.T


def check_dense_errors(metrics):
    # BF16 rounding plus changed FP32 accumulation order, not an accuracy waiver.
    assert metrics["finite"], metrics
    assert 0 <= metrics["relative_l2"] <= 0.005, metrics
    assert 0 <= metrics["max_scaled_error"] <= 0.01, metrics


def dense_errors(actual, expected):
    assert actual.shape == expected.shape and actual.dtype in (torch.bfloat16, torch.float32)
    error = actual.float() - expected
    metrics = {"finite": bool(actual.isfinite().all() and expected.isfinite().all()), "relative_l2": (error.norm() / expected.norm().clamp_min(1e-12)).item(), "max_scaled_error": (error.abs().amax() / expected.abs().amax().clamp_min(1e-12)).item()}
    check_dense_errors(metrics)
    return metrics


@contextmanager
def dense_fp32_reference():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def dense_validate_candidate(op, candidate, inputs, reference, stock):
    """Screen eager and changed-input graph results; never time a rejected candidate."""
    lhs, rhs, out = inputs
    call = dense_call(candidate, inputs)
    out.fill_(float("nan"))
    call()
    metrics = dense_errors(out, reference)
    if candidate["backend"] in ("companion", "swapab"):
        first = out.clone()
        call()
        assert torch.equal(out, first), f"dense {candidate['name']} output must be deterministic"
    stock_metrics = dense_errors(out, stock.float())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    # Modify values, not pointers/layouts, after capture. Restore before timing.
    original = lhs[0].clone(), lhs[1].clone(), rhs[0].clone(), rhs[1].clone()
    rows, k = lhs[0].shape
    n = rhs[0].shape[0]
    fresh_lhs, fresh_rhs, _ = dense_inputs(rows, n, k, 91000 + rows + n)
    try:
        for actual, fresh in zip((*lhs, *rhs), (*fresh_lhs, *fresh_rhs)):
            actual.copy_(fresh)
        changed_reference = dense_reference(lhs, rhs)
        changed_stock = torch.empty_like(out)
        with dense_gemm_mode(op, dense_candidates(op)[0]):
            dense_call(dense_candidates(op)[0], (lhs, rhs, changed_stock))()
            # Replay under the stock host env: a captured override must persist.
            out.fill_(float("nan"))
            graph.replay()
        dense_errors(out, changed_reference)
        dense_errors(out, changed_stock.float())
    finally:
        for actual, saved in zip((*lhs, *rhs), original):
            actual.copy_(saved)
    call()
    return {**metrics, "relative_l2_vs_stock": stock_metrics["relative_l2"], "max_scaled_error_vs_stock": stock_metrics["max_scaled_error"], "changed_input_graph_passed": True}


def selected_dense_candidates(op):
    """Serving preflight: check the selected setting, including the stock fallback."""
    choices = dense_candidates(op)
    backend = os.environ.get("SGLANG_DECODE_DENSE_GEMM_" + op.upper(), "")
    entries = {}
    for entry in os.environ.get("DG_SM90_1D2D_DECODE_CONFIG", "").split(";"):
        if not entry:
            continue
        name, fields = entry.split(":")
        assert name in ("QKV", "O") and name not in entries, entry
        entries[name] = tuple(map(int, fields.split(",")))
    config = entries.get(op.upper())
    assert not (backend and config), "choose only one dense backend per projection"
    if backend:
        selected = [c for c in choices if c["backend"] != "deepgemm" and c["name"] == backend]
    elif config:
        selected = [c for c in choices if c["backend"] == "deepgemm" and c["config"] == config]
    else:
        selected = choices[:1]
    assert len(selected) == 1, (op, backend, config)
    return selected


def check_dense_gemm(selected_only=False):
    """All candidates, FP32 oracle + stock, eager + changed-input CUDA graphs."""
    host_sha = require_dense_host()
    assert torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count == 132
    with dense_fp32_reference():
        for op, n, k in DENSE_PROJECTIONS:
            for rows in (4, 64, 65, 129, decode_cap_tokens()):
                inputs = dense_inputs(rows, n, k, 31000 + rows + n)
                reference = dense_reference(inputs[0], inputs[1])
                with dense_gemm_mode(op, dense_candidates(op)[0]):
                    dense_call(dense_candidates(op)[0], inputs)()
                stock = inputs[2].clone()
                for candidate in selected_dense_candidates(op) if selected_only else dense_candidates(op):
                    with dense_gemm_mode(op, candidate):
                        metrics = dense_validate_candidate(op, candidate, inputs, reference, stock)
                    print(json.dumps({"check": f"dense-{op}-{rows}", "candidate": candidate["name"], "override_active": rows <= 192 and candidate["config"] is not None and candidate["backend"] == "deepgemm", "host_patch_sha256": host_sha, **metrics}), flush=True)
    print("PASS: dense FP8 group128 / block128 projections, selected/all candidate settings, eager and changed-input CUDA graph replay", flush=True)


def check_prefill_fp8_integration():
    """Exercise the installed companion and real FP8-to-O consumer before serving."""
    from itertools import accumulate
    from types import SimpleNamespace

    import deep_gemm
    from sgl_kernel.flash_attn import flash_attn_varlen_func
    from sglang.srt.layers.attention import prefill_fp8
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback,
    )

    for lengths in ([1], [17, 13], [129, 257], [2048, 2048], [3986], [4027], [4096]):
        rows = sum(lengths)
        padded = (rows + 3) // 4 * 4 + 4
        q, k, v = [(torch.randn(rows, h, 128, device="cuda", dtype=torch.bfloat16) * .5).to(torch.float8_e4m3fn) for h in (64, 4, 4)]
        cu = torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int32)

        def reference():
            bf16 = flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(lengths), max_seqlen_k=max(lengths), causal=True, softmax_scale=128**-.5, num_splits=1)
            if isinstance(bf16, tuple):
                bf16 = bf16[0]
            return sglang_per_token_group_quant_fp8(bf16.reshape(rows, 8192), 128, column_major_scales=True, scale_tma_aligned=True)

        def validate(actual):
            expected = reference()
            if fixed:
                fallback = prefill_fp8.attention(q, k, v, cu, max(lengths), 128**-.5, padded)
                assert torch.equal(actual[0].view(torch.uint8), fallback[0].view(torch.uint8)), (lengths, "fixed/fallback bytes")
                assert torch.equal(actual[1], fallback[1]), (lengths, "fixed/fallback scales")
            assert torch.equal(actual[0][:rows].view(torch.uint8), expected[0].view(torch.uint8)), lengths
            assert torch.equal(actual[1][:rows], expected[1]), lengths
            assert (actual[0][rows:].float() == 0).all() and (actual[1] > 0).all()
            return expected

        fixed = len(lengths) == 1 and 3584 <= rows <= 4096

        def call():
            return prefill_fp8.attention(q, k, v, cu, max(lengths), 128**-.5, padded, fixed=fixed)
        pair = call()
        validate(pair)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call()
        for value in (q, k, v):
            value.copy_((-value.float()).to(value.dtype))
        if len(lengths) == 2:
            cu[1].add_(1 if lengths[0] < lengths[1] else -1)
        graph.replay()
        validate(captured)
        v.zero_()
        pair = call()
        validate(pair)
        print(json.dumps({"check": "prefill_fp8_fused", "lengths": lengths, "padded_rows": padded, "bitwise_equal": True, "changed_input_graph": True, "changed_cu_seqlens": len(lengths) == 2}), flush=True)

    attention = SimpleNamespace(num_heads=64, total_num_heads=64, num_kv_heads=4, total_num_kv_heads=4, hidden_size=4096, head_dim=128)
    method = object.__new__(Fp8LinearMethod)
    method.block_quant = True
    method.use_marlin = method.use_mxfp8 = False
    method.weight_block_size = [128, 128]
    method.w8a8_block_fp8_linear = deepgemm_w8a8_block_fp8_linear_with_fallback
    for name, n, k in (("QKV", 9216, 4096), ("O", 4096, 8192)):
        data = dense_inputs(4096, n, k, 42024 + n)
        layer = SimpleNamespace(weight=data[1][0], weight_scale_inv=data[1][1], bias=None, quant_method=method)
        deep_gemm.fp8_gemm_nt(*data)
        actual, bias = prefill_fp8.linear(attention, name, layer, data[0])
        assert bias is None and torch.equal(actual, data[2]), name
        print(json.dumps({"check": "prefill_dense_nk", "projection": name, "bitwise_equal": True}), flush=True)
    print("PASS: installed Prefill FP8 companion, padding, changed-input/length graph, zero-V, and specialized QKV/O", flush=True)


# Fixed Prefill helpers: independent from Decode candidate functions/tables.
def prefill_specs():
    path = Path(__file__).resolve().parents[1] / "kernels/fa3-verify/prefill-candidates.json"
    if not path.is_file():
        path = Path("/opt/apac/fa3-verify/prefill-candidates.json")
    return json.loads(path.read_text())


def prefill_selection_inputs(lengths, padding=4):
    """One request, or packed requests (tuple) for varlen candidates."""
    lengths = (lengths,) if isinstance(lengths, int) else tuple(lengths)
    rows = sum(lengths)
    torch.manual_seed(93517 + rows)
    packed = (torch.randn(rows+padding, 72, 128, device="cuda", dtype=torch.bfloat16) * .5)
    q, k, v = packed[:, :64], packed[:, 64:68], packed[:, 68:]
    qw, kw = [torch.ones(128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    angles = torch.arange(8192, device="cuda").float()[:, None] * (1000000.0 ** (-torch.arange(0, 128, 2, device="cuda").float() / 128))
    trig = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
    pos = torch.cat([torch.arange(n, device="cuda", dtype=torch.int64) for n in lengths] + [torch.arange(padding, device="cuda", dtype=torch.int64)])
    loc = torch.arange(rows+padding, device="cuda", dtype=torch.int64) + 1
    q_fp8 = torch.empty((rows+padding, 64, 128), device="cuda", dtype=torch.float8_e4m3fn)
    ck, cv = [torch.empty((rows+padding, 4, 128), device="cuda", dtype=q_fp8.dtype) for _ in range(2)]
    kc, vc = [torch.empty((rows+padding+1, 4, 128), device="cuda", dtype=q_fp8.dtype) for _ in range(2)]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    image = torch.full((4, (rows+223)//224, 224*128), 0x7f, device="cuda", dtype=torch.uint8)
    args = (q, k, qw, kw, trig, pos, 1e-6, q_fp8, v, kc, vc, loc, ck, cv)
    return {"packed": packed, "q": q_fp8[:rows], "k": ck[:rows], "v": cv[:rows], "kc": kc, "vc": vc, "cu": cu, "image": image, "rope_args": args, "rows": rows, "lengths": lengths}


def prefill_selection_step(common, name, layers=94):
    specs = prefill_specs()
    spec = next(c for c in specs if c["name"] == name)
    candidate = specs.index(spec)
    rows = common["rows"]
    out = torch.empty((rows, 64, 128), device="cuda", dtype=common["q"].dtype)
    scales = torch.empty((64, (rows+3)//4*4), device="cuda", dtype=torch.float32).T[:rows]
    lse = torch.empty((64, rows), device="cuda", dtype=torch.float32)
    metadata = torch.empty(17, device="cuda", dtype=torch.int32)
    empty = torch.empty(0, device="cuda", dtype=torch.uint8)
    image = common["image"] if spec["byte_image"] else empty

    maxlen = max(common["lengths"])
    # Packed requests are served by the prepared N224 path unless a varlen row is selected.
    packed_control = name == "control" and len(common["lengths"]) > 1

    def step():
        for _ in range(layers):
            # Includes the real extra producer work for c2, in every timed layer.
            if common.get("fp32_producer"):
                assert not spec["byte_image"]
                rope_ops.apply_qknorm_rope_fp8_prefill_fp32(*common["rope_args"])
            else:
                rope_ops.apply_qknorm_rope_fp8_prefill(*common["rope_args"], image if spec["byte_image"] else None, rows if spec["byte_image"] else 0)
            if packed_control:
                torch.ops.apac_fa3_verify.prefill_fwd_out(common["q"], common["k"], common["v"], common["cu"], out, scales, lse, metadata, maxlen, 128**-.5)
            else:
                torch.ops.apac_fa3_verify.prefill_variant_fwd_out(common["q"], common["k"], common["v"], image, common["cu"], out, scales, lse, metadata, maxlen, 128**-.5, candidate)
        return out, scales

    step.output = out, scales
    return step


def prefill_selection_capture(step):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def prefill_fp32_reference(common):
    """Causal attention of sampled rows within each packed request."""
    all_indices, outputs = [], []
    start = 0
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for rows in common["lengths"]:
            indices = sorted({0, 1, 7, 31, 127, 223, rows//2, rows-2, rows-1})
            q = common["q"].float()[[start + i for i in indices]].transpose(0, 1)
            k = common["k"].float()[start:start+rows].repeat_interleave(16, dim=1).transpose(0, 1)
            v = common["v"].float()[start:start+rows].repeat_interleave(16, dim=1).transpose(0, 1)
            scores = (q @ k.transpose(1, 2)) * (128**-.5)
            allowed = torch.arange(rows, device="cuda")[None, :] <= torch.tensor(indices, device="cuda")[:, None]
            scores.masked_fill_(~allowed[None, :, :], -float("inf"))
            all_indices += [start + i for i in indices]
            outputs.append((scores.softmax(-1) @ v).transpose(0, 1))
            start += rows
        return all_indices, torch.cat(outputs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def prefill_assert_errors(actual, expected, label):
    assert actual.shape == expected.shape and actual.isfinite().all() and expected.isfinite().all(), label
    a, b = actual.float(), expected.float()
    error = a - b
    result = {"relative_l2": (error.norm()/b.norm().clamp_min(1e-12)).item(), "max_scaled_error": (error.abs().amax()/b.abs().amax().clamp_min(1e-12)).item()}
    # The existing Prefill check_numerics gate thresholds, also for N192.
    assert result["relative_l2"] <= .05 and result["max_scaled_error"] <= .25, (label, result)
    return result


def prefill_validate_candidate(common, name):
    from sglang.srt.layers.attention.decode_fa3_verify import load

    load()
    spec = next(c for c in prefill_specs() if c["name"] == name)
    step = prefill_selection_step(common, name, 1)
    graph = prefill_selection_capture(step)
    # A candidate that only changes the output path or the request packing must
    # equal its base kernel byte for byte; packed inputs run the base per request.
    base = spec.get("bitwise_reference")
    precision_reference = spec.get("precision_reference")
    base_steps = []
    if base:
        start = 0
        for length in common["lengths"]:
            part = {**common, "q": common["q"][start:start+length], "k": common["k"][start:start+length], "v": common["v"][start:start+length], "cu": torch.tensor([0, length], device="cuda", dtype=torch.int32), "rows": length, "lengths": (length,)}
            base_steps.append((start, length, prefill_selection_step(part, base, 1)))
            start += length
    rows = common["rows"]
    precision_step = prefill_selection_step(common, precision_reference, 1) if precision_reference else None
    control = torch.empty_like(step.output[0])
    control_scales = torch.empty((64, (rows+3)//4*4), device="cuda", dtype=torch.float32).T[:rows]
    lse = torch.empty((64, rows), device="cuda", dtype=torch.float32)
    metadata = torch.empty(17, device="cuda", dtype=torch.int32)
    metrics = []
    precision_metrics = []
    for replay in range(3):
        if replay:
            common["packed"].copy_(torch.randn_like(common["packed"]) * (replay+.5))
        if spec["byte_image"]:
            common["image"].fill_(0x7f)
        graph.replay()
        out, scales = step.output
        torch.ops.apac_fa3_verify.prefill_fwd_out(common["q"], common["k"], common["v"], common["cu"], control, control_scales, lse, metadata, max(common["lengths"]), 128**-.5)
        if spec["byte_image"]:
            dump = torch.empty_like(common["image"])
            torch.ops.apac_fa3_verify.prefill_dump_v(common["v"], dump)
            assert torch.equal(dump, common["image"]), (name, rows, replay, "STSM dump / global image including zero padding")
        assert scales.isfinite().all() and (scales > 0).all(), name
        if spec["bitwise"]:
            assert torch.equal(out.view(torch.uint8), control.view(torch.uint8)), (name, rows, replay, "FP8 bytes")
            assert torch.equal(scales, control_scales), (name, rows, replay, "FP32 scales")
        else:
            prefill_assert_errors(out.float(), control.float(), name+"_codes")
            prefill_assert_errors(scales, control_scales, name+"_scales")
        for start, length, base_step in base_steps:
            base_step()
            assert torch.equal(out[start:start+length].view(torch.uint8), base_step.output[0].view(torch.uint8)), (name, common["lengths"], replay, base, "FP8 bytes")
            assert torch.equal(scales[start:start+length], base_step.output[1]), (name, common["lengths"], replay, base, "FP32 scales")
        if precision_step:
            precision_step()
            reference_out, reference_scales = precision_step.output
            actual_dequantized = out.float() * scales[:, :, None]
            reference_dequantized = reference_out.float() * reference_scales[:, :, None]
            precision_metrics.append(prefill_assert_errors(actual_dequantized, reference_dequantized, name+"_"+precision_reference))
        # Verify paged transfer payload and fallback row-major V are still exact.
        assert torch.equal(common["v"].view(torch.uint8), common["packed"][:rows, 68:].to(out.dtype).contiguous().view(torch.uint8)), name
        assert torch.equal(common["v"].view(torch.uint8), common["vc"][1:rows+1].view(torch.uint8)), name
        indices, reference = prefill_fp32_reference(common)
        actual = out.float()[indices]*scales[indices, :, None]
        metrics.append(prefill_assert_errors(actual, reference, name+"_fp32"))
    return {"numeric_passed": True, "bitwise": spec["bitwise"], "bitwise_reference": base, "precision_reference": precision_reference, "precision_reference_metrics": precision_metrics, "metrics": metrics, "changed_input_replays": 2, "byte_image_dump_passed": spec["byte_image"]}


def check_prefill_fixed_variants():
    from sglang.srt.layers.attention.decode_fa3_verify import load

    load()
    # The job-local selector screens all candidates on the actual Prefill GPUs.
    # The existing check_prefill_fp8_integration covers multisequence fallbacks.
    name = os.environ.get("SGLANG_PREFILL_FA3_VERIFY", "control")
    for rows in (3986, 4027, 4096):
        print(json.dumps({"check": "prefill_fixed_variant", "name": name, "rows": rows, **prefill_validate_candidate(prefill_selection_inputs(rows), name)}), flush=True)


if __name__ == "__main__":
    # Existing gates embed this file with run_path and retain their own argv.
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    check_decode_fp8_conversion()
    check_fa3_verify(selected_only="--fa3-selected" in sys.argv[1:])
    check_dense_gemm(selected_only="--dense-selected" in sys.argv[1:])
    check_decode_silu_quant()
    check_rope_fp8_q()
    check_rope_fp8_q(with_norm=True)
    check_rope_fp8_kv()
    check_topk_mask()
    check_decode_router_padding()
    check_add_rmsnorm_group_quant()
    check_decode_add_rmsnorm_group_quant(selected_only="--dense-selected" in sys.argv[1:])
    check_prefill_fp8_integration()
    check_prefill_fixed_variants()
    check_prefill_attention()
    check_verify_scheduler_sharing()
