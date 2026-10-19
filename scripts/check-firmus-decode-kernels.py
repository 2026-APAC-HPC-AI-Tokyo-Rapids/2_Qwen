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
"""
# ruff: noqa: B023 - the nested helpers run inside the loop iteration that defines them, so the loop variables are current
import json

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
        for rows in (1, 7, 31, 128):
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
            # NaN payload/sign is immaterial, but all finite values, signed
            # zeros, untouched cache rows, and padding sink must match exactly.
            non_nan = ~b.float().isnan()
            assert torch.equal(a.view(torch.uint8)[non_nan], b.view(torch.uint8)[non_nan]), (label, name, "not bitwise equal")

    # Zero rows, odd tails, Decode verify sizes and Prefill chunks (token-warp grid).
    for rows in (0, 1, 3, 31, 126, 128, 144, 3986, 4027):
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
    (the task rules allow FP summation-order differences). The bitwise status of
    every reduction variant is printed for information.
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

    def compare(actual, expected, label, strict=True):
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
        if strict:
            assert scale_rel <= 2.0**-7, (label, "scale", scale_rel)
            assert value_ok, (label, "dequantized value")
        return bitwise

    for rows in (0, 1, 2, 3, 7, 31, 126, 127, 128, 144, 2047, 2048, 2049, 3986, 4027):
        x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * 2
        residual = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * 8
        if rows:
            x[0].zero_()
            residual[0].zero_()  # all-zero row: amax clamp 1e-10 path
        expected = reference(x, residual)
        work = residual.clone()
        q, scale = nq.add_rmsnorm_group_quant_fp8(x, work, weight, eps)
        compare((q, scale, work), expected, str(rows))

    # Information: which reduction variant reproduces the CuTe DSL kernel bit for bit.
    x = torch.randn(4027, 4096, device="cuda", dtype=torch.bfloat16) * 2
    residual = torch.randn(4027, 4096, device="cuda", dtype=torch.bfloat16) * 8
    expected = reference(x, residual)
    for variant in range(8):
        work = residual.clone()
        q, scale = nq.add_rmsnorm_group_quant_fp8(x, work, weight, eps, variant=variant)
        compare((q, scale, work), expected, f"variant{variant}", strict=variant == nq.FLASHINFER_EXACT_VARIANT)

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
        compare((*actual, work), expected, f"graph_{replay}")
    print("PASS: fused add-RMSNorm + group-128 FP8 quant within rounding of flashinfer fused_add_rmsnorm + per_token_group_quant (residual exact); zero/odd/Prefill rows; graph replay", flush=True)


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
                # The virtual-lane tree must additionally match the existing
                # fused kernel byte for byte, including both BF16 roundings.
                old_q, old_k = torch.empty_like(out), k.clone()
                apply_qknorm_rope_fp8_q(q, old_k, qw, kw, trig, pos, 1e-6, old_q)
                equal(out, old_q, label + "_exact_q")
                equal(ck, old_k.to(out.dtype), label + "_exact_k")
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
    print("PASS: Prefill prologue exact fused bytes, unfused reference, zero/odd/4k rows, paged + contiguous KV, graph replay and FA3 multi-request boundaries", flush=True)


if __name__ == "__main__":
    check_decode_silu_quant()
    check_rope_fp8_q()
    check_rope_fp8_q(with_norm=True)
    check_rope_fp8_kv()
    check_topk_mask()
    check_add_rmsnorm_group_quant()
    check_prefill_attention()
