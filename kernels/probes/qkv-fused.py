"""One-H100 paired probe of the PP4 direct-register QKV producer."""

import json
import statistics
import time

import deep_gemm
import torch
from sglang.kernels.ops.attention.qkv_fused_prefill import qkv_fused_prefill
from sglang.kernels.ops.attention.rope import apply_qknorm_rope_fp8_prefill_fp32
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8


def fp64_codes(x):
    code = torch.arange(127, device=x.device, dtype=torch.int64)
    exp, frac = code >> 3, code & 7
    values = torch.where(exp == 0, frac.double() * 2.0**-9, (1 + frac.double() / 8) * torch.exp2(exp.double() - 7))
    mid = (values[:-1] + values[1:]) / 2
    mag = x.abs().contiguous()
    index = torch.bucketize(mag, mid)
    tie = (index < 126) & (mag == mid[index.clamp_max(125)]) & ((index & 1) == 1)
    index += tie
    sign = torch.signbit(x).to(torch.uint8) << 7
    codes = index.to(torch.uint8) | sign
    codes = torch.where(mag > 464, 127 | sign, codes)
    return torch.where(torch.isnan(x), 127, codes).to(torch.uint8)


def make_case(m):
    torch.manual_seed(731 + m)
    a, sa = sglang_per_token_group_quant_fp8(torch.randn((m, 4096), device="cuda", dtype=torch.bfloat16) * 0.7, 128, column_major_scales=True, scale_tma_aligned=True)
    raw_w = torch.randn((72, 128, 32, 128), device="cuda") * 0.02
    sw = raw_w.abs().amax((1, 3)).clamp_min(1e-10) / 448
    w = (raw_w / sw[:, None, :, None]).reshape(9216, 4096).to(torch.float8_e4m3fn)
    qw, kw = [(torch.randn(128, device="cuda") * 0.2 + 1).bfloat16() for _ in range(2)]
    angle = torch.arange(8192, device="cuda").float()[:, None] * 1000000.0 ** (-torch.arange(0, 128, 2, device="cuda").float() / 128)
    trig = torch.cat((angle.cos(), angle.sin()), -1).contiguous()
    dtype = torch.int64 if m == 7988 else torch.int32
    pos = torch.randint(0, 8192, (m,), device="cuda", dtype=dtype)
    loc = torch.randperm(m, device="cuda").to(dtype) + 1
    loc[0] = 0
    dense = torch.empty((m, 9216), device="cuda", dtype=torch.bfloat16)
    return a, sa, w, sw, qw, kw, trig, pos, loc, dense


def outputs(m):
    contiguous = tuple(torch.full((m + 256, heads, 128), 3, device="cuda", dtype=torch.float8_e4m3fn)[:m] for heads in (64, 4, 4))
    caches = tuple(torch.full((m + 17, 4, 128), 3, device="cuda", dtype=torch.float8_e4m3fn) for _ in range(2))
    return (*contiguous, *caches)


def call(data, out, mode):
    a, sa, w, sw, qw, kw, trig, pos, loc, dense = data
    q, k, v, kc, vc = out
    if mode == "fused":
        qkv_fused_prefill(a, sa, w, sw, qw, kw, trig, pos, loc, 1e-6, q, k, v, kc, vc)
        return
    if mode in ("stock", "gemm"):
        deep_gemm.fp8_gemm_nt((a, sa), (w, sw), dense, compiled_dims="nk")
    if mode in ("stock", "post"):
        dq, dk, dv = dense.split((8192, 512, 512), -1)
        apply_qknorm_rope_fp8_prefill_fp32(dq.view(-1, 64, 128), dk.view(-1, 4, 128), qw, kw, trig, pos, 1e-6, q, dv.view(-1, 4, 128), kc, vc, loc, k, v)


def reference(data):
    a, sa, w, sw, qw, kw, trig, pos, _loc, _ = data
    lhs = a.double() * sa.double().repeat_interleave(128, 1)
    rhs = w.double() * sw.double().repeat_interleave(128, 0).repeat_interleave(128, 1)
    dense = lhs @ rhs.T
    del lhs, rhs
    parts = dense.split((8192, 512, 512), -1)
    result = []
    for x, weight in zip(parts[:2], (qw, kw), strict=True):
        x = x.reshape(a.shape[0], -1, 128)
        rho = torch.rsqrt(x.square().mean(-1) + 1e-6)
        y = x * weight.double()
        u, v = y[..., :64], y[..., 64:]
        c = trig[pos.long(), :64].double()[:, None]
        s = trig[pos.long(), 64:].double()[:, None]
        result.append(torch.cat(((u * c - v * s) * rho[..., None], (u * s + v * c) * rho[..., None]), -1))
    result.append(parts[2].reshape(a.shape[0], 4, 128).clone())
    return result


def check(data, candidate, control, exact, label):
    rows = data[0].shape[0]
    metrics = {}
    passed = True
    for i, name in enumerate(("q", "k", "v", "k_cache", "v_cache")):
        target = exact[i if i < 3 else i - 2]
        got, old = candidate[i], control[i]
        if i < 3:
            for tensor in (candidate[i], control[i]):
                assert torch.all(tensor._base[rows:].float() == 3), "M-tail guard overwritten"
        if i >= 3:
            slots = data[8].long()
            got, old, target = got[slots[1:]], old[slots[1:]], target[1:]
            touched = torch.zeros(rows + 17, device="cuda", dtype=torch.bool)
            touched[slots[1:]] = True
            for tensor in (candidate[i], control[i]):
                assert torch.all(tensor[~touched].float() == 3), "untouched cache/slot zero modified"
            source = candidate[i - 2][1:]
            assert torch.equal(got.view(torch.uint8), source.view(torch.uint8)), "cache codes differ from contiguous"
        code = fp64_codes(target)
        err = (got.double() - target).abs()
        old_err = (old.double() - target).abs()
        new_max, old_max = err.max().item(), old_err.max().item()
        new_mean, old_mean = err.mean().item(), old_err.mean().item()
        mismatch = (got.view(torch.uint8) != code).double().mean().item()
        old_mismatch = (old.view(torch.uint8) != code).double().mean().item()
        between = (got.view(torch.uint8) != old.view(torch.uint8)).double().mean().item()
        ok = bool(torch.isfinite(err).all()) and new_max <= old_max and new_mean <= old_mean and mismatch <= old_mismatch
        passed &= ok
        metrics[name] = {"fused_max_abs": new_max, "fp32_max_abs": old_max, "fused_mean_abs": new_mean, "fp32_mean_abs": old_mean, "fused_fp64_code_mismatch": mismatch, "fp32_fp64_code_mismatch": old_mismatch, "fused_fp32_code_mismatch": between, "passed": ok}
    print(json.dumps({"check": label, "rows": rows, "outputs": metrics, "passed": passed}), flush=True)
    return passed


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(12):
            fn()
    return graph


def timing(data, out):
    modes = ("stock", "fused", "gemm", "post")
    graphs = {mode: capture(lambda mode=mode: call(data, out, mode)) for mode in modes}
    samples = {mode: [] for mode in modes}
    for block in range(2):
        for r in range(12):
            order = modes if (r + block) % 2 == 0 else modes[::-1]
            for mode in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[mode].replay()
                end.record()
                end.synchronize()
                samples[mode].append(start.elapsed_time(end) * 1000 / 12)
    medians = {mode: statistics.median(x) for mode, x in samples.items()}
    ratio = medians["fused"] / medians["stock"]
    print(json.dumps({"rows": data[0].shape[0], "median_us": medians, "ratio": ratio, "samples_us": samples, "blocks": 2, "order": "interleaved_forward_reverse"}), flush=True)
    return ratio


def main():
    started = time.monotonic()
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_properties(0).multi_processor_count == 132
    print(json.dumps({"device": torch.cuda.get_device_name(), "torch": torch.__version__, "scope": "QKV plus fp32 writer, all five outputs, no serving", "reference": "FP64 GEMM from identical dequantized FP8 operands", "norm_weights": "BF16 random 1 +/- 0.2; no model files"}), flush=True)
    ratios = {}
    for m in (129, 3994, 7988):
        data = make_case(m)
        fused, stock = outputs(m), outputs(m)
        call(data, stock, "stock")
        call(data, fused, "fused")
        exact = reference(data)
        numeric = check(data, fused, stock, exact, "initial")
        if m != 129:
            ratios[m] = timing(data, fused)
        for _ in range(300):
            call(data, fused, "fused")
        torch.cuda.synchronize()
        numeric &= check(data, fused, stock, exact, "after_300_calls")
        assert numeric, f"numeric gate failed at M{m}"
        del data, fused, stock, exact
        torch.cuda.empty_cache()
    print(json.dumps({"passed": True, "ratios": ratios, "speed_gate": ratios[7988] <= 0.97, "elapsed_s": time.monotonic() - started}), flush=True)
    print(f"PASS rows=7988 ratio={ratios[7988]:.6f} numerics=true endurance_calls=300 speed_gate={ratios[7988] <= 0.97}", flush=True)


if __name__ == "__main__":
    main()
