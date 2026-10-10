"""Finite one-GPU probe: decode-verify attention, FA3 versus FlashInfer XQA.

Shapes follow the P3/D1 decode worker at TP4 on Qwen3-235B: 16 query heads
and 1 KV head per rank, head_dim 128, FP8 E4M3 KV with unit scales, 4
verify tokens per request (EAGLE3 steps3/topk1/draft4), ~4100-token
context. Each backend is called the way SGLang fc275674 calls it in target
verify; XQA variants test conventions suspected after the broken serving
run. Outputs are compared with an FP32 reference over the dequantized
cache. Nothing here is served; the JSON is diagnostic only.
"""
import json
import math
import sys
import time

import torch

OUT = sys.argv[1]
torch.manual_seed(2026)
dev = torch.device("cuda")
HQ, HK, D, QLEN = 16, 1, 128, 4
SCALE = 1.0 / math.sqrt(D)
CTX = 8192
# Failures of one backend/variant are recorded so the others still run.
PROBE_ERRORS = (RuntimeError, ValueError, TypeError, AssertionError, NotImplementedError, ImportError, AttributeError)
result = {"schema": "decode-verify-attention-probe/v1", "serving_adoption": False, "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cases": [], "errors": []}


def save():
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)


def timed(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # microseconds


def make_inputs(bs, prefix, page_size):
    seq = prefix + QLEN
    pages_per_req = (seq + page_size - 1) // page_size
    num_pages = bs * pages_per_req + 1
    k_fp8 = (torch.randn(num_pages * page_size, HK, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    v_fp8 = (torch.randn(num_pages * page_size, HK, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    # Shuffle page assignment so requests do not read contiguous memory.
    perm = torch.randperm(num_pages - 1, device=dev)[: bs * pages_per_req] + 1
    page_table = perm.view(bs, pages_per_req).to(torch.int32)
    q = (torch.randn(bs * QLEN, HQ, D, device=dev) * 0.5).to(torch.bfloat16)
    seq_lens = torch.full((bs,), seq, dtype=torch.int32, device=dev)
    return q, k_fp8, v_fp8, page_table, seq_lens, seq


def reference(q, k_fp8, v_fp8, page_table, seq, page_size, q_for_ref=None):
    bs = page_table.shape[0]
    qr = (q if q_for_ref is None else q_for_ref).float().view(bs, QLEN, HQ, D)
    out = torch.empty(bs, QLEN, HQ, D, device=dev)
    offs = torch.arange(page_size, device=dev)
    for b in range(bs):
        tok = (page_table[b].long().unsqueeze(1) * page_size + offs).reshape(-1)[:seq]
        k = k_fp8[tok, 0].float()  # [seq, D]
        v = v_fp8[tok, 0].float()
        s = torch.einsum("qhd,kd->qhk", qr[b], k) * SCALE  # [QLEN, HQ, seq]
        pos = torch.arange(seq, device=dev)
        limit = (seq - QLEN) + torch.arange(QLEN, device=dev)  # query i sees keys <= prefix+i
        mask = pos.unsqueeze(0) <= limit.unsqueeze(1)
        s = s.masked_fill(~mask.unsqueeze(1), float("-inf"))
        out[b] = torch.einsum("qhk,kd->qhd", torch.softmax(s, dim=-1), v)
    return out.view(bs * QLEN, HQ, D)


def err(o, ref):
    o = o.float().view_as(ref)
    return {"max_abs": float((o - ref).abs().max()), "rel_l2": float((o - ref).norm() / ref.norm())}


def xqa_mask(bs):
    words = (QLEN + 31) // 32
    causal = torch.arange(words * 32, device=dev).unsqueeze(0) <= torch.arange(QLEN, device=dev).unsqueeze(1)
    packed = (causal.view(QLEN, words, 32).to(torch.int64) << torch.arange(32, device=dev, dtype=torch.int64)).sum(-1).to(torch.int32)
    return packed.unsqueeze(0).expand(bs, QLEN, words).contiguous().view(torch.uint16)


def run_fa3(q, k_fp8, v_fp8, page_table, seq_lens, page_size):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    bs = page_table.shape[0]
    kc = k_fp8.view(-1, page_size, HK, D)
    vc = v_fp8.view(-1, page_size, HK, D)
    one = torch.ones(1, device=dev)
    descale = one.expand(bs, HK)
    cu_q = torch.arange(0, bs * QLEN + 1, QLEN, dtype=torch.int32, device=dev)
    q8 = q.to(torch.float8_e4m3fn)
    def call():
        return flash_attn_with_kvcache(q=q8, k_cache=kc, v_cache=vc, page_table=page_table, cache_seqlens=seq_lens, cu_seqlens_q=cu_q, cu_seqlens_k_new=None, max_seqlen_q=QLEN, softmax_scale=SCALE, causal=True, window_size=(-1, -1), softcap=0.0, k_descale=descale, v_descale=descale)
    return call, q8


def run_xqa(q, k_fp8, v_fp8, page_table, seq_lens, page_size, variant):
    import flashinfer
    from sglang.kernels.ops.attention.utils import canonicalize_stride
    bs = page_table.shape[0]
    ws = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=dev)
    kc = k_fp8.view(-1, page_size, HK, D)
    vc = v_fp8.view(-1, page_size, HK, D)
    layout = "HND"
    if variant == "nhd":
        layout = "NHD"
    else:
        kc, vc = canonicalize_stride(kc.permute(0, 2, 1, 3)), canonicalize_stride(vc.permute(0, 2, 1, 3))
    lens = seq_lens - QLEN if variant == "seq_without_window" else seq_lens
    mask = xqa_mask(bs)
    def call():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(query=q, kv_cache=(kc, vc), workspace_buffer=ws, block_tables=page_table, seq_lens=lens, max_seq_len=CTX, bmm1_scale=SCALE, bmm2_scale=1.0, window_left=-1, out_dtype=torch.bfloat16, q_len_per_req=QLEN, mask=mask, kv_layout=layout)
    return call


prefix = 4096
for bs in (16, 64, 32, 96):
    for page_size in (1, 64):
        case = {"batch": bs, "prefix": prefix, "page_size": page_size}
        try:
            q, k8, v8, pt, sl, seq = make_inputs(bs, prefix, page_size)
            check = bs in (16, 64)
            ref = reference(q, k8, v8, pt, seq, page_size) if check else None
            call, q8 = run_fa3(q, k8, v8, pt, sl, page_size)
            o = call()
            case["fa3_us"] = timed(call)
            if check:
                case["fa3_err_vs_bf16_q_ref"] = err(o, ref)
                case["fa3_err_vs_fp8_q_ref"] = err(o, reference(q, k8, v8, pt, seq, page_size, q_for_ref=q8))
            kv_bytes = bs * seq * HK * D * 2
            case["kv_bytes_per_call"] = kv_bytes
            case["fa3_kv_TBps"] = kv_bytes / case["fa3_us"] / 1e6
            if page_size == 64:
                for variant in ("sglang", "seq_without_window", "nhd"):
                    try:
                        t0 = time.time()
                        xcall = run_xqa(q, k8, v8, pt, sl, page_size, variant)
                        ox = xcall()
                        torch.cuda.synchronize()
                        case[f"xqa_{variant}_first_call_s"] = time.time() - t0
                        case[f"xqa_{variant}_us"] = timed(xcall)
                        case[f"xqa_{variant}_kv_TBps"] = kv_bytes / case[f"xqa_{variant}_us"] / 1e6
                        if check:
                            case[f"xqa_{variant}_err"] = err(ox, ref)
                    except PROBE_ERRORS as e:
                        case[f"xqa_{variant}_exception"] = repr(e)[:500]
        except PROBE_ERRORS as e:
            case["exception"] = repr(e)[:800]
        result["cases"].append(case)
        save()
        print(json.dumps(case), flush=True)
result["complete"] = True
save()
