"""Finite one-GPU probe: FA3 split-KV and GQA packing for decode-verify attention.

Shapes follow the decode worker at TP4 on Qwen3-235B: 16 query heads and 1
KV head per rank, head_dim 128, FP8 E4M3 KV with unit scales, 4 verify
tokens per request (EAGLE3 steps3/topk1/draft4), 4096-token prefix,
randomly scattered pages. FA3 is called as SGLang fc275674 calls it for
target verify (sgl_kernel, the serving default), except that num_splits and
pack_gqa are swept instead of left at 0/None. The serving-like page table
is as wide as context-length 8192, and the auto setting is also replayed
from a CUDA graph as the decode worker does. L2 is flushed before every
timed call because each layer reads a different KV cache. Outputs are
compared with an FP32 reference that uses the same FP8 query. Nothing here
is served; the JSON is diagnostic only.
"""
import json
import math
import statistics
import sys

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

OUT = sys.argv[1]
torch.manual_seed(2026)
dev = torch.device("cuda")
HQ, HK, D, QLEN = 16, 1, 128, 4
SCALE = 1.0 / math.sqrt(D)
CONTEXT_LENGTH = 8192
PREFIX = 4096
BATCHES = (32, 64, 72, 96, 112)
PAGE_SIZES = (1, 64)
SPLITS = (0, 1, 2, 3, 4, 5, 6, 8, 12, 16)
PACKS = (None, True, False)
# Failures of one setting are recorded so the others still run.
PROBE_ERRORS = (RuntimeError, ValueError, TypeError, AssertionError, NotImplementedError)
result = {"schema": "decode-verify-attention-probe/v2", "serving_adoption": False, "device": torch.cuda.get_device_name(0), "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "torch": torch.__version__, "prefix": PREFIX, "context_length": CONTEXT_LENGTH, "cases": []}
flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)


def save():
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)


def timed(fn, iters=40, warmup=5):
    for _ in range(warmup):
        fn()
    events = []
    for _ in range(iters):
        flush.zero_()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) * 1000.0 for s, e in events)  # microseconds


def make_inputs(bs, page_size):
    seq = PREFIX + QLEN
    pages_per_req = (seq + page_size - 1) // page_size
    num_pages = bs * pages_per_req + 1
    k_fp8 = (torch.randn(num_pages * page_size, HK, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    v_fp8 = (torch.randn(num_pages * page_size, HK, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    perm = torch.randperm(num_pages - 1, device=dev)[: bs * pages_per_req] + 1
    exact = perm.view(bs, pages_per_req).to(torch.int32)
    # Serving page tables are as wide as the context length; unused tail entries are zero.
    wide = torch.zeros(bs, CONTEXT_LENGTH // page_size, dtype=torch.int32, device=dev)
    wide[:, :pages_per_req] = exact
    q8 = (torch.randn(bs * QLEN, HQ, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    seq_lens = torch.full((bs,), seq, dtype=torch.int32, device=dev)
    return q8, k_fp8, v_fp8, exact, wide, seq_lens, seq


def reference(q8, k_fp8, v_fp8, page_table, seq, page_size):
    bs = page_table.shape[0]
    qr = q8.float().view(bs, QLEN, HQ, D)
    out = torch.empty(bs, QLEN, HQ, D, device=dev)
    offs = torch.arange(page_size, device=dev)
    pos = torch.arange(seq, device=dev)
    limit = (seq - QLEN) + torch.arange(QLEN, device=dev)  # query i sees keys <= prefix+i
    mask = pos.unsqueeze(0) <= limit.unsqueeze(1)
    for b in range(bs):
        tok = (page_table[b].long().unsqueeze(1) * page_size + offs).reshape(-1)[:seq]
        k = k_fp8[tok, 0].float()
        v = v_fp8[tok, 0].float()
        s = torch.einsum("qhd,kd->qhk", qr[b], k) * SCALE
        s = s.masked_fill(~mask.unsqueeze(1), float("-inf"))
        out[b] = torch.einsum("qhk,kd->qhd", torch.softmax(s, dim=-1), v)
    return out.view(bs * QLEN, HQ, D)


def rel_l2(o, ref):
    o = o.float().view_as(ref)
    return float((o - ref).norm() / ref.norm())


def fa3_call(q8, k_fp8, v_fp8, page_table, seq_lens, page_size, num_splits, pack_gqa):
    bs = page_table.shape[0]
    kc = k_fp8.view(-1, page_size, HK, D)
    vc = v_fp8.view(-1, page_size, HK, D)
    descale = torch.ones(1, device=dev).expand(bs, HK)
    cu_q = torch.arange(0, bs * QLEN + 1, QLEN, dtype=torch.int32, device=dev)
    def call():
        return flash_attn_with_kvcache(q=q8, k_cache=kc, v_cache=vc, page_table=page_table, cache_seqlens=seq_lens, cu_seqlens_q=cu_q, cu_seqlens_k_new=None, max_seqlen_q=QLEN, softmax_scale=SCALE, causal=True, window_size=(-1, -1), softcap=0.0, k_descale=descale, v_descale=descale, num_splits=num_splits, pack_gqa=pack_gqa)
    return call


def graph_replay(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    return graph, out


for page_size in PAGE_SIZES:
    for bs in BATCHES:
        case = {"batch": bs, "page_size": page_size, "settings": []}
        q8, k8, v8, exact, wide, sl, seq = make_inputs(bs, page_size)
        case["kv_bytes_per_call"] = bs * seq * HK * D * 2
        ref = reference(q8, k8, v8, exact, seq, page_size) if bs == 64 else None
        for table_name, table in (("wide", wide), ("exact", exact)):
            for num_splits in SPLITS:
                for pack_gqa in PACKS:
                    if table_name == "exact" and (num_splits, pack_gqa) != (0, None):
                        continue
                    setting = {"table": table_name, "num_splits": num_splits, "pack_gqa": pack_gqa}
                    try:
                        call = fa3_call(q8, k8, v8, table, sl, page_size, num_splits, pack_gqa)
                        o = call()
                        setting["us"] = timed(call)
                        setting["kv_TBps"] = case["kv_bytes_per_call"] / setting["us"] / 1e6
                        if ref is not None:
                            setting["rel_l2"] = rel_l2(o, ref)
                    except PROBE_ERRORS as e:
                        setting["exception"] = repr(e)[:400]
                    case["settings"].append(setting)
        try:
            graph, out = graph_replay(fa3_call(q8, k8, v8, wide, sl, page_size, 0, None))
            case["graph_auto_us"] = timed(graph.replay)
            if ref is not None:
                case["graph_auto_rel_l2"] = rel_l2(out, ref)
        except PROBE_ERRORS as e:
            case["graph_exception"] = repr(e)[:400]
        timed_settings = [s for s in case["settings"] if "us" in s and s["table"] == "wide"]
        best = min(timed_settings, key=lambda s: s["us"])
        auto = next(s for s in timed_settings if s["num_splits"] == 0 and s["pack_gqa"] is None)
        case["best"] = {"num_splits": best["num_splits"], "pack_gqa": best["pack_gqa"], "us": best["us"], "speedup_vs_auto": auto["us"] / best["us"]}
        result["cases"].append(case)
        save()
        print(json.dumps({k: case[k] for k in ("batch", "page_size", "best") if k in case} | {"auto_us": auto["us"], "graph_auto_us": case.get("graph_auto_us")}), flush=True)
result["complete"] = True
save()
