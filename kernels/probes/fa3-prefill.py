"""Compare FA3 plus group-128 FP8 quantization with a fused native epilogue."""
import ctypes
import itertools
import json
import runpy
import statistics
from pathlib import Path

import deep_gemm
import torch
from sgl_kernel.flash_attn import flash_attn_varlen_func
from sglang.kernels.ops.quantization import per_token_group_quant

assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(20261024)
lib = ctypes.CDLL(str(Path(__file__).with_name("fa3-prefill.so")))
lib.prefill_run.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 9 + [ctypes.c_int, ctypes.c_void_p]
lib.prefill_run.restype = ctypes.c_int
assert lib.prefill_init() == 0


def inputs(lengths):
    rows = sum(lengths)
    q, k, v = [torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn) for heads in (64, 4, 4)]
    cu = torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty((64, rows), device="cuda", dtype=torch.float32)
    metadata = torch.zeros(17, device="cuda", dtype=torch.int32)
    fp8 = torch.full_like(q, 3.)
    scale = torch.full((64, (rows+3)//4*4), -1., device="cuda").T[:rows]
    return lengths, q, k, v, cu, out, lse, metadata, fp8, scale


def call(data, mode):
    lengths, q, k, v, cu, out, lse, metadata, fp8, scale = data
    if mode in ("stock", "stock_quant"):
        flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(lengths), max_seqlen_k=max(lengths), causal=True, softmax_scale=128**-.5, num_splits=0, out=out)
    elif mode != "quant":
        rc = lib.prefill_run(mode in ("n192", "n192_quant", "fp8"), q.shape[0], len(lengths), max(lengths), *(t.data_ptr() for t in (q, k, v, cu, out, lse, metadata)), fp8.data_ptr() if mode == "fp8" else None, scale.data_ptr() if mode == "fp8" else None, scale.stride(1), torch.cuda.current_stream().cuda_stream)
        assert rc == 0, rc
    if mode in ("stock_quant", "n192_quant", "quant"):
        per_token_group_quant(out.view(-1,8192), fp8.view(-1,8192), scale, 128)
    if mode in ("stock_quant", "n192_quant", "quant", "fp8"):
        return fp8, scale
    return out


def compare(a, b, label):
    a, b = a.float(), b.float()
    assert a.isfinite().all() and b.isfinite().all(), label
    e = a-b
    l2 = (e.norm()/b.norm().clamp_min(1e-12)).item()
    scaled = (e.abs().max()/b.abs().max().clamp_min(1e-12)).item()
    print(json.dumps(dict(check=label, relative_l2=l2, max_scaled=scaled)), flush=True)
    assert l2 <= .05 and scaled <= .25, label


def dequant(pair):
    q, s = pair
    return q.float()*s[:,:,None]


def compare_quant(actual, expected, label):
    q,s = actual
    eq,es = expected
    assert torch.isfinite(s).all() and (s>0).all(), label
    sr = ((s-es).abs()/es).max().item()
    mismatch = int((q.view(torch.uint8)!=eq.view(torch.uint8)).sum())
    print(json.dumps(dict(check=label, scale_relative=sr, fp8_codes_differing=mismatch)),flush=True)
    assert sr == 0 and mismatch == 0,label


def reference(data):
    lengths, q, k, v, *_ = data
    out, indices, offset = [], [], 0
    for n in lengths:
        sample = torch.tensor(sorted({0, min(127, n-1), n//2, n-1}), device="cuda")
        qs = q[offset+sample].float().reshape(-1,4,16,128).permute(1,2,0,3)
        ks = k[offset:offset+n].float().permute(1,2,0)
        vs = v[offset:offset+n].float().permute(1,0,2)
        score = torch.einsum("hgsd,hdk->hgsk",qs,ks) * (128**-.5)
        score.masked_fill_(torch.arange(n,device="cuda")[None,:]>sample[:,None],float("-inf"))
        out.append(torch.einsum("hgsk,hkd->shgd",score.softmax(-1),vs).reshape(-1,64,128))
        indices.append(offset+sample)
        offset += n
    return torch.cat(indices), torch.cat(out)


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): fn()
    return graph


for lengths in ([1], [17,13], [129,257], [3986], [4027], [4096]):
    data = inputs(lengths)
    stock = call(data,"stock").clone()
    indices, expected = reference(data)
    compare(stock[indices],expected,f"stock_fp32/{lengths}")
    for mode in ("n224","n192"):
        out = call(data,mode)
        compare(out,stock,f"{mode}_stock/{lengths}")
        compare(out[indices],expected,f"{mode}_fp32/{lengths}")
    quant_ref = tuple(x.clone() for x in call(data,"n192_quant"))
    data[5].fill_(123.)
    call(data,"fp8")
    assert (data[5] == 123.).all(), "fused path must omit the BF16 global store"
    compare_quant(data[8:10],quant_ref,f"fp8_same_gemm/{lengths}")
    compare(dequant(data[8:10])[indices],expected,f"fp8_fp32/{lengths}")
    if lengths == [4027]:
        graph = capture(lambda: call(data,"fp8"))
        for replay in range(2):
            for x in data[1:4]: x.copy_(torch.randn(x.shape,device="cuda",dtype=torch.bfloat16).to(x.dtype))
            quant_ref = tuple(x.clone() for x in call(data,"n192_quant"))
            graph.replay()
            compare_quant(data[8:10],quant_ref,f"graph/{replay}")
            indices, expected = reference(data)
            compare(dequant(data[8:10])[indices],expected,f"graph_fp32/{replay}")

data = inputs([17,13])
data[3].zero_()
quant_ref = tuple(x.clone() for x in call(data,"n192_quant"))
compare_quant(call(data,"fp8"),quant_ref,"zero_value_scale_floor")

for rows in (3986,4027,4096):
    data = [inputs([rows]) for _ in range(4)]
    modes = ("stock_quant","n192_quant","fp8","quant")
    graphs = {}
    for mode in modes:
        def batch(mode=mode):
            for i in range(8): call(data[i%4],mode)
        graphs[mode] = capture(batch)
        for _ in range(3): graphs[mode].replay()
    samples = {mode: [] for mode in modes}
    start,end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for i in range(25):
        for mode in modes if i%2==0 else modes[::-1]:
            start.record(); graphs[mode].replay(); end.record(); end.synchronize()
            samples[mode].append(start.elapsed_time(end)*1000/8)
    b,control,c,quant = [statistics.median(samples[mode]) for mode in modes]
    print(f"PASS kernel=fa3_fp8 rows={rows} baseline_median_us={b:.3f} candidate_median_us={c:.3f} native_n192_quant_us={control:.3f} quant_us={quant:.3f} saved_us={b-c:.3f} ratio={c/b:.5f} samples=25 rotating_sets=4",flush=True)
    del data,graphs

# Verify the direct FP8 buffer and column-major scales with the real consumer.
checks = runpy.run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
for rows in (3986,4027,4096):
    data = [inputs([rows]) for _ in range(4)]
    dense = [checks["dense_inputs"](rows,4096,8192,6100+i) for i in range(4)]
    def attention_o(i, mode):
        q,s = call(data[i],mode)
        deep_gemm.fp8_gemm_nt((q.view(rows,8192),s),dense[i][1],dense[i][2])
        return dense[i][2]
    expected = attention_o(0,"n192_quant").clone()
    actual = attention_o(0,"fp8").clone()
    assert torch.equal(actual,expected), "O GEMM must match the unfused N192 path exactly"
    stock = attention_o(0,"stock_quant").clone()
    compare(actual,stock,f"fa3_o_stock/{rows}")
    q,s = call(data[0],"fp8")
    indices = torch.tensor([0,rows//2,rows-1],device="cuda")
    fp32 = checks["dense_reference"]((q.view(rows,8192)[indices],s[indices]),dense[0][1])
    print(dict(check="fa3_o_fp32_gemm", rows=rows, **checks["dense_errors"](actual[indices],fp32)),flush=True)
    if rows == 4027:
        graph = capture(lambda: attention_o(0,"fp8"))
        for x in data[0][1:4]: x.copy_((-x.float()).to(x.dtype))
        expected = attention_o(0,"n192_quant").clone()
        graph.replay()
        assert torch.equal(dense[0][2],expected), "O GEMM changed-input graph"
    modes = ("stock_quant","n192_quant","fp8")
    graphs = {}
    for mode in modes:
        def batch(mode=mode):
            for i in range(8): attention_o(i%4,mode)
        graphs[mode] = capture(batch)
        for _ in range(3): graphs[mode].replay()
    samples = {mode: [] for mode in modes}
    for i in range(25):
        for mode in modes if i%2==0 else modes[::-1]:
            start.record(); graphs[mode].replay(); end.record(); end.synchronize()
            samples[mode].append(start.elapsed_time(end)*1000/8)
    b,control,c = [statistics.median(samples[mode]) for mode in modes]
    print(f"PASS kernel=fa3_fp8_o rows={rows} baseline_median_us={b:.3f} candidate_median_us={c:.3f} native_n192_quant_o_us={control:.3f} saved_us={b-c:.3f} ratio={c/b:.5f} samples=25 rotating_sets=4",flush=True)
    del data,dense,graphs
