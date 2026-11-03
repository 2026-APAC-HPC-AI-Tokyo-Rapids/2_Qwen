"""Use a full CTA per input row to shorten fused norm/quant dependency chains."""
import os
import runpy
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
import sglang.kernels.ops.layernorm.add_rmsnorm_group_quant as nq


@triton.jit
def norm_quant(X, R, W, Q, S, stride, eps: tl.constexpr, SQUARE_FMA: tl.constexpr = False):
    row = tl.program_id(0)
    col = tl.arange(0, 4096)
    # Keep the unrounded FP32 sum for normalization, BF16 for residual storage.
    h = tl.load(X + row * 4096 + col).to(tl.float32) + tl.load(R + row * 4096 + col).to(tl.float32)
    tl.store(R + row * 4096 + col, h)
    if SQUARE_FMA:
        # Static split/reshape selects this lane's registers without a gather.
        h0_8, h8_8 = tl.split(tl.trans(tl.reshape(h, (2, 2048))))
        h0_4, h4_4 = tl.split(tl.trans(tl.reshape(h0_8, (2, 1024))))
        h0_2, h2_2 = tl.split(tl.trans(tl.reshape(h0_4, (2, 512))))
        v0, v1 = tl.split(tl.trans(tl.reshape(h0_2, (2, 256))))
        v2, v3 = tl.split(tl.trans(tl.reshape(h2_2, (2, 256))))
        h4_2, h6_2 = tl.split(tl.trans(tl.reshape(h4_4, (2, 512))))
        v4, v5 = tl.split(tl.trans(tl.reshape(h4_2, (2, 256))))
        v6, v7 = tl.split(tl.trans(tl.reshape(h6_2, (2, 256))))
        h8_4, h12_4 = tl.split(tl.trans(tl.reshape(h8_8, (2, 1024))))
        h8_2, h10_2 = tl.split(tl.trans(tl.reshape(h8_4, (2, 512))))
        v8, v9 = tl.split(tl.trans(tl.reshape(h8_2, (2, 256))))
        v10, v11 = tl.split(tl.trans(tl.reshape(h10_2, (2, 256))))
        h12_2, h14_2 = tl.split(tl.trans(tl.reshape(h12_4, (2, 512))))
        v12, v13 = tl.split(tl.trans(tl.reshape(h12_2, (2, 256))))
        v14, v15 = tl.split(tl.trans(tl.reshape(h14_2, (2, 256))))
        parts = (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
        sums = v0 * v0
        for i in tl.static_range(1, 16):
            value = parts[i]
            sums = tl.fma(value, value, sums)
        square_sum = tl.sum(sums, 0)
    else:
        square_sum = tl.sum(h * h, 0)
    rstd = tl.rsqrt(square_sum / 4096.0 + eps)
    y = ((h * rstd) * tl.load(W + col).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    groups = tl.reshape(y, (32, 128))
    amax = tl.maximum(tl.max(tl.abs(groups), 1), 1.0e-10)
    gain = tl.div_rn(448.0, amax)
    q = tl.minimum(groups * gain[:, None], 448.0).to(tl.float8e4nv)
    tl.store(Q + row * 4096 + col, tl.reshape(q, (4096,)))
    tl.store(S + row + tl.arange(0, 32) * stride, amax * (1.0 / 448.0))


def candidate_out(x, residual, weight, eps, q, scale):
    if x.shape[0]:
        norm_quant[(x.shape[0],)](x, residual, weight, q, scale, scale.stride(1), eps, SQUARE_FMA=os.environ.get("SGLANG_PREFILL_NORM_SQUARE_FMA", "0") == "1", num_warps=8, enable_fp_fusion=False)


def candidate(x, residual, weight, eps):
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    rows = x.shape[0]
    scale = torch.empty((32, triton.cdiv(rows, 4) * 4), device=x.device).T[:rows]
    candidate_out(x, residual, weight, eps, q, scale)
    return q, scale


assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
baseline_out = nq.add_rmsnorm_group_quant_fp8_out
# Reuse the existing gate against FlashInfer + group quant, including changed-input graphs.
nq.add_rmsnorm_group_quant_fp8 = candidate
checks = runpy.run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
checks["check_add_rmsnorm_group_quant"]()
torch.manual_seed(20261024)
weight = (torch.randn(4096, device="cuda") * .1 + 1).bfloat16()

for rows in (3986, 4027, 4096):
    data = []
    for _ in range(4):
        x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
        r = torch.randn_like(x) * 4
        q, s = candidate(x, r, weight, 1e-6)
        data.append((x, -x, r, q, s))
    graphs = {}
    for name, fn in (("baseline", baseline_out), ("candidate", candidate_out)):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for x, nx, r, q, s in data:
                fn(x, r, weight, 1e-6, q, s)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for x, nx, r, q, s in data:
                fn(x, r, weight, 1e-6, q, s)
                fn(nx, r, weight, 1e-6, q, s)
        graphs[name] = graph
        for _ in range(3):
            graph.replay()
    samples = {name: [] for name in graphs}
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for i in range(25):
        for name in ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline"):
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 8)
    b, c = [statistics.median(samples[name]) for name in ("baseline", "candidate")]
    print(f"PASS kernel=input_norm_quant rows={rows} baseline_median_us={b:.3f} candidate_median_us={c:.3f} saved_us={b-c:.3f} ratio={c/b:.5f} samples=25 rotating_sets=4", flush=True)
    del graphs, data
