"""Specialize N/K through DeepGEMM's existing API, retaining stock geometry."""
import runpy
import statistics
from pathlib import Path

import deep_gemm
import torch

assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
torch.backends.cuda.matmul.allow_tf32 = False
checks = runpy.run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))


def call(data, compiled_dims):
    deep_gemm.fp8_gemm_nt(*data, compiled_dims=compiled_dims)


def compare(data):
    call(data, "")
    stock = data[2].clone()
    call(data, "nk")
    metrics = checks["dense_errors"](data[2], stock.float())
    a, sa = data[0]
    b, sb = data[1]
    indices = torch.tensor(sorted({0, a.shape[0]//2, a.shape[0]-1}), device="cuda")
    expected = (a[indices].float()*sa[indices].repeat_interleave(128,1)) @ (b.float()*sb.repeat_interleave(128,0).repeat_interleave(128,1)).T
    checks["dense_errors"](data[2][indices], expected)
    return metrics


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): fn()
    return graph


for name, n, k in (("qkv",9216,4096),("o",4096,8192)):
    for rows in (3986,4027,4096):
        data = checks["dense_inputs"](rows,n,k,rows+93)
        print(dict(check=name, rows=rows, **compare(data)),flush=True)
        if rows == 4027:
            graph = capture(lambda: call(data,"nk"))
            for replay in range(2):
                a = data[0][0]
                a.copy_((-a.float()).to(a.dtype))
                call(data,"")
                expected = data[2].clone()
                graph.replay()
                checks["dense_errors"](data[2],expected.float())
    for rows in (3986,4027,4096):
        data = [checks["dense_inputs"](rows,n,k,5200+i) for i in range(4)]
        modes = ("","nk")
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
        b,c = [statistics.median(samples[mode]) for mode in modes]
        print(f"PASS kernel=dense_{name} rows={rows} baseline_median_us={b:.3f} candidate_median_us={c:.3f} saved_us={b-c:.3f} ratio={c/b:.5f} samples=25 rotating_sets=4",flush=True)
        del data,graphs
