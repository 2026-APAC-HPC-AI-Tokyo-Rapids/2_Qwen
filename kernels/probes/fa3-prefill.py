"""Compare FA3 plus group-128 FP8 quantization with a fused native epilogue."""
import ctypes
import itertools
import json
import runpy
import statistics
import subprocess
import sys
import time
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
E4_MODES = ("e4", "e4-regs232", "e4-n192")
VARIANT_MODES = ("nobarrier", "regs232", "lazy", "lazy-counter", *E4_MODES)
libraries = {"control": lib}
for variant in VARIANT_MODES:
    path = Path(__file__).with_name(f"fa3-prefill-{variant}.so")
    if path.exists():
        handle = ctypes.CDLL(str(path))
        handle.prefill_run.argtypes = lib.prefill_run.argtypes
        handle.prefill_run.restype = ctypes.c_int
        assert handle.prefill_init() == 0
        libraries[variant] = handle


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
    variant = mode if mode in VARIANT_MODES else "control"
    fused = mode in ("fp8", "fp8_n224", "shape4096", "shape4x1024", "nobarrier", "regs232", "lazy", "lazy-counter", *E4_MODES)
    if mode in ("stock", "stock_quant"):
        flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(lengths), max_seqlen_k=max(lengths), causal=True, softmax_scale=128**-.5, num_splits=0, out=out)
    elif mode != "quant":
        rc = libraries[variant].prefill_run(mode in ("n192", "n192_quant", "fp8"), q.shape[0], len(lengths), max(lengths), *(t.data_ptr() for t in (q, k, v, cu, out, lse, metadata)), fp8.data_ptr() if fused else None, scale.data_ptr() if fused else None, scale.stride(1), torch.cuda.current_stream().cuda_stream)
        assert rc == 0, rc
    if mode in ("stock_quant", "n192_quant", "quant"):
        per_token_group_quant(out.view(-1,8192), fp8.view(-1,8192), scale, 128)
    if fused or mode in ("stock_quant", "n192_quant", "quant"):
        return fp8, scale
    return out


def compare(a, b, label):
    a, b = a.float(), b.float()
    assert a.isfinite().all() and b.isfinite().all(), label
    e = a-b
    l2 = (e.norm()/b.norm().clamp_min(1e-12)).item()
    scaled = (e.abs().max()/b.abs().max().clamp_min(1e-12)).item()
    print(json.dumps({"check": label, "relative_l2": l2, "max_scaled": scaled}), flush=True)
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
    print(json.dumps({"check": label, "scale_relative": sr, "fp8_codes_differing": mismatch}),flush=True)
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


def work_counts(lengths, block_n=224):
    # PackGQA: each M128 tile covers eight tokens and sixteen Q heads.
    tiles = sum((n + 7)//8 * 4 for n in lengths)
    iterations = sum(4 * sum((min((m+1)*8, n) + block_n-1)//block_n for m in range((n+7)//8)) for n in lengths)
    return iterations, tiles


def m1():
    began = time.monotonic()
    modes = ("shape4096", "shape4x1024", "n192", "fp8", "nobarrier", "regs232", "lazy", *E4_MODES)
    # n192 is the existing BF16 mode; fp8 is its fused epilogue.
    gates = {mode: [] for mode in modes}
    for lengths in ([1], [17,13], [129,257], [2048,2048], [3986], [4027], [4096], [1024]*4):
        data = inputs(lengths)
        stock = call(data, "stock").clone()
        indices, expected = reference(data)
        compare(stock[indices], expected, f"stock_fp32/{lengths}")
        exact = tuple(x.clone() for x in call(data, "stock_quant"))
        native192 = call(data, "n192").clone()
        compare(native192, stock, f"n192_stock/{lengths}")
        compare(native192[indices], expected, f"n192_fp32/{lengths}")
        quant192 = tuple(x.clone() for x in call(data, "n192_quant"))
        for mode in modes:
            if mode == "n192":
                compare(call(data, mode), stock, f"{mode}/{lengths}")
            else:
                data[5].fill_(123.)
                result = call(data, mode)
                assert (data[5] == 123.).all(), (mode, "fused BF16 global store")
                compare_quant(result, quant192 if mode in ("fp8", "e4-n192") else exact, f"{mode}/{lengths}")
                compare(dequant(result)[indices], expected, f"{mode}_fp32/{lengths}")
            gates[mode].append(str(lengths))
        compare_quant(call(data, "lazy-counter"), exact, f"counter/{lengths}")
        if lengths == [4027]:
            for mode in ("fp8", "nobarrier", "regs232", "lazy", *E4_MODES):
                graph = capture(lambda data=data, mode=mode: call(data, mode))
                for replay in range(2):
                    for x in data[1:4]:
                        x.copy_(torch.randn(x.shape, device="cuda", dtype=torch.bfloat16).to(x.dtype))
                    ref = tuple(x.clone() for x in call(data, "n192_quant" if mode in ("fp8", "e4-n192") else "stock_quant"))
                    graph.replay()
                    compare_quant(data[8:10], ref, f"changed_input/{mode}/{replay}")
    data = inputs([17,13])
    data[3].zero_()
    ref = tuple(x.clone() for x in call(data, "stock_quant"))
    for mode in ("shape4096", "nobarrier", "regs232", "lazy", "lazy-counter", *E4_MODES):
        compare_quant(call(data, mode), ref, "zero_v/" + mode)

    # O GEMM bitwise gate for every exact variant and for N192 versus its own quant path.
    checks = runpy.run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
    for rows in (3986,4027,4096):
        data = inputs([rows])
        weight = torch.randn(4096,8192,device="cuda",dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        ws = 10.**(torch.rand(32,64,device="cuda")*3-3)
        out = torch.empty(rows,4096,device="cuda",dtype=torch.bfloat16)
        def dense(mode, data=data, rows=rows, weight=weight, ws=ws, out=out):
            q, scale = call(data, mode)
            deep_gemm.fp8_gemm_nt((q.view(rows,8192),scale),(weight,ws),out)
            return out
        expected224 = dense("stock_quant").clone()
        expected192 = dense("n192_quant").clone()
        for mode in ("shape4096", "fp8", "nobarrier", "regs232", "lazy", *E4_MODES):
            assert torch.equal(dense(mode), expected192 if mode in ("fp8", "e4-n192") else expected224), ("O GEMM", mode, rows)
        q, scale = call(data, "lazy")
        selected = torch.tensor([0, rows//2, rows-1], device="cuda")
        expected_fp32 = checks["dense_reference"]((q.view(rows,8192)[selected], scale[selected]), (weight,ws))
        compare(dense("lazy")[selected], expected_fp32, f"lazy_O_fp32/{rows}")
        graph = capture(lambda dense=dense: dense("lazy"))
        for x in data[1:4]:
            x.copy_((-x.float()).to(x.dtype))
        expected_o = dense("stock_quant").clone()
        graph.replay()
        assert torch.equal(out, expected_o), "O GEMM changed input lazy"

    data = {shape: [inputs(lengths) for _ in range(4)] for shape, lengths in (("shape4096", [4096]), ("shape4x1024", [1024]*4))}
    graphs = {}
    for mode in modes:
        shape = "shape4x1024" if mode == "shape4x1024" else "shape4096"
        def batch(mode=mode, shape=shape):
            for i in range(8):
                call(data[shape][i%4], mode)
        graphs[mode] = capture(batch)
        for _ in range(3):
            graphs[mode].replay()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    samples = {mode: [] for mode in modes}
    clock_samples = []
    gpu_uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    def clock():
        return int(subprocess.check_output(["nvidia-smi", "--id="+gpu_uuid, "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"], text=True).strip())
    for i in range(25):
        if i in (0,12,24):
            clock_samples.append(clock())
        for mode in modes if i%2 == 0 else modes[::-1]:
            start.record()
            graphs[mode].replay()
            end.record()
            end.synchronize()
            samples[mode].append(start.elapsed_time(end)*1000/8)
    medians = {mode: statistics.median(values) for mode, values in samples.items()}
    i1,t1 = work_counts([4096])
    i2,t2 = work_counts([1024]*4)
    det = i1*t2-i2*t1
    x = (medians["shape4096"]*t2-medians["shape4x1024"]*t1)/det
    y = (i1*medians["shape4x1024"]-i2*medians["shape4096"])/det
    counter = libraries["lazy-counter"]
    counter.prefill_counts.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    assert counter.prefill_counts_reset() == 0
    for dataset in data["shape4096"]:
        call(dataset, "lazy-counter")
    counts = torch.empty(2, device="cuda", dtype=torch.int64)
    assert counter.prefill_counts(counts.data_ptr(), torch.cuda.current_stream().cuda_stream) == 0
    skip, total = counts.cpu().tolist()
    assert total > 0 and 0 <= skip <= total
    sm_clock = statistics.median(clock_samples)
    runtime = Path(__file__).resolve().parents[1]
    output = {"scope": "synthetic Q/K only; skip rate is a reference value", "samples": 25, "rotating_sets": 4,
                  "gpu_uuid": gpu_uuid, "sm_clock_mhz": sm_clock, "sm_clock_samples_mhz": clock_samples, "clock_method": "nvidia-smi sampled outside timed graphs; approximate cycles",
                  "X_us_per_iteration": x, "Y_us_per_tile": y, "Y_fraction_4096": y*t1/medians["shape4096"], "model_nonnegative": x>=0 and y>=0,
                  "counts": {"shape4096": [i1,t1], "shape4x1024": [i2,t2]},
                  "skip_rate": skip/total, "skip": skip, "total": total,
                  "ptxas": json.loads((runtime/"results"/"ptxas.json").read_text()),
                  "modes": [{"mode": mode, "median_us": medians[mode], "cycles": medians[mode]*sm_clock,
                              "samples_us": samples[mode], "gate": {"passed": True, "shapes": gates[mode], "fp8": "numeric BF16" if mode=="n192" else "bitwise", "o_gemm": "N192 quant control via fused fp8" if mode=="n192" else "N192 quant control" if mode in ("fp8", "e4-n192") else "N224 stock quant control"}} for mode in modes],
                  "elapsed_s": time.monotonic()-began}
    (runtime/"results"/"fa3-prefill-m1.json").write_text(json.dumps(output, indent=2))
    print("RESULT_JSON " + json.dumps(output), flush=True)
    print(f"PASS kernel=fa3_prefill_m1 modes={len(modes)} gates=passed result=fa3-prefill-m1.json", flush=True)


def profile():
    # Four matching native launches: skip the first three, profile the fourth.
    data = inputs([4096])
    for _ in range(4):
        call(data, "shape4096")
    torch.cuda.synchronize()
    print("PASS kernel=fa3_prefill_m2 control=n224_fused shape=4096 launches=4", flush=True)


if __name__ == "__main__":
    {"m1": m1, "profile": profile}[sys.argv[1] if len(sys.argv)>1 else "m1"]()
