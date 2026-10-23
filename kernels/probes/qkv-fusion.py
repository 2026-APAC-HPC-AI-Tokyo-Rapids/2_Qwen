"""Compare deployed QKV, the previous BN128 fusion, and a BN256 fusion."""
import ctypes
import json
import runpy
import statistics
from pathlib import Path

import deep_gemm
import torch
from sglang.kernels.ops.attention.rope import apply_qknorm_rope_fp8_prefill

ROOT = Path(__file__).resolve().parent
checks = runpy.run_path(str(ROOT / "check-firmus-decode-kernels.py"))
assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
torch.cuda.set_device(0)
torch.manual_seed(20261024)
torch.backends.cuda.matmul.allow_tf32 = False
lib = ctypes.CDLL(str(ROOT / "qkv-fusion.so"))


class Params(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in ("qw", "kw", "trig", "pos", "loc", "q", "k", "v", "kc", "vc")]


lib.qkv_run.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 5 + [Params, ctypes.c_void_p]
lib.qkv_run.restype = ctypes.c_int
assert lib.qkv_init() == 0
print(json.dumps({"device": torch.cuda.get_device_name(), "scope": "QKV GEMM + QK norm/RoPE/FP8 Q and contiguous/paged KV", "tiles_bm_bn_stages_cluster": [[128, 128, 5, 1], [128, 256, 3, 2]], "serving": False}), flush=True)
qw, kw = [(torch.randn(128, device="cuda") * .25 + 1).bfloat16() for _ in range(2)]
angles = torch.arange(8192, device="cuda").float()[:, None] * (1000000.0 ** (-torch.arange(0, 128, 2, device="cuda").float() / 128))
trig = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()


def outputs(rows):
    return [torch.full(shape, 3., device="cuda", dtype=torch.float8_e4m3fn) for shape in ((rows, 64, 128), (rows, 4, 128), (rows, 4, 128), (8256, 4, 128), (8256, 4, 128))]


def inputs(rows, seed):
    lhs, rhs, packed = checks["dense_inputs"](rows, 9216, 4096, seed)
    pos = torch.randint(0, 8192, (rows,), device="cuda", dtype=torch.int64)
    loc = torch.randperm(8192, device="cuda")[:rows] + 1
    loc[-min(rows, 7):] = 0
    return lhs, rhs, packed, pos, loc


def post(data, out):
    _, _, packed, pos, loc = data
    q, k, v = packed.view(-1, 72, 128).split([64, 4, 4], dim=1)
    oq, ck, cv, kc, vc = out
    apply_qknorm_rope_fp8_prefill(q, k, qw, kw, trig, pos, 1e-6, oq, v, kc, vc, loc, ck, cv)


def native(data, out, mode):
    (a, sa), (b, sb), packed, pos, loc = data
    p = Params(*(tensor.data_ptr() for tensor in (qw, kw, trig, pos, loc, *out)))
    result = lib.qkv_run(mode, a.shape[0], sa.stride(1), *(tensor.data_ptr() for tensor in (a, sa, b, sb, packed)), p, torch.cuda.current_stream().cuda_stream)
    assert result == 0, result


def call(data, out, mode):
    if mode == "stock":
        deep_gemm.fp8_gemm_nt(data[0], data[1], data[2], compiled_dims="nk")
        post(data, out)
    elif mode == "head128":
        native(data, out, 0)
        post(data, out)
    elif mode == "old_fused":
        native(data, out, 1)
    elif mode == "head256":
        native(data, out, 2)
        post(data, out)
    elif mode == "fused":
        native(data, out, 3)
    elif mode == "gemm":
        deep_gemm.fp8_gemm_nt(data[0], data[1], data[2], compiled_dims="nk")
    else:
        post(data, out)


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def compare(actual, expected, label, exact=False):
    for i, (a, b) in enumerate(zip(actual, expected)):
        checks["check_numerics"](a, b, f"{label}/{i}")
        if exact:
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (label, i, "not bitwise equal")


# Tail tiles, normal Prefill sizes, changed inputs and cache slot-zero masking.
for rows in (129, 3986, 4027, 4096):
    data = inputs(rows, rows + 91)
    stock, head, fused = [outputs(rows) for _ in range(3)]
    call(data, stock, "stock")
    baseline = data[2].clone()
    # Independent FP32 reference on sampled rows, without a full 4096-row FP32 GEMM.
    indices = torch.tensor(sorted({0, rows // 2, rows - 1}), device="cuda")
    a, sa = data[0]
    b, sb = data[1]
    reference = (a[indices].float() * sa[indices].repeat_interleave(128, 1)) @ (b.float() * sb.repeat_interleave(128, 0).repeat_interleave(128, 1)).T
    for plain_mode, fused_mode in (("head128", "old_fused"), ("head256", "fused")):
        call(data, head, plain_mode)
        checks["dense_errors"](data[2], baseline.float())
        checks["dense_errors"](data[2][indices], reference)
        call(data, fused, fused_mode)
        compare(fused, head, f"rows={rows}/{fused_mode}/same_gemm", exact=True)
        compare(fused, stock, f"rows={rows}/{fused_mode}/stock")
    if rows in (129, 4027):
        graph = capture(lambda data=data, fused=fused: call(data, fused, "fused"))
        for replay in range(2):
            a.copy_((-a.float()).to(a.dtype))
            data[3].copy_(torch.randint_like(data[3], 0, 8192))
            data[4].copy_(data[4].roll(3))
            for out in (stock, head, fused):
                for tensor in out:
                    tensor.fill_(3.)
            call(data, stock, "stock")
            call(data, head, "head256")
            graph.replay()
            compare(fused, head, f"rows={rows}/graph={replay}", exact=True)
            compare(fused, stock, f"rows={rows}/stock_graph={replay}")
    print(f"CHECK rows={rows} old_and_new_epilogue_exact=true independent_dense_reference=true", flush=True)
    del data, stock, head, fused, reference, baseline, a, sa, b, sb
torch.cuda.synchronize()

for rows in (3986, 4027, 4096):
    data_sets = [inputs(rows, 5200 + i) for i in range(4)]
    out_sets = [outputs(rows) for _ in data_sets]
    modes = ("stock", "head128", "old_fused", "head256", "fused", "gemm", "post")
    graphs = {}
    repeats = 12
    for mode in modes:
        def batch(mode=mode, repeats=repeats, data_sets=data_sets, out_sets=out_sets):
            for i in range(repeats):
                call(data_sets[i % 4], out_sets[i % 4], mode)
        graphs[mode] = capture(batch)
        for _ in range(3):
            graphs[mode].replay()
    samples = {mode: [] for mode in modes}
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for replay in range(25):
        for mode in modes if replay % 2 == 0 else modes[::-1]:
            start.record()
            graphs[mode].replay()
            end.record()
            end.synchronize()
            samples[mode].append(start.elapsed_time(end) * 1000 / repeats)
    result = {mode: statistics.median(values) for mode, values in samples.items()}
    print(f"PASS rows={rows} baseline_median_us={result['stock']:.3f} candidate_median_us={result['fused']:.3f} saved_us={result['stock']-result['fused']:.3f} ratio={result['fused']/result['stock']:.5f} old_fused_us={result['old_fused']:.3f} head128_total_us={result['head128']:.3f} head256_total_us={result['head256']:.3f} stock_gemm_us={result['gemm']:.3f} prologue_us={result['post']:.3f} samples=25 rotating_sets=4", flush=True)
    del graphs, data_sets, out_sets
