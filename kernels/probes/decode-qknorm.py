"""One-GPU exact old/new Decode QK-norm/RoPE probe; timings are indicative."""
import json
import runpy
import statistics
from pathlib import Path

import torch
from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args
from sglang.kernels.ops.attention.rope import apply_qknorm_rope_fp8_kv

checks = runpy.run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
torch.cuda.set_device(0)
torch.manual_seed(20261024)
reference = load_jit("decode_qknorm_reference", cuda_files=["elementwise/rope-decode-baseline.cuh"], cuda_wrappers=[("run", f"QKNormRopeFP8Kernel<{make_cpp_args(is_arch_support_pdl())}>::run_store")]).run
# The general RoPE checks also run in the serving gate; within the 165 s probe budget keep only
# the exhaustive old/new conversion check and the old/new kernel comparison below (40065833135
# timed out at 90 s after the general checks).
checks["check_decode_fp8_conversion"]()
angles = torch.arange(8192, device="cuda").float()[:, None] * (1000000.0 ** (-torch.arange(0, 128, 2, device="cuda").float() / 128))
trig = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
qw, kw = [(torch.randn(128, device="cuda") * .25 + 1).bfloat16() for _ in range(2)]


def capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph


# 193 exercises the shared token-warp fallback outside the Decode launch cap.
for rows in (0, 1, 3, 31, 126, 128, 129, 132, 144, 145, 160, 192, 193):
    for dtype in (torch.int32, torch.int64):
        packed = torch.randn(rows, 72, 128, device="cuda", dtype=torch.bfloat16) * 4
        q, k, v = packed.split((64, 4, 4), dim=1)
        if dtype == torch.int64:
            q, k, v = [x.contiguous() for x in (q, k, v)]
        pos = torch.randint(0, 8192, (rows,), device="cuda", dtype=dtype)
        loc = torch.arange(rows, device="cuda", dtype=dtype) + 1
        loc[-min(rows, 7):] = 0
        outputs = [tuple(torch.full(shape, 3., device="cuda", dtype=torch.float8_e4m3fn) for shape in ((rows, 64, 128), (257, 4, 128), (257, 4, 128))) for _ in range(2)]

        def call(index, outputs=outputs, q=q, k=k, v=v, pos=pos, loc=loc):
            oq, kc, vc = outputs[index]
            (reference if index == 0 else apply_qknorm_rope_fp8_kv)(q, k, qw, kw, trig, pos, 1e-6, oq, v, kc, vc, loc)

        def compare(label, outputs=outputs, rows=rows, dtype=dtype):
            for name, a, b in zip(("q", "kc", "vc"), *outputs):
                assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (rows, dtype, label, name)
            for out in outputs:
                assert (out[1][0].float() == 3.).all() and (out[2][0].float() == 3.).all(), "reserved cache slot zero"

        original = [x.clone() for x in (q, k, v)]
        call(0)
        call(1)
        compare("eager")
        assert all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip((q, k, v), original)), "eager modified QKV input"
        graphs = [capture(lambda i=i: call(i)) for i in (0, 1)]
        for replay in range(2):
            for x in (q, k, v):
                x.copy_(torch.randn_like(x) * (replay + 1) * 8)
            pos.copy_(torch.randint_like(pos, 0, 8192))
            loc.copy_(torch.randperm(256, device="cuda")[:rows].to(dtype) + 1)
            loc[-min(rows, 1 + replay * 3):] = 0
            original = [x.clone() for x in (q, k, v)]
            for out in outputs:
                for tensor in out:
                    tensor.fill_(3.)
            for graph in graphs:
                graph.replay()
            compare(f"changed_graph_{replay}")
            assert all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip((q, k, v), original)), "QKV input modified"
        if rows in (128, 160, 192) and dtype == torch.int32:
            samples = [[], []]
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            for repeat in range(25):
                for i in (0, 1) if repeat % 2 == 0 else (1, 0):
                    start.record()
                    for _ in range(20):
                        graphs[i].replay()
                    end.record()
                    end.synchronize()
                    samples[i].append(start.elapsed_time(end) * 1000 / 20)
            print(json.dumps({"rows": rows, "baseline_median_us": statistics.median(samples[0]), "candidate_median_us": statistics.median(samples[1]), "samples": 25, "serving_score": False}), flush=True)
        print(f"CHECK rows={rows} ids={dtype} packed_or_contiguous=true old_new_bitwise_equal=true changed_graphs=2", flush=True)
print("PASS: Decode QK-norm/RoPE exact conversion, old/new kernel bytes, tails, cache slot zero and changed-input graphs", flush=True)
