"""Independent norm_quant accuracy, timing and changed-input endurance gate."""
import hashlib
import importlib
import json
import math
import os
import statistics
import time
from pathlib import Path

SCHEMA = "firmus-prefill-norm-square-fma/v1"
KEY = "SGLANG_PREFILL_NORM_SQUARE_FMA"
CALLS, ROUNDS, SECONDS = 1200, 12, 60
ROWS = (7988, 7989, 3994, 2049)
CASES = tuple(str(n) for n in ROWS) + ("zero", "extreme", "fp8-boundary")
# ~60 us at 7988 rows: 0.5 us is ~0.8%, below the conditional 6 us gain,
# but above event resolution; shorter shapes retain a 0.5% relative floor.
MARGIN_US, MARGIN_FRACTION = 0.5, 0.005


def validate(r):
    assert set(r["accuracy"]) == set(CASES), "missing numerical cases"
    for case in r["accuracy"].values():
        assert case["residual_exact"] is True and case["finite"] is True
        for key in ("max_abs_error", "fp8_code_mismatches"):
            c, t = (case[m][key] for m in ("control", "candidate"))
            assert type(c) in (float, int) and type(t) in (float, int) and math.isfinite(c) and math.isfinite(t) and 0 <= t <= c, "FP64 reference regression"
    assert set(r["timing"]) == {str(n) for n in ROWS}
    for shape in r["timing"].values():
        for order in ("forward", "reverse"):
            row = shape[order]
            assert set(row) == {"control", "candidate"}
            assert all(len(v) == ROUNDS and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in v) for v in row.values())
            c, t = (statistics.median(row[m]) for m in ("control", "candidate"))
            assert c - t >= max(MARGIN_US, MARGIN_FRACTION * c), "below norm timing margin"
    e = r["endurance"]
    assert e["calls"] == CALLS and e["input_versions"] == 2 and e["buffers"] == 2
    assert set(e["exact"]) == {"0", "1"} and all(v is True for v in e["exact"].values()), "endurance or restore mismatch"


def run(output, run_id, deadline, image):
    begin = time.monotonic()
    import torch

    nq = importlib.import_module("sglang.kernels.ops.layernorm.add_rmsnorm_group_quant")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.manual_seed(1127 + rank)
    device = torch.device("cuda", rank)
    result = {"schema": SCHEMA, "run_id": run_id, "rank": rank, "worker": output.name, "uuid": str(torch.cuda.get_device_properties(rank).uuid), "device": torch.cuda.get_device_name(rank), "image_sha256": image, "source_sha256": hashlib.sha256(Path(nq.__file__).read_bytes()).hexdigest(), "accuracy": {}, "timing": {}, "healthy": False}

    def allocate(x):
        return x.clone(), torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((32, (len(x) + 3) // 4 * 4), device=device).T[:len(x)]

    def launch(x, residual, w, target, candidate):
        r, q, s = target
        r.copy_(residual)
        nq.norm_quant[(len(x),)](x, r, w, q, s, s.stride(1), 1e-6, SQUARE_FMA=candidate, num_warps=8, enable_fp_fusion=False)

    def exact(left, right):
        return all(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)) for a, b in zip(left, right, strict=True))

    cached = {}
    for name in CASES:
        assert time.time() < deadline, "norm worker deadline"
        rows = int(name) if name.isdecimal() else 256
        x = torch.randn(rows, 4096, device=device).bfloat16()
        residual = (torch.randn_like(x.float()) * 4).bfloat16()
        w = (torch.randn(4096, device=device) * 0.1 + 1).bfloat16()
        if name == "zero":
            x.zero_(); residual.zero_()
        if name == "extreme":
            x[::4] *= 2.0 ** 50
            residual[::4] *= 2.0 ** 50
            x[1::4] *= 2.0 ** -60
            residual[1::4] *= 2.0 ** -60
            x[2::4] = -residual[2::4]
        if name == "fp8-boundary":
            # Midpoints of every finite positive E4M3 code, with both signs
            # and adjacent FP32 values, then rounded to actual BF16 inputs.
            codes = torch.arange(127, device=device, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
            mid = (codes[:-1] + codes[1:]) * .5 / 448
            values = torch.cat((mid, torch.nextafter(mid, torch.full_like(mid, math.inf)), torch.nextafter(mid, torch.full_like(mid, -math.inf))))
            x.copy_(values.repeat((rows * 4096 + len(values) - 1) // len(values))[:rows * 4096].reshape_as(x))
            x[:, 0] = 1
            x[1::2].neg_(); residual.zero_(); w.fill_(1)
        targets = {m: allocate(x) for m in ("control", "candidate")}
        for mode, target in targets.items():
            launch(x, residual, w, target, mode == "candidate")
        # Independent double precision reference through normalization and BF16
        # output rounding, then ideal FP8 codes, with the same group/format.
        h = x.double() + residual.double()
        y = h * torch.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * w.double()
        groups = y.bfloat16().double().reshape(rows, 32, 128)
        amax = groups.abs().amax(-1).clamp_min(1e-10)
        ref_codes = (groups * (448.0 / amax)[..., None]).to(torch.float8_e4m3fn).view(torch.uint8)
        metrics = {}
        for mode, (r, q, s) in targets.items():
            dequant = q.double().reshape(rows, 32, 128) * s.double()[..., None]
            metrics[mode] = {"max_abs_error": (dequant.reshape_as(y) - y).abs().max().item(), "fp8_code_mismatches": int((q.view(torch.uint8).reshape_as(ref_codes) != ref_codes).sum().item())}
        result["accuracy"][name] = {**metrics, "residual_exact": torch.equal(targets["control"][0].view(torch.uint8), targets["candidate"][0].view(torch.uint8)) and torch.equal(targets["control"][0].view(torch.uint8), h.bfloat16().view(torch.uint8)), "finite": all(torch.isfinite(q.float()).all().item() and torch.isfinite(s).all().item() for _, q, s in targets.values())}
        if name.isdecimal():
            cached[name] = (x, residual, w)
        del targets, h, y, groups, amax, ref_codes
    for name, (x, residual, w) in cached.items():
        targets = {m: allocate(x) for m in ("control", "candidate")}
        graphs = {}
        for mode, target in targets.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8):
                    launch(x, residual, w, target, mode == "candidate")
            graphs[mode] = graph
            graph.replay()
        samples = {order: {m: [] for m in targets} for order in ("forward", "reverse")}
        for _ in range(ROUNDS):
            assert time.time() < deadline, "norm timing deadline"
            for order, modes in (("forward", ("control", "candidate")), ("reverse", ("candidate", "control"))):
                for mode in modes:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda._sleep(100000)
                    start.record(); graphs[mode].replay(); end.record(); end.synchronize()
                    samples[order][mode].append(start.elapsed_time(end) * 1000 / 8)
        result["timing"][name] = samples
        del graphs, targets
    x, residual, w = cached["7988"]
    changed = (x, -x)
    targets = [allocate(x), allocate(x)]
    expected = [allocate(x), allocate(x)]
    for i in range(2):
        launch(changed[i], residual, w, expected[i], True)
    # Ordinary kernels, alternating changed inputs and target buffers. Reset is
    # part of each call, so no residual accumulation can hide stale input use.
    for i in range(CALLS):
        launch(changed[(i // 2) % 2], residual, w, targets[i % 2], True)
    torch.cuda.synchronize()
    checks = {}
    for i in range(2):
        checks[str(i)] = exact(targets[i], expected[1])
        launch(changed[0], residual, w, targets[i], True)
        torch.cuda.synchronize()
        checks[str(i)] &= exact(targets[i], expected[0])
    result["endurance"] = {"calls": CALLS, "input_versions": 2, "buffers": 2, "exact": checks}
    result.update(healthy=True, elapsed_seconds=time.monotonic() - begin, finished_at=time.time())
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("accuracy", "timing")}), flush=True)
