"""Gate and select two memory-bound Prefill kernels against their pins before model load.

Run under torchrun on the four GPUs of each Prefill worker (no process group: every
GPU checks, times and stresses alone); --aggregate reads all twelve GPUs.

moe_input: the pinned onepass_moe_input (one 64-row program per SM, about half the
HBM bandwidth) against onepass-split (onepass_split_moe_input: the same per-group
loop over half or a quarter of the column groups, two programs per SM, then a
finalize kernel). Gate, on the PP4 microbatch row counts and the 2048-row floor
(ROWS) with an all-zero and a large row: residual, FP8 codes and pad slots equal
onepass bit for bit; scales within MAX_SCALE_RELATIVE of onepass's and of the FP64
amax(|y|) / 448 (only the FP32 order of the partial sums differs); ids distinct, a
valid top-8 of the FP64 logits within two BF16 steps, the same set as onepass on at
least MIN_SAME_IDS of the rows, weights summing to one and within the BF16 logit
rounding of onepass's where the sets agree.

qknorm: the pinned FP32 Q/K norm writer against fp32-heads4 / fp32-heads8 (four or
eight heads per CTA; the position, cos/sin row and norm weight loaded once per CTA).
Gate: every byte of FP8 Q, contiguous K/V and paged K/V (slots never addressed
keep the sentinel) equals the fp32 writer's, on ROWS x int64/int32 positions with
zero, large, past-E4M3, NaN and Inf rows.

Timing: per path a CUDA graph of 24 calls at TIMED_ROWS (8064, the full two-request
microbatch, and 4032, one request), rotated for ROUNDS rounds. Endurance: none of
the candidates has served, so each runs ENDURANCE_CALLS eager calls back to back,
alternating the two timed shapes, and must reproduce its first outputs bit for bit
(a hang ends in the job's timeout and leaves no result).

--aggregate adopts a candidate only if all twelve GPUs reported, it passed every
gate case and its endurance run everywhere, saved max(MARGIN_US, MARGIN_FRACTION)
per layer at 8064 rows and was not slower at 4032 rows on every GPU; of the two
Q/K candidates the larger summed saving wins. The winners are appended to
prefill-kernels.env after the pins (the worker sources it, last line wins);
anything missing keeps the pins.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

HIDDEN, EXPERTS, TOPK, CAP, EPS = 4096, 128, 8, 8192, 1e-6
Q_HEADS, KV_HEADS, DIM, THETA = 64, 4, 128, 1000000.0
# Two 4032-token requests per PP4 microbatch (7988 = 2 x 3994 in older profiles), one request, the floor.
ROWS = (8064, 7988, 4032, 2049)
TIMED_ROWS, LAYERS, ROUNDS = (8064, 4032), 24, 12
MARGIN_US, MARGIN_FRACTION = 3.0, 0.02
ENDURANCE_CALLS = 600
# Partial sums of h^2 and logits added in another FP32 order: a few ulp on rstd.
MAX_SCALE_RELATIVE = 1e-5
MIN_SAME_IDS = 0.999
SENTINEL = 0x5A
# selection -> (pinned control, candidates, environment switch)
SELECTIONS = {"moe_input": ("onepass", ("onepass-split",), "SGLANG_PREFILL_MOE_INPUT"), "qknorm": ("fp32", ("fp32-heads4", "fp32-heads8"), "SGLANG_PREFILL_QKNORM")}


def cases():
    return {"moe_input": [str(rows) for rows in ROWS], "qknorm": [f"{rows}-{dtype}" for rows in ROWS for dtype in ("int64", "int32")]}


def bf16_step(torch, value):
    """Spacing of BF16 numbers at each row's largest |value| (normal range)."""
    return torch.exp2(torch.floor(torch.log2(value.abs().amax(-1, keepdim=True).clamp_min(2.0**-126))) - 7)


def moe_buffer(torch, device):
    return {"x": torch.empty(CAP, HIDDEN, dtype=torch.float8_e4m3fn, device=device), "sf": torch.empty(CAP, HIDDEN // 128, device=device), "ids": torch.empty(CAP, TOPK, dtype=torch.int64, device=device), "w": torch.empty(CAP, TOPK, device=device)}


def scramble(torch, buf):
    buf["x"].view(torch.uint8).fill_(0x55)
    buf["sf"].fill_(7.0)
    buf["ids"].fill_(77)
    buf["w"].fill_(5.0)


def moe_call(kernels, variant, weights, x, residual, buf):
    kernels[variant](x, residual, weights["norm"], EPS, weights["gate"], buf["x"], buf["sf"], buf["ids"], buf["w"])


def check_moe_input(torch, kernels, weights, bufs, x0, r0):
    """onepass and onepass-split on copies of (x0, r0); the candidate judged against both onepass and FP64."""
    rows = x0.shape[0]
    residuals = {}
    for variant, buf in bufs.items():
        scramble(torch, buf)
        residuals[variant] = r0.clone()
        moe_call(kernels, variant, weights, x0.clone(), residuals[variant], buf)
    torch.cuda.synchronize()
    ctrl, cand = bufs["onepass"], bufs["onepass-split"]
    h = (x0.float() + r0.float()).double()
    y = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + EPS) * weights["norm"].double()
    amax = y.view(rows, HIDDEN // 128, 128).abs().amax(-1)
    exact = y @ weights["gate"].double().T
    live = amax > 0
    step = bf16_step(torch, exact)
    ids, ctrl_ids = cand["ids"][:rows], ctrl["ids"][:rows]
    safe = ids.clamp(0, EXPERTS - 1)
    same = (ids.sort(-1).values == ctrl_ids.sort(-1).values).all(-1)
    weight_error = ((cand["w"][:rows].gather(1, ids.argsort(-1)) - ctrl["w"][:rows].gather(1, ctrl_ids.argsort(-1))).abs() - 2 * step - 1e-6)[same]
    scales, ctrl_scales = cand["sf"][:rows].double(), ctrl["sf"][:rows].double()
    result = {
        "residual_bitwise": torch.equal(residuals["onepass-split"].view(torch.int16), residuals["onepass"].view(torch.int16)),
        "x_bitwise": torch.equal(cand["x"].view(torch.uint8), ctrl["x"].view(torch.uint8)),
        "pad_scales_untouched": bool((cand["sf"][rows:] == 7.0).all().item()),
        "scale_max_relative_to_onepass": ((scales - ctrl_scales).abs() / ctrl_scales.clamp_min(1e-30))[live].max().item(),
        "scale_max_relative_to_fp64": ((scales - amax / 448).abs() / (amax / 448))[live].max().item(),
        "ids_in_range_distinct": bool(((ids >= 0) & (ids < EXPERTS)).all().item()) and bool((ids.sort(-1).values.diff(dim=-1) > 0).all().item()),
        "valid_topk": bool((exact.gather(1, safe).amin(-1, keepdim=True) >= exact.scatter(1, safe, float("-inf")).amax(-1, keepdim=True) - 2 * step).all().item()),
        "same_set_as_onepass": same.float().mean().item(),
        "weights_sum_error": (cand["w"][:rows].sum(-1) - 1).abs().max().item(),
        "weights_within_bf16_logit_rounding": bool((weight_error <= 0).all().item()),
        "pad_slots": torch.equal(cand["ids"][rows:], ctrl["ids"][rows:]) and torch.equal(cand["w"][rows:].view(torch.int32), ctrl["w"][rows:].view(torch.int32)) and bool((cand["ids"][rows:] == -1).all().item()),
    }
    result["passed"] = result["residual_bitwise"] and result["x_bitwise"] and result["pad_scales_untouched"] and result["scale_max_relative_to_onepass"] <= MAX_SCALE_RELATIVE and result["scale_max_relative_to_fp64"] <= MAX_SCALE_RELATIVE and result["ids_in_range_distinct"] and result["valid_topk"] and result["same_set_as_onepass"] >= MIN_SAME_IDS and result["weights_sum_error"] <= 1e-5 and result["weights_within_bf16_logit_rounding"] and result["pad_slots"]
    return result


def make_qknorm_inputs(torch, device, generator, rows, id_dtype):
    """Packed BF16 QKV rows, norm weights, the Qwen3 cos/sin cache, positions and cache slots."""
    qkv = (torch.randn(rows, Q_HEADS + 2 * KV_HEADS, DIM, device=device, generator=generator) * torch.rand(rows, 1, 1, device=device, generator=generator) * 4).to(torch.bfloat16)
    # Special rows: zeros (RMS floor), a large row, V past the E4M3 range, NaN and Inf in Q, K and V.
    qkv[0].zero_()
    qkv[1].mul_(512)
    qkv[2, Q_HEADS + KV_HEADS:, :8] = torch.tensor([500.0, -470.0, 464.0, -1000.0, 448.0, 65504.0, -0.0, 1e-3], device=device, dtype=torch.bfloat16)
    qkv[3, 5, 7] = float("nan")
    qkv[4, Q_HEADS + 1, 70] = float("inf")
    qkv[5, Q_HEADS + KV_HEADS + 2, 3] = float("nan")
    qkv[6, Q_HEADS + KV_HEADS + 1, 100] = float("-inf")
    weights = [(torch.rand(DIM, device=device, generator=generator) * 2.7 + 0.3).to(torch.bfloat16) for _ in range(2)]
    angles = torch.arange(8192, device=device).double()[:, None] * THETA ** (-torch.arange(0, DIM, 2, device=device).double() / DIM)
    trig = torch.cat((angles.cos(), angles.sin()), dim=-1).float().contiguous()
    half = rows // 2
    positions = torch.cat([torch.arange(half, device=device), torch.arange(rows - half, device=device)]).to(id_dtype)
    # Slots of a 2*rows pool in shuffled order; a few rows map to slot 0 (no paged write).
    slots = (torch.randperm(2 * rows, device=device, generator=generator)[:rows] + 1).to(id_dtype)
    slots[7:rows:997] = 0
    return {"qkv": qkv, "weights": weights, "trig": trig, "positions": positions, "slots": slots, "pool": 2 * rows + 1}


def make_qknorm_outputs(torch, device, rows, pool):
    fp8 = torch.float8_e4m3fn
    out = {"q": torch.empty(rows, Q_HEADS, DIM, device=device, dtype=fp8), "k": torch.empty(rows, KV_HEADS, DIM, device=device, dtype=fp8), "v": torch.empty(rows, KV_HEADS, DIM, device=device, dtype=fp8)}
    out["k_cache"], out["v_cache"] = [torch.empty(pool, KV_HEADS, DIM, device=device, dtype=fp8) for _ in range(2)]
    for tensor in out.values():
        tensor.view(torch.uint8).fill_(SENTINEL)
    return out


def qknorm_call(kernels, variant, inputs, out):
    qkv = inputs["qkv"]
    args = (qkv[:, :Q_HEADS], qkv[:, Q_HEADS:Q_HEADS + KV_HEADS], *inputs["weights"], inputs["trig"], inputs["positions"], EPS, out["q"], qkv[:, Q_HEADS + KV_HEADS:], out["k_cache"], out["v_cache"], inputs["slots"], out["k"], out["v"])
    if variant == "fp32":
        kernels["fp32"](*args)
    else:
        kernels["heads"](*args, int(variant.removeprefix("fp32-heads")))


def qknorm_bitwise(torch, a, b):
    return all(torch.equal(a[name].view(torch.uint8), b[name].view(torch.uint8)) for name in ("q", "k", "v", "k_cache", "v_cache"))


def check_qknorm(torch, kernels, inputs, outs):
    for variant, out in outs.items():
        qknorm_call(kernels, variant, inputs, out)
    torch.cuda.synchronize()
    result = {variant: {"bitwise": qknorm_bitwise(torch, outs[variant], outs["fp32"])} for variant in SELECTIONS["qknorm"][1]}
    for entry in result.values():
        entry["passed"] = entry["bitwise"]
    return result


def time_graphs(torch, steps):
    """steps: variant -> callable running one layer. Returns microseconds per layer."""
    graphs = {}
    stream = torch.cuda.Stream()
    for variant, step in steps.items():
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(LAYERS):
                step()
        graphs[variant] = graph
        graph.replay()
    torch.cuda.synchronize()
    names = list(steps)
    us = {variant: [] for variant in names}
    for round_index in range(ROUNDS):
        for offset in range(len(names)):
            variant = names[(round_index + offset) % len(names)]
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[variant].replay()
            end.record()
            end.synchronize()
            us[variant].append(start.elapsed_time(end) * 1000.0 / LAYERS)
    return us


def run_selection(output):
    begin = time.monotonic()
    import torch
    from sglang.kernels.ops.attention.rope import (
        apply_qknorm_rope_fp8_prefill_fp32,
        apply_qknorm_rope_fp8_prefill_fp32_heads,
    )
    from sglang.kernels.ops.moe.prefill_moe_input import (
        onepass_moe_input,
        onepass_split_moe_input,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    device = torch.device("cuda", rank)
    generator = torch.Generator(device=device)
    generator.manual_seed(1031 + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    local = {"rank": rank, "device": torch.cuda.get_device_name(rank), "uuid": str(torch.cuda.get_device_properties(rank).uuid), "cases": {"moe_input": {}, "qknorm": {}}, "us": {}, "endurance": {}}

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device=device, generator=generator) * scale).to(torch.bfloat16)

    moe_kernels = {"onepass": onepass_moe_input, "onepass-split": onepass_split_moe_input}
    moe_weights = {"norm": (torch.randn(HIDDEN, device=device, generator=generator) * 0.1 + 1.0).to(torch.bfloat16), "gate": randn(EXPERTS, HIDDEN, scale=0.02)}
    bufs = {variant: moe_buffer(torch, device) for variant in moe_kernels}
    for rows in ROWS:
        x0, r0 = randn(rows, HIDDEN, scale=0.6), randn(rows, HIDDEN, scale=3.0)
        x0[0].zero_()
        r0[0].zero_()  # all-zero row: the 1e-10 absmax floor
        r0[1].mul_(64)  # large residual row: scale range
        local["cases"]["moe_input"][str(rows)] = {"onepass-split": check_moe_input(torch, moe_kernels, moe_weights, bufs, x0, r0)}
        print(json.dumps({"rank": rank, "moe_input": rows, **local["cases"]["moe_input"][str(rows)]["onepass-split"]}), flush=True)
    qk_kernels = {"fp32": apply_qknorm_rope_fp8_prefill_fp32, "heads": apply_qknorm_rope_fp8_prefill_fp32_heads}
    qk_variants = ("fp32",) + SELECTIONS["qknorm"][1]
    for rows in ROWS:
        for id_dtype, name in ((torch.int64, "int64"), (torch.int32, "int32")):
            inputs = make_qknorm_inputs(torch, device, generator, rows, id_dtype)
            outs = {variant: make_qknorm_outputs(torch, device, rows, inputs["pool"]) for variant in qk_variants}
            local["cases"]["qknorm"][f"{rows}-{name}"] = check_qknorm(torch, qk_kernels, inputs, outs)
            print(json.dumps({"rank": rank, "qknorm": f"{rows}-{name}", **local["cases"]["qknorm"][f"{rows}-{name}"]}), flush=True)
    # Timing: in place on the activations, as select-firmus-prefill-moe-input.py timed the paths.
    for rows in TIMED_ROWS:
        moe = {variant: (randn(rows, HIDDEN, scale=0.6), randn(rows, HIDDEN, scale=3.0)) for variant in moe_kernels}
        local["us"][f"moe_input-{rows}"] = time_graphs(torch, {variant: (lambda v=variant, m=moe: moe_call(moe_kernels, v, moe_weights, *m[v], bufs[v])) for variant in moe_kernels})
        inputs = make_qknorm_inputs(torch, device, generator, rows, torch.int64)
        outs = {variant: make_qknorm_outputs(torch, device, rows, inputs["pool"]) for variant in qk_variants}
        local["us"][f"qknorm-{rows}"] = time_graphs(torch, {variant: (lambda v=variant, i=inputs, o=outs: qknorm_call(qk_kernels, v, i, o[v])) for variant in qk_variants})
    # Endurance: ENDURANCE_CALLS eager calls per candidate, alternating the timed shapes; the last
    # call of each shape must reproduce the first (MoE input: from the same residual each call).
    moe_inputs = {rows: (randn(rows, HIDDEN, scale=0.6), randn(rows, HIDDEN, scale=3.0), torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)) for rows in TIMED_ROWS}

    def moe_snapshot(rows):
        """One call from the shape's initial residual; what it wrote (x and scales of its rows, all top-k slots)."""
        x, r0, residual = moe_inputs[rows]
        residual.copy_(r0)
        buf = bufs["onepass-split"]
        moe_call(moe_kernels, "onepass-split", moe_weights, x, residual, buf)
        return (residual, buf["x"][:rows].view(torch.uint8), buf["sf"][:rows].view(torch.int32), buf["ids"], buf["w"].view(torch.int32))

    start = time.monotonic()
    first = {rows: [t.clone() for t in moe_snapshot(rows)] for rows in TIMED_ROWS}
    for call in range(ENDURANCE_CALLS):
        moe_snapshot(TIMED_ROWS[call % len(TIMED_ROWS)])
    stable = True
    for rows in TIMED_ROWS:
        stable = stable and all(torch.equal(a, b) for a, b in zip(moe_snapshot(rows), first[rows]))
    torch.cuda.synchronize()
    local["endurance"]["onepass-split"] = {"calls": ENDURANCE_CALLS + 2 * len(TIMED_ROWS), "stable": stable, "seconds": time.monotonic() - start}
    qk_inputs = {rows: make_qknorm_inputs(torch, device, generator, rows, torch.int64) for rows in TIMED_ROWS}
    for variant in SELECTIONS["qknorm"][1]:
        start = time.monotonic()
        reference = {rows: make_qknorm_outputs(torch, device, rows, qk_inputs[rows]["pool"]) for rows in TIMED_ROWS}
        outs = {rows: make_qknorm_outputs(torch, device, rows, qk_inputs[rows]["pool"]) for rows in TIMED_ROWS}
        for rows in TIMED_ROWS:
            qknorm_call(qk_kernels, "fp32", qk_inputs[rows], reference[rows])
        for call in range(ENDURANCE_CALLS):
            rows = TIMED_ROWS[call % len(TIMED_ROWS)]
            qknorm_call(qk_kernels, variant, qk_inputs[rows], outs[rows])
        torch.cuda.synchronize()
        local["endurance"][variant] = {"calls": ENDURANCE_CALLS, "stable": all(qknorm_bitwise(torch, outs[rows], reference[rows]) for rows in TIMED_ROWS), "seconds": time.monotonic() - start}
    local["median_us"] = {key: {variant: statistics.median(values) for variant, values in timed.items()} for key, timed in local["us"].items()}
    local["elapsed_seconds"] = time.monotonic() - begin
    print(json.dumps({"rank": rank, "median_us_per_layer": local["median_us"], "endurance": local["endurance"], "elapsed_seconds": local["elapsed_seconds"]}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(local, indent=2, allow_nan=False) + "\n")


def judge(entries, expected, selection, candidate):
    """(eligible, reason, savings at the full microbatch) of one candidate over all GPUs."""
    control_name = SELECTIONS[selection][0]
    if len(entries) != expected or any(e is None for e in entries):
        return False, "missing-results", []
    savings = []
    for global_rank, entry in enumerate(entries):
        checked = entry["cases"][selection]
        if set(checked) != set(cases()[selection]) or not all(c[candidate]["passed"] for c in checked.values()):
            return False, f"failed-gate gpu {global_rank}", savings
        endurance = entry["endurance"].get(candidate, {})
        if endurance.get("calls", 0) < ENDURANCE_CALLS or not endurance.get("stable"):
            return False, f"failed-endurance gpu {global_rank}", savings
        for rows in TIMED_ROWS:
            timed = entry["us"][f"{selection}-{rows}"]
            if not all(len(timed.get(v, [])) == ROUNDS for v in (control_name, candidate)):
                return False, f"untimed gpu {global_rank}", savings
            control, value = statistics.median(timed[control_name]), statistics.median(timed[candidate])
            required = max(MARGIN_US, MARGIN_FRACTION * control) if rows == TIMED_ROWS[0] else 0.0
            if control - value < required:
                return False, f"below-margin gpu {global_rank} at {rows} rows: {control:.1f} -> {value:.1f} us", savings
            if rows == TIMED_ROWS[0]:
                savings.append({"gpu": global_rank, "control_us": control, "candidate_us": value, "saving_us": control - value})
    return True, "all-gpus-faster", savings


def choose(entries, expected):
    result = {"env": {}}
    for selection, (control_name, candidates, variable) in SELECTIONS.items():
        judged = {candidate: judge(entries, expected, selection, candidate) for candidate in candidates}
        eligible = [c for c, (ok, _, _) in judged.items() if ok]
        winner = max(eligible, key=lambda c: sum(s["saving_us"] for s in judged[c][2])) if eligible else control_name
        result[selection] = {"winner": winner, "candidates": {c: {"eligible": ok, "reason": reason, "savings": savings} for c, (ok, reason, savings) in judged.items()}}
        if winner != control_name:
            result["env"][variable] = winner
    return result


def aggregate(directory, output, env_output, workers):
    entries = []
    for worker in range(workers):
        for rank in range(4):
            path = directory / f"prefill-{worker}" / f"rank-{rank}.json"
            entries.append(json.loads(path.read_text()) if path.is_file() else None)
    result = {"schema": "firmus-prefill-elementwise-selection/v1", "rule": {"rows": ROWS, "timed_rows": TIMED_ROWS, "layers": LAYERS, "rounds": ROUNDS, "margin_us": MARGIN_US, "margin_fraction": MARGIN_FRACTION, "endurance_calls": ENDURANCE_CALLS, "max_scale_relative": MAX_SCALE_RELATIVE, "min_same_ids": MIN_SAME_IDS}, **choose(entries, workers * 4)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with env_output.open("a") as env:
        env.write("".join(f"{k}={v}\n" for k, v in result["env"].items()))
    print(json.dumps({"env": result["env"], **{s: {"winner": result[s]["winner"], **{c: v["reason"] for c, v in result[s]["candidates"].items()}} for s in SELECTIONS}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="worker result directory (selection) or selection JSON (aggregate)")
    parser.add_argument("--aggregate", type=Path, help="directory of the prefill-<worker> result directories")
    parser.add_argument("--env-output", type=Path)
    parser.add_argument("--workers", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    if args.aggregate:
        assert args.env_output is not None
        aggregate(args.aggregate, args.output, args.env_output, args.workers)
    else:
        run_selection(args.output)
