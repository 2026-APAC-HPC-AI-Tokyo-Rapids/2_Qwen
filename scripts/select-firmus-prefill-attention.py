"""Gate and select the Prefill MoE input path before model load.

Run under torchrun on four GPUs of the serving image with the SGLang overlays
(no process group: every GPU checks and times alone). The Q/K norm and dense
GEMM selections that ran here before are settled: the FP32 Q/K norm writer won in
runs 40911121619 and 40925558339 and is pinned by prepare-firmus-pp4-settings.py
(SGLANG_PREFILL_QKNORM=fp32); the staggered QKV/O DeepGEMM schedule lost in both
(about +12%) and was removed.

moe_input: fused (the pinned post-attention path: post_norm_moe_input writes the
residual, BF16 y in place, the MegaMoE FP8 input and scales; the gate GEMM reads y;
the router writes the buffer's top-k) or onepass (onepass_moe_input: one kernel that
never writes y; the FP8 codes come from the unrounded FP32 y, the gate operand is
BF16(h * w) and rstd scales the FP32 logits). Gate, on the PP4 microbatch row counts
and the 2048-row floor (ROWS) with an all-zero row and a large row: the residual
equals fused bit for bit; each FP8 code is within one E4M3 step of the FP64 RMSNorm
output rounded once under the candidate's own scale (exact on at least MIN_EXACT),
the dequantized input's relative L2 error to that reference is not above fused's and
the scales are amax(|y|) / 448 within MAX_SCALE_ERROR; the top-k ids are distinct, a
valid top-8 of the FP64 logits within two BF16 steps, match the FP64 top-8 set about
as often as fused does (of the rows where exactly one path matches it, onepass loses
no more than SET_SIGMAS standard deviations of a fair sign test), and the weights
sum to one and agree with fused within the rounding of BF16 logits where the sets
agree; the pad slots equal fused's.

Timing: one CUDA graph of 24 layers per path at 7988 rows (the post-attention
input of one layer), rotated for 12 rounds. --aggregate adopts onepass when it
passed everywhere and saved max(3 us, 2%) per layer on every GPU; anything
missing keeps fused.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

EPS, HIDDEN = 1e-6, 4096
# MegaMoE input buffer: SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK rows, 128 experts, top-8.
CAP, EXPERTS, TOPK = 8192, 128, 8
# PP4 microbatches are 7988 rows (2 x 3994; 7989 and 3994 also occur, prefill-0 of run 40825417205).
ROWS = (7988, 7989, 3994, 2049)
TIMED_ROWS, LAYERS, ROUNDS = 7988, 24, 12
MARGIN_US, MARGIN_FRACTION = 3.0, 0.02
# Rounding the FP64 value once can differ from FP32 arithmetic only at E4M3 midpoints.
MIN_EXACT = 0.9999
# onepass scales: FP32 products and the approximate rsqrt (both paths use it) against FP64.
MAX_SCALE_ERROR = 1e-5
# Rows where exactly one path selects the FP64 top-8 set (near-ties at the BF16 logit
# rounding): onepass may lose at most this many standard deviations of a fair sign test.
SET_SIGMAS = 3.0
# selection -> (control, candidate, environment switch)
SELECTIONS = {"moe_input": ("fused", "onepass", "SGLANG_PREFILL_MOE_INPUT")}


def cases():
    return {"moe_input": [str(rows) for rows in ROWS]}


def e4m3_steps(torch, a, b):
    """Distance in E4M3 codes between finite values (sign-magnitude to a monotone integer line)."""
    def line(x):
        bits = x.view(torch.uint8).int()
        return torch.where(bits >= 128, -(bits - 128), bits)
    return (line(a) - line(b)).abs()


def make_moe_inputs(torch, device, generator):
    """Attention output, residual, norm and gate weights, as select-firmus-prefill-moe-input.py
    draws them; and one MegaMoE input buffer per path."""
    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device=device, generator=generator) * scale).to(torch.bfloat16)

    weights = {"norm": (torch.randn(HIDDEN, device=device, generator=generator) * 0.1 + 1.0).to(torch.bfloat16), "gate": randn(EXPERTS, HIDDEN, scale=0.02), "zero_bias": torch.zeros(EXPERTS, dtype=torch.float32, device=device)}
    bufs = {variant: {"x": torch.empty(CAP, HIDDEN, dtype=torch.float8_e4m3fn, device=device), "sf": torch.empty(CAP, HIDDEN // 128, device=device), "ids": torch.empty(CAP, TOPK, dtype=torch.int64, device=device), "w": torch.empty(CAP, TOPK, device=device)} for variant in SELECTIONS["moe_input"][:2]}
    return randn, weights, bufs


def moe_input_call(kernels, F, variant, weights, x, residual, buf):
    if variant == "fused":
        kernels["post_norm"](x, residual, weights["norm"], EPS, buf["x"], buf["sf"], buf["ids"], buf["w"])
        kernels["route"](F.linear(x, weights["gate"]), weights["zero_bias"], buf["ids"], buf["w"])
    else:
        kernels["onepass"](x, residual, weights["norm"], EPS, weights["gate"], buf["x"], buf["sf"], buf["ids"], buf["w"])


def check_moe_input(torch, F, kernels, weights, bufs, x0, r0):
    """Both paths on copies of (x0, r0); onepass judged against fused and the FP64 reference."""
    rows = x0.shape[0]
    residuals = {}
    for variant, buf in bufs.items():
        buf["x"].view(torch.uint8).fill_(0x55)
        buf["sf"].fill_(7.0)
        buf["ids"].fill_(77)
        buf["w"].fill_(5.0)
        residuals[variant] = r0.clone()
        moe_input_call(kernels, F, variant, weights, x0.clone(), residuals[variant], buf)
    torch.cuda.synchronize()
    ctrl, cand = bufs["fused"], bufs["onepass"]
    # FP64 RMSNorm of the FP32 residual sum, and its gate logits.
    h = (x0.float() + r0.float()).double()
    y = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + EPS) * weights["norm"].double()
    groups = y.view(rows, HIDDEN // 128, 128)
    amax = groups.abs().amax(-1)
    exact = y @ weights["gate"].double().T

    def dequantized(buf):
        return (buf["x"][:rows].double().view(rows, -1, 128) * buf["sf"][:rows, :, None].double()).view(rows, HIDDEN)

    reference_codes = (groups / cand["sf"][:rows, :, None].double()).float().to(torch.float8_e4m3fn).view(rows, HIDDEN)
    steps = e4m3_steps(torch, cand["x"][:rows], reference_codes)
    errors = [((dequantized(buf) - y).norm() / y.norm()).item() for buf in (cand, ctrl)]
    live = amax > 0
    step = bf16_logit_step(torch, exact)
    ids, ctrl_ids = cand["ids"][:rows], ctrl["ids"][:rows]
    safe = ids.clamp(0, EXPERTS - 1)
    exact_set = exact.topk(TOPK, dim=-1).indices.sort(-1).values
    hits = {name: (b.sort(-1).values == exact_set).all(-1) for name, b in (("onepass", ids), ("fused", ctrl_ids))}
    wins, losses = int((hits["onepass"] & ~hits["fused"]).sum().item()), int((hits["fused"] & ~hits["onepass"]).sum().item())
    same = (ids.sort(-1).values == ctrl_ids.sort(-1).values).all(-1)
    weight_error = ((cand["w"][:rows].gather(1, ids.argsort(-1)) - ctrl["w"][:rows].gather(1, ctrl_ids.argsort(-1))).abs() - 2 * step - 1e-6)[same]
    result = {
        "residual_bitwise": torch.equal(residuals["onepass"].view(torch.int16), residuals["fused"].view(torch.int16)),
        "x_exact_fraction": (steps == 0).float().mean().item(),
        "x_max_steps": int(steps.max().item()),
        "x_relative_l2": errors[0],
        "fused_x_relative_l2": errors[1],
        "scale_max_relative": ((cand["sf"][:rows].double() - amax / 448).abs() / (amax / 448))[live].max().item(),
        "ids_in_range_distinct": bool(((ids >= 0) & (ids < EXPERTS)).all().item()) and bool((ids.sort(-1).values.diff(dim=-1) > 0).all().item()),
        "valid_topk": bool((exact.gather(1, safe).amin(-1, keepdim=True) >= exact.scatter(1, safe, float("-inf")).amax(-1, keepdim=True) - 2 * step).all().item()),
        "exact_set_fraction": hits["onepass"].float().mean().item(),
        "fused_exact_set_fraction": hits["fused"].float().mean().item(),
        "exact_set_wins": wins,
        "exact_set_losses": losses,
        "same_set_as_fused": same.float().mean().item(),
        "weights_sum_error": (cand["w"][:rows].sum(-1) - 1).abs().max().item(),
        "weights_within_bf16_logit_rounding": bool((weight_error <= 0).all().item()),
        "pad_slots": torch.equal(cand["ids"][rows:], ctrl["ids"][rows:]) and torch.equal(cand["w"][rows:].view(torch.int32), ctrl["w"][rows:].view(torch.int32)) and bool((cand["ids"][rows:] == -1).all().item()),
    }
    quantized = result["residual_bitwise"] and result["x_exact_fraction"] >= MIN_EXACT and result["x_max_steps"] <= 1 and result["x_relative_l2"] <= result["fused_x_relative_l2"] and result["scale_max_relative"] <= MAX_SCALE_ERROR
    routed = result["ids_in_range_distinct"] and result["valid_topk"] and losses - wins <= SET_SIGMAS * (wins + losses) ** 0.5 + 2 and result["weights_sum_error"] <= 1e-5 and result["weights_within_bf16_logit_rounding"]
    result["passed"] = quantized and routed and result["pad_slots"]
    return result


def bf16_logit_step(torch, exact):
    """Spacing of BF16 numbers at each row's largest |logit| (normal range)."""
    return torch.exp2(torch.floor(torch.log2(exact.abs().amax(-1, keepdim=True).clamp_min(2.0**-126))) - 7)


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
    import torch.nn.functional as F
    from sglang.kernels.ops.moe.prefill_moe_input import (
        onepass_moe_input,
        post_norm_moe_input,
        route_into_buffer,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    device = torch.device("cuda", rank)
    generator = torch.Generator(device=device)
    generator.manual_seed(1030 + rank)
    kernels = {"post_norm": post_norm_moe_input, "route": route_into_buffer, "onepass": onepass_moe_input}
    torch.backends.cuda.matmul.allow_tf32 = False
    local = {"rank": rank, "device": torch.cuda.get_device_name(rank), "uuid": str(torch.cuda.get_device_properties(rank).uuid), "cases": {selection: {} for selection in SELECTIONS}, "us": {}}
    randn, moe_weights, moe_bufs = make_moe_inputs(torch, device, generator)
    for rows in ROWS:
        x0, r0 = randn(rows, HIDDEN, scale=0.6), randn(rows, HIDDEN, scale=3.0)
        x0[0].zero_()
        r0[0].zero_()  # all-zero row: the 1e-10 absmax floor
        r0[1].mul_(64)  # large residual row: scale range
        local["cases"]["moe_input"][str(rows)] = check_moe_input(torch, F, kernels, moe_weights, moe_bufs, x0, r0)
        print(json.dumps({"rank": rank, "moe_input": rows, **local["cases"]["moe_input"][str(rows)]}), flush=True)
    # In place on 7988-row activations, as select-firmus-prefill-moe-input.py timed the paths.
    moe = {variant: (randn(TIMED_ROWS, HIDDEN, scale=0.6), randn(TIMED_ROWS, HIDDEN, scale=3.0)) for variant in moe_bufs}
    local["us"]["moe_input"] = time_graphs(torch, {variant: (lambda v=variant: moe_input_call(kernels, F, v, moe_weights, *moe[v], moe_bufs[v])) for variant in moe_bufs})
    local["median_us"] = {selection: {variant: statistics.median(values) for variant, values in timed.items()} for selection, timed in local["us"].items()}
    local["elapsed_seconds"] = time.monotonic() - begin
    print(json.dumps({"rank": rank, "median_us_per_layer": local["median_us"], "elapsed_seconds": local["elapsed_seconds"]}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(local, indent=2, allow_nan=False) + "\n")


def choose(entries, expected, selection):
    """The candidate passed every case and saved the margin on every GPU."""
    control_name, candidate_name, _ = SELECTIONS[selection]
    if len(entries) != expected or any(e is None for e in entries):
        return {"winner": control_name, "reason": "missing-results"}
    reason, savings = None, []
    for entry in entries:
        timed = entry.get("us", {}).get(selection, {})
        checked = entry.get("cases", {}).get(selection, {})
        if not all(len(timed.get(v, [])) == ROUNDS for v in (control_name, candidate_name)) or set(checked) != set(cases()[selection]) or not all(c["passed"] for c in checked.values()):
            reason = f"failed-or-untimed gpu {entry['rank']}"
            continue
        control, candidate = statistics.median(timed[control_name]), statistics.median(timed[candidate_name])
        savings.append({"gpu": entry["rank"], "control_us": control, "candidate_us": candidate, "saving_us": control - candidate, "required_us": max(MARGIN_US, MARGIN_FRACTION * control)})
    if reason is None and not all(s["saving_us"] >= s["required_us"] for s in savings):
        reason = "below-margin"
    return {"winner": candidate_name if reason is None else control_name, "reason": reason or "all-gpus-faster", "savings": savings}


def aggregate(directory, output, env_output, gpus=4):
    entries = []
    for rank in range(gpus):
        path = directory / f"rank-{rank}.json"
        entries.append(json.loads(path.read_text()) if path.is_file() else None)
    result = {"schema": "firmus-prefill-attention-selection/v1", "rule": {"rows": ROWS, "timed_rows": TIMED_ROWS, "layers": LAYERS, "rounds": ROUNDS, "margin_us": MARGIN_US, "margin_fraction": MARGIN_FRACTION, "min_exact": MIN_EXACT}, "env": {}}
    for selection, (control_name, _, variable) in SELECTIONS.items():
        result[selection] = choose(entries, gpus, selection)
        if result[selection]["winner"] != control_name:
            result["env"][variable] = result[selection]["winner"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with env_output.open("a") as env:
        env.write("".join(f"{k}={v}\n" for k, v in result["env"].items()))
    print(json.dumps({"env": result["env"], **{s: {k: result[s][k] for k in ("winner", "reason")} for s in SELECTIONS}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="GPU result directory (selection) or selection JSON (aggregate)")
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--env-output", type=Path)
    args = parser.parse_args()
    if args.aggregate:
        assert args.env_output is not None
        aggregate(args.aggregate, args.output, args.env_output)
    else:
        run_selection(args.output)
