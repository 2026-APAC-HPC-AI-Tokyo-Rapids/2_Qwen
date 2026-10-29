"""Gate and select the Prefill MegaMoE input path before model load.

Run under torchrun on four GPUs of the serving image with the SGLang overlay
(no process group: every GPU checks and times alone). Paths per layer:
control (flashinfer fused_add_rmsnorm, gate GEMM, zero bias, Triton router,
mega_moe_pre_dispatch_sm90), fused (post_norm_moe_input writes the residual,
BF16 y, FP8 x, scales and pad slots; the router writes the buffer's top-k) and
fused-gate (fused, with the gate GEMM and router as one kernel).

Numeric gate, on the PP4 microbatch row counts and the 2048-row floor:
fused keeps the residual bit for bit, keeps y within one BF16 rounding of
flashinfer, and its FP8 x, scales and top-k buffer (pad slots included) must
equal what pre-dispatch and the router write from the same y; fused-gate must
write the same FP8 input, select a valid top-8 of the FP32 logits within two
BF16 steps, keep the router's ids on at least 99% of rows and its weights
within the rounding of BF16 logits. Timing: one CUDA graph of 24 layers per
path, rotated for 12 rounds. --aggregate adopts fused when it passed and saved
max(3 us, 2%) per layer on every GPU, fused-gate when it also saved 5 us per
layer over fused (part of its gain is launch latency, which returns less at the
Prefill power cap); anything missing keeps control.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

HIDDEN, EXPERTS, TOPK, CAP, EPS = 4096, 128, 8, 8192, 1e-6
# PP4 microbatches are 7988 rows (2 x 3994; 7989 and 3994 also occur, prefill-0 of run 40825417205).
ROWS = (7988, 7989, 3994, 2049)
TIMED_ROWS, LAYERS, ROUNDS = 7988, 24, 12
VARIANTS = ("control", "fused", "fused-gate")
MARGIN_US, MARGIN_FRACTION, GATE_MARGIN_US = 3.0, 0.02, 5.0
MIN_SAME_IDS = 0.99


def bf16_step(torch, value):
    """Spacing of BF16 numbers at |value| (normal range)."""
    return torch.exp2(torch.floor(torch.log2(value.abs().clamp_min(2.0**-126))) - 7)


def make_paths(torch, F, kernels, weight, gate, zero_bias):
    """The three per-layer paths on (x, residual, buffer); kernels holds the callables."""
    def control(x, residual, buf):
        kernels["norm"](x, residual, weight, EPS)
        w, ids = kernels["router"](F.linear(x, gate), torch.zeros(EXPERTS, dtype=torch.float32, device=x.device), TOPK, scoring_func="softmax", renormalize=True)
        kernels["pre_dispatch"](x, ids, w, buf["x"], buf["sf"], buf["ids"], buf["w"], num_tokens=x.shape[0], group_size=128, routed_scaling_factor=1.0)

    def fused(x, residual, buf):
        kernels["post_norm"](x, residual, weight, EPS, buf["x"], buf["sf"], buf["ids"], buf["w"])
        kernels["route"](F.linear(x, gate), zero_bias, buf["ids"], buf["w"])

    def fused_gate(x, residual, buf):
        kernels["post_norm"](x, residual, weight, EPS, buf["x"], buf["sf"], buf["ids"], buf["w"])
        kernels["gate_route"](x, gate, buf["ids"], buf["w"])

    return {"control": control, "fused": fused, "fused-gate": fused_gate}


def make_buffer(torch, device):
    return {"x": torch.empty(CAP, HIDDEN, dtype=torch.float8_e4m3fn, device=device), "sf": torch.empty(CAP, HIDDEN // 128, device=device), "ids": torch.empty(CAP, TOPK, dtype=torch.int64, device=device), "w": torch.empty(CAP, TOPK, device=device)}


def scramble(torch, buf):
    buf["x"].view(torch.uint8).fill_(0x55)
    buf["sf"].fill_(7.0)
    buf["ids"].fill_(77)
    buf["w"].fill_(5.0)


def check_rows(torch, F, kernels, paths, x0, r0, gate, zero_bias, bufs, reference):
    """Run every path on copies of (x0, r0) and judge fused and fused-gate."""
    rows = x0.shape[0]
    state = {}
    for variant in VARIANTS:
        x, residual = x0.clone(), r0.clone()
        scramble(torch, bufs[variant])
        paths[variant](x, residual, bufs[variant])
        state[variant] = (x, residual)
    (xc, rc), (xa, ra), (xb, rb) = state["control"], state["fused"], state["fused-gate"]
    ca, cb, gb = bufs["control"], bufs["fused"], bufs["fused-gate"]
    # fused: pre-dispatch and the router applied to the candidate's own y.
    w_ref, ids_ref = kernels["router"](F.linear(xa, gate), zero_bias, TOPK, scoring_func="softmax", renormalize=True)
    scramble(torch, reference)
    kernels["pre_dispatch"](xa, ids_ref, w_ref, reference["x"], reference["sf"], reference["ids"], reference["w"], num_tokens=rows, group_size=128, routed_scaling_factor=1.0)
    y_err = ((xa.float() - xc.float()).abs() / xc.float().abs().clamp_min(1e-30))[xc != 0]
    fused = {
        "residual_bitwise": torch.equal(ra.view(torch.int16), rc.view(torch.int16)),
        "y_bitwise_fraction": (xa.view(torch.int16) == xc.view(torch.int16)).float().mean().item(),
        "y_max_relative": y_err.max().item() if y_err.numel() else 0.0,
        "y_zero_pattern": torch.equal(xa == 0, xc == 0),
        "x_fp8_bitwise": torch.equal(cb["x"][:rows].view(torch.uint8), reference["x"][:rows].view(torch.uint8)),
        "scales_bitwise": torch.equal(cb["sf"][:rows].view(torch.int32), reference["sf"][:rows].view(torch.int32)),
        "topk_buffer_bitwise": torch.equal(cb["ids"], reference["ids"]) and torch.equal(cb["w"].view(torch.int32), reference["w"].view(torch.int32)),
        "ids_same_as_control": (cb["ids"][:rows] == ca["ids"][:rows]).all(-1).float().mean().item(),
    }
    # One BF16 rounding apart: flashinfer and the Triton norm may differ in the FP32 sum order.
    fused["passed"] = fused["residual_bitwise"] and fused["y_max_relative"] <= 2.0**-7 and fused["y_zero_pattern"] and fused["x_fp8_bitwise"] and fused["scales_bitwise"] and fused["topk_buffer_bitwise"]
    # fused-gate: same norm kernel; its routing is judged against the FP32 logits.
    exact = xa.float() @ gate.float().T
    step = bf16_step(torch, exact.abs().amax(-1, keepdim=True))
    ids = gb["ids"][:rows]
    safe = ids.clamp(0, EXPERTS - 1)
    same = (ids == ids_ref.long()).all(-1)
    weight_error = ((gb["w"][:rows] - w_ref).abs() - 2 * step - 1e-6)[same]
    fused_gate = {
        "fp8_input_bitwise": torch.equal(gb["x"][:rows].view(torch.uint8), cb["x"][:rows].view(torch.uint8)) and torch.equal(gb["sf"][:rows].view(torch.int32), cb["sf"][:rows].view(torch.int32)) and torch.equal(xb.view(torch.int16), xa.view(torch.int16)) and torch.equal(rb.view(torch.int16), ra.view(torch.int16)),
        "pad_slots": torch.equal(gb["ids"][rows:], reference["ids"][rows:]) and torch.equal(gb["w"][rows:].view(torch.int32), reference["w"][rows:].view(torch.int32)),
        "ids_in_range_distinct": bool(((ids >= 0) & (ids < EXPERTS)).all().item()) and bool((ids.sort(-1).values.diff(dim=-1) > 0).all().item()),
        "valid_topk": bool((exact.gather(1, safe).amin(-1, keepdim=True) >= exact.scatter(1, safe, float("-inf")).amax(-1, keepdim=True) - 2 * step).all().item()),
        "ids_same_as_router": same.float().mean().item(),
        "weights_sum_error": (gb["w"][:rows].sum(-1) - 1).abs().max().item(),
        "weights_within_bf16_logit_rounding": bool((weight_error <= 0).all().item()),
    }
    fused_gate["passed"] = fused["passed"] and fused_gate["fp8_input_bitwise"] and fused_gate["pad_slots"] and fused_gate["ids_in_range_distinct"] and fused_gate["valid_topk"] and fused_gate["ids_same_as_router"] >= MIN_SAME_IDS and fused_gate["weights_sum_error"] <= 1e-5 and fused_gate["weights_within_bf16_logit_rounding"]
    return {"fused": fused, "fused-gate": fused_gate}


def run_selection(output):
    begin = time.monotonic()
    import deep_gemm
    import torch
    import torch.nn.functional as F
    from sgl_kernel import fused_add_rmsnorm
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
    from sglang.kernels.ops.moe.prefill_moe_input import (
        gate_route_into_buffer,
        post_norm_moe_input,
        route_into_buffer,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", rank)
    generator = torch.Generator(device=device)
    generator.manual_seed(1029 + rank)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device=device, generator=generator) * scale).to(torch.bfloat16)

    weight = (torch.randn(HIDDEN, device=device, generator=generator) * 0.1 + 1.0).to(torch.bfloat16)
    gate = randn(EXPERTS, HIDDEN, scale=0.02)
    zero_bias = torch.zeros(EXPERTS, dtype=torch.float32, device=device)
    kernels = {"norm": fused_add_rmsnorm, "router": moe_fused_gate, "pre_dispatch": deep_gemm.mega_moe_pre_dispatch_sm90, "post_norm": post_norm_moe_input, "route": route_into_buffer, "gate_route": gate_route_into_buffer}
    paths = make_paths(torch, F, kernels, weight, gate, zero_bias)
    local = {"rank": rank, "device": torch.cuda.get_device_name(rank), "uuid": str(torch.cuda.get_device_properties(rank).uuid), "rows": {}, "us": {}}
    bufs = {variant: make_buffer(torch, device) for variant in VARIANTS}
    reference = make_buffer(torch, device)
    for rows in ROWS:
        x0, r0 = randn(rows, HIDDEN, scale=0.6), randn(rows, HIDDEN, scale=3.0)
        x0[0].zero_()
        r0[0].zero_()  # all-zero row: the 1e-10 absmax floor
        r0[1].mul_(64)  # large residual row: scale range
        local["rows"][str(rows)] = check_rows(torch, F, kernels, paths, x0, r0, gate, zero_bias, bufs, reference)
        print(json.dumps({"rank": rank, "rows": rows, **local["rows"][str(rows)]}), flush=True)
    # Timing: 24 layers of one path per CUDA graph, in place on 7988-row activations.
    graphs = {}
    stream = torch.cuda.Stream()
    for variant in VARIANTS:
        x, residual = randn(TIMED_ROWS, HIDDEN, scale=0.6), randn(TIMED_ROWS, HIDDEN, scale=3.0)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            paths[variant](x, residual, bufs[variant])
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(LAYERS):
                paths[variant](x, residual, bufs[variant])
        graphs[variant] = (graph, x, residual)
        graph.replay()
    torch.cuda.synchronize()
    local["us"] = {variant: [] for variant in VARIANTS}
    for round_index in range(ROUNDS):
        for offset in range(len(VARIANTS)):
            variant = VARIANTS[(round_index + offset) % len(VARIANTS)]
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[variant][0].replay()
            end.record()
            end.synchronize()
            local["us"][variant].append(start.elapsed_time(end) * 1000.0 / LAYERS)
    local["median_us"] = {variant: statistics.median(values) for variant, values in local["us"].items()}
    local["elapsed_seconds"] = time.monotonic() - begin
    print(json.dumps({"rank": rank, "median_us_per_layer": local["median_us"], "elapsed_seconds": local["elapsed_seconds"]}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(local, indent=2, allow_nan=False) + "\n")


def choose(entries, expected):
    """fused: passed and saved the margin on every GPU; fused-gate: the same, and GATE_MARGIN_US faster than fused."""
    if len(entries) != expected or any(e is None for e in entries):
        return {"winner": "control", "reason": "missing-results"}
    choices = {}
    for name in VARIANTS[1:]:
        reason, savings = None, []
        for entry in entries:
            timed = all(len(entry["us"][v]) == ROUNDS for v in VARIANTS)
            if not timed or set(entry["rows"]) != {str(r) for r in ROWS} or not all(entry["rows"][str(r)][name]["passed"] for r in ROWS):
                reason = f"failed-or-untimed gpu {entry['rank']}"
                continue
            control, candidate = statistics.median(entry["us"]["control"]), statistics.median(entry["us"][name])
            required = max(MARGIN_US, MARGIN_FRACTION * control)
            if name == "fused-gate":
                required = max(required, control - statistics.median(entry["us"]["fused"]) + GATE_MARGIN_US)
            savings.append({"gpu": entry["rank"], "control_us": control, "candidate_us": candidate, "saving_us": control - candidate, "required_us": required})
        if reason is None and not all(s["saving_us"] >= s["required_us"] for s in savings):
            reason = "below-margin"
        choices[name] = {"eligible": reason is None, "reason": reason or "all-gpus-faster", "savings": savings}
    if choices["fused-gate"]["eligible"]:
        winner = "fused-gate"
    elif choices["fused"]["eligible"]:
        winner = "fused"
    else:
        winner = "control"
    return {"winner": winner, "reason": "eligible" if winner != "control" else "no-eligible-candidate", "choices": choices}


def aggregate(directory, output, env_output, gpus=4):
    entries = []
    for rank in range(gpus):
        path = directory / f"rank-{rank}.json"
        entries.append(json.loads(path.read_text()) if path.is_file() else None)
    selection = choose(entries, gpus)
    selection["env"] = {} if selection["winner"] == "control" else {"SGLANG_PREFILL_MOE_INPUT": selection["winner"]}
    result = {"schema": "firmus-prefill-moe-input-selection/v1", "rule": {"rows": ROWS, "timed_rows": TIMED_ROWS, "layers": LAYERS, "rounds": ROUNDS, "margin_us": MARGIN_US, "margin_fraction": MARGIN_FRACTION, "gate_margin_us": GATE_MARGIN_US, "min_same_ids": MIN_SAME_IDS}, **selection}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with env_output.open("a") as env:
        env.write("".join(f"{k}={v}\n" for k, v in selection["env"].items()))
    print(json.dumps({"winner": result["winner"], "reason": result["reason"], "env": result["env"], **{name: {k: v for k, v in choice.items() if k != "savings"} for name, choice in result.get("choices", {}).items()}}), flush=True)


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
