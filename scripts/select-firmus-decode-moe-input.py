"""Gate and select the Decode TARGET_VERIFY MegaMoE input path before model load.

Run under torchrun on the four Decode GPUs of the serving image with the SGLang
overlay (no process group: every GPU checks and times alone). Paths per layer:
control (flashinfer fused_add_rmsnorm, gate GEMM, the Triton router with the
graph-padding mask, mega_moe_pre_dispatch_sm90), as forward_megamoe runs it in
the verify graphs, and fused (post_norm_moe_input writes the residual, BF16 y,
FP8 x and scales of the live rows; the same router launch writes the buffer's
top-k; no pre-dispatch kernel).

Numeric gate, at 4 x (cap/4 - 4, cap/4 - 2, cap/4) verify rows with the last
request padded, at cap/4 without padding, and at one request: fused keeps the
residual bit for bit, keeps y within one BF16 rounding of flashinfer, and its
FP8 x, scales, top-k ids (padded rows -1) and weights must equal what the router
and pre-dispatch write from the same y; the buffer past the rows is untouched.
Timing: one CUDA graph of 94 layers per path and batch, two blocks of
alternating replays. --aggregate adopts fused when it passed everywhere and saved
max(1 us, 2%) per layer at every batch, in both blocks, on every GPU; anything
missing keeps control (pre-dispatch).
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

HIDDEN, EXPERTS, TOPK, CAP, EPS = 4096, 128, 8, 4096, 1e-6
TOKENS_PER_REQUEST, LAYERS, BLOCKS, PAIRS = 4, 94, 2, 10
VARIANTS = ("control", "fused")
MARGIN_US, MARGIN_FRACTION = 1.0, 0.02


def batches_from_config(path):
    config = json.loads(path.read_text())
    per_rank = config["decode_max_running_requests"] // config["decode_dp_size"]
    return (per_rank - 4, per_rank - 2, per_rank)


def check_cases(batches):
    """(rows, non-padded rows): the last request padded at every batch, then full and single-request graphs."""
    cases = [(TOKENS_PER_REQUEST * b, TOKENS_PER_REQUEST * (b - 1)) for b in batches]
    return cases + [(TOKENS_PER_REQUEST * batches[-1],) * 2, (TOKENS_PER_REQUEST,) * 2]


def make_buffer(torch, device):
    return {"x": torch.empty(CAP, HIDDEN, dtype=torch.float8_e4m3fn, device=device), "sf": torch.empty(CAP, HIDDEN // 128, device=device), "ids": torch.empty(CAP, TOPK, dtype=torch.int64, device=device), "w": torch.empty(CAP, TOPK, device=device)}


def scramble(torch, buf):
    buf["x"].view(torch.uint8).fill_(0x55)
    buf["sf"].fill_(7.0)
    buf["ids"].fill_(77)
    buf["w"].fill_(5.0)


def make_paths(torch, F, kernels, weight, gate):
    def control(x, residual, buf, count):
        kernels["norm"](x, residual, weight, EPS)
        w, ids = kernels["router"](F.linear(x, gate), None, TOPK, scoring_func="softmax", renormalize=True, num_token_non_padded=count)
        kernels["pre_dispatch"](x, ids.to(torch.int32), w.to(torch.float32), buf["x"], buf["sf"], buf["ids"], buf["w"], num_tokens=x.shape[0], group_size=128, routed_scaling_factor=1.0)

    def fused(x, residual, buf, count):
        kernels["post_norm"](x, residual, weight, EPS, buf["x"], buf["sf"], buf["ids"], buf["w"], pad_slots=False)
        kernels["route"](F.linear(x, gate), None, buf["ids"], buf["w"], count)

    return {"control": control, "fused": fused}


def check_rows(torch, F, kernels, paths, x0, r0, gate, count, bufs, reference):
    rows, live = x0.shape[0], int(count.item())
    state = {}
    for variant in VARIANTS:
        x, residual = x0.clone(), r0.clone()
        scramble(torch, bufs[variant])
        paths[variant](x, residual, bufs[variant], count)
        state[variant] = (x, residual)
    (xc, rc), (xa, ra) = state["control"], state["fused"]
    fb = bufs["fused"]
    # The router and pre-dispatch applied to the candidate's own y.
    w_ref, ids_ref = kernels["router"](F.linear(xa, gate), None, TOPK, scoring_func="softmax", renormalize=True, num_token_non_padded=count)
    scramble(torch, reference)
    kernels["pre_dispatch"](xa, ids_ref.to(torch.int32), w_ref.to(torch.float32), reference["x"], reference["sf"], reference["ids"], reference["w"], num_tokens=rows, group_size=128, routed_scaling_factor=1.0)
    y_err = ((xa.float() - xc.float()).abs() / xc.float().abs().clamp_min(1e-30))[xc != 0]
    result = {
        "residual_bitwise": torch.equal(ra.view(torch.int16), rc.view(torch.int16)),
        "y_bitwise_fraction": (xa.view(torch.int16) == xc.view(torch.int16)).float().mean().item(),
        "y_max_relative": y_err.max().item() if y_err.numel() else 0.0,
        "y_zero_pattern": torch.equal(xa == 0, xc == 0),
        "x_fp8_bitwise": torch.equal(fb["x"][:rows].view(torch.uint8), reference["x"][:rows].view(torch.uint8)),
        "scales_bitwise": torch.equal(fb["sf"][:rows].view(torch.int32), reference["sf"][:rows].view(torch.int32)),
        "topk_bitwise": torch.equal(fb["ids"][:rows], reference["ids"][:rows]) and torch.equal(fb["w"][:rows].view(torch.int32), reference["w"][:rows].view(torch.int32)),
        "padded_ids_masked": bool((fb["ids"][live:rows] == -1).all().item()),
        "rows_past_untouched": bool((fb["x"][rows:].view(torch.uint8) == 0x55).all().item()) and bool((fb["sf"][rows:] == 7.0).all().item()) and bool((fb["ids"][rows:] == 77).all().item()) and bool((fb["w"][rows:] == 5.0).all().item()),
        "ids_same_as_control": (fb["ids"][:live] == bufs["control"]["ids"][:live]).all(-1).float().mean().item(),
    }
    # One BF16 rounding apart: flashinfer and the Triton norm may differ in the FP32 sum order.
    result["passed"] = result["residual_bitwise"] and result["y_max_relative"] <= 2.0**-7 and result["y_zero_pattern"] and result["x_fp8_bitwise"] and result["scales_bitwise"] and result["topk_bitwise"] and result["padded_ids_masked"] and result["rows_past_untouched"]
    return result


def run_selection(output, config_path):
    begin = time.monotonic()
    import deep_gemm
    import torch
    import torch.nn.functional as F
    from sgl_kernel import fused_add_rmsnorm
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
    from sglang.kernels.ops.moe.prefill_moe_input import (
        post_norm_moe_input,
        route_into_buffer,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", rank)
    generator = torch.Generator(device=device)
    generator.manual_seed(1030 + rank)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device=device, generator=generator) * scale).to(torch.bfloat16)

    batches = batches_from_config(config_path)
    weight = (torch.randn(HIDDEN, device=device, generator=generator) * 0.1 + 1.0).to(torch.bfloat16)
    gate = randn(EXPERTS, HIDDEN, scale=0.02)
    kernels = {"norm": fused_add_rmsnorm, "router": moe_fused_gate, "pre_dispatch": deep_gemm.mega_moe_pre_dispatch_sm90, "post_norm": post_norm_moe_input, "route": route_into_buffer}
    paths = make_paths(torch, F, kernels, weight, gate)
    local = {"rank": rank, "device": torch.cuda.get_device_name(rank), "uuid": str(torch.cuda.get_device_properties(rank).uuid), "batches": list(batches), "rows": {}, "us": {}}
    bufs = {variant: make_buffer(torch, device) for variant in VARIANTS}
    reference = make_buffer(torch, device)
    for rows, live in check_cases(batches):
        x0, r0 = randn(rows, HIDDEN, scale=0.6), randn(rows, HIDDEN, scale=3.0)
        x0[0].zero_()
        r0[0].zero_()  # all-zero row: the 1e-10 absmax floor
        if rows > 1:
            r0[1].mul_(64)  # large residual row: scale range
        count = torch.tensor(live, dtype=torch.int32, device=device)
        key = f"{rows}/{live}"
        local["rows"][key] = check_rows(torch, F, kernels, paths, x0, r0, gate, count, bufs, reference)
        print(json.dumps({"rank": rank, "rows": key, **local["rows"][key]}), flush=True)
    # Timing: 94 layers of one path per CUDA graph, in place on the verify rows of each batch.
    stream = torch.cuda.Stream()
    for batch in batches:
        rows = TOKENS_PER_REQUEST * batch
        count = torch.tensor(rows - TOKENS_PER_REQUEST, dtype=torch.int32, device=device)
        graphs = {}
        for variant in VARIANTS:
            x, residual = randn(rows, HIDDEN, scale=0.6), randn(rows, HIDDEN, scale=3.0)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                paths[variant](x, residual, bufs[variant], count)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(LAYERS):
                    paths[variant](x, residual, bufs[variant], count)
            graphs[variant] = (graph, x, residual)
            graph.replay()
        torch.cuda.synchronize()
        blocks = []
        for block in range(BLOCKS):
            times = {variant: [] for variant in VARIANTS}
            for pair in range(PAIRS):
                order = VARIANTS if (pair + block) % 2 == 0 else tuple(reversed(VARIANTS))
                for variant in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[variant][0].replay()
                    end.record()
                    end.synchronize()
                    times[variant].append(start.elapsed_time(end) * 1000.0 / LAYERS)
            blocks.append(times)
        local["us"][str(batch)] = blocks
        print(json.dumps({"rank": rank, "batch": batch, "median_us_per_layer": [{v: statistics.median(t[v]) for v in VARIANTS} for t in blocks]}), flush=True)
        del graphs
    local["elapsed_seconds"] = time.monotonic() - begin
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(local, indent=2, allow_nan=False) + "\n")


def choose(entries, expected):
    """fused: passed every case on every GPU and saved the margin at every batch in both blocks on every GPU."""
    if len(entries) != expected or any(e is None for e in entries):
        return {"winner": "control", "reason": "missing-results"}
    batches = {tuple(e["batches"]) for e in entries}
    if len(batches) != 1:
        return {"winner": "control", "reason": "inconsistent-batches"}
    batches = batches.pop()
    cases = {f"{rows}/{live}" for rows, live in check_cases(batches)}
    reason, savings = None, []
    for entry in entries:
        timed = set(entry["us"]) == {str(b) for b in batches} and all(len(blocks) == BLOCKS and all(len(t[v]) == PAIRS for t in blocks for v in VARIANTS) for blocks in entry["us"].values())
        if not timed or set(entry["rows"]) != cases or not all(entry["rows"][case]["passed"] for case in cases):
            reason = f"failed-or-untimed gpu {entry['rank']}"
            continue
        for batch in batches:
            for block, times in enumerate(entry["us"][str(batch)]):
                control, candidate = statistics.median(times["control"]), statistics.median(times["fused"])
                savings.append({"gpu": entry["rank"], "batch": batch, "block": block, "control_us": control, "candidate_us": candidate, "saving_us": control - candidate, "required_us": max(MARGIN_US, MARGIN_FRACTION * control)})
    if reason is None and not all(s["saving_us"] >= s["required_us"] for s in savings):
        reason = "below-margin"
    winner = "fused" if reason is None else "control"
    return {"winner": winner, "reason": "all-gpus-batches-blocks-faster" if reason is None else reason, "batches": list(batches), "savings": savings}


def aggregate(directory, output, env_output, gpus=4):
    entries = []
    for rank in range(gpus):
        path = directory / f"rank-{rank}.json"
        entries.append(json.loads(path.read_text()) if path.is_file() else None)
    selection = choose(entries, gpus)
    selection["env"] = {} if selection["winner"] == "control" else {"SGLANG_DECODE_MOE_INPUT": selection["winner"]}
    result = {"schema": "firmus-decode-moe-input-selection/v1", "rule": {"tokens_per_request": TOKENS_PER_REQUEST, "layers": LAYERS, "blocks": BLOCKS, "pairs": PAIRS, "margin_us": MARGIN_US, "margin_fraction": MARGIN_FRACTION}, **selection}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with env_output.open("a") as env:
        env.write("".join(f"{k}={v}\n" for k, v in selection["env"].items()))
    print(json.dumps({"winner": result["winner"], "reason": result["reason"], "env": result["env"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="GPU result directory (selection) or selection JSON (aggregate)")
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--env-output", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "source/firmus-framework.json")
    args = parser.parse_args()
    if args.aggregate:
        assert args.env_output is not None
        aggregate(args.aggregate, args.output, args.env_output)
    else:
        run_selection(args.output, args.config)
