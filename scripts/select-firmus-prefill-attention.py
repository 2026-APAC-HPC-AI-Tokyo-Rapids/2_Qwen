"""Gate and select two Prefill attention-block paths before model load.

Run under torchrun on four GPUs of the serving image with the SGLang and
DeepGEMM overlays (no process group: every GPU checks and times alone).

qknorm: control (apply_qknorm_rope_fp8_prefill: BF16 rounding after RMSNorm and
after RoPE, NOSAT repair on every FP8 pair) or fp32
(apply_qknorm_rope_fp8_prefill_fp32: FP32 from the BF16 QKV to the FP8 outputs,
hardware SATFINITE unless a lane holds a value past the NOSAT midpoint). Both
write FP8 Q, contiguous K/V and paged K/V. Gate, on the PP4 microbatch row
counts and the 2048-row floor, int64 and int32 positions/slots, with rows of
large, infinite and NaN values: V (contiguous and paged) and every NaN position
equal control bit for bit; finite Q and K equal the FP64 RMSNorm + RoPE rounded
once to E4M3 on at least MIN_EXACT of the elements and within one E4M3 step
everywhere, with a relative L2 error to that reference not above control's;
paged K equals contiguous K; slot 0 and slots not addressed stay untouched.

dense: the QKV (N 9216, K 4096) and O (N 4096, K 8192) DeepGEMM 1D2D calls with
compiled_dims "nk" (control) or "n" (schedule: the deep-gemm overlay offsets
the two math warpgroups by one WGMMA batch and skips WGMMA waves past the last
row). Gate: both outputs bit for bit equal to control on every row count.

Timing: one CUDA graph of 24 layers per path at 7988 rows (qknorm: one call per
layer; dense: QKV and O per layer), rotated for 12 rounds. --aggregate adopts a
candidate when it passed everywhere and saved max(3 us, 2%) per layer on every
GPU; anything missing keeps control.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

Q_HEADS, KV_HEADS, DIM, EPS, THETA, HIDDEN = 64, 4, 128, 1e-6, 1000000.0, 4096
# PP4 microbatches are 7988 rows (2 x 3994; 7989 and 3994 also occur, prefill-0 of run 40825417205).
ROWS = (7988, 7989, 3994, 2049)
TIMED_ROWS, LAYERS, ROUNDS = 7988, 24, 12
MARGIN_US, MARGIN_FRACTION = 3.0, 0.02
# Rounding the FP64 value once can differ from FP32 arithmetic only at E4M3 midpoints.
MIN_EXACT = 0.9999
SENTINEL = 0x5A
# selection -> (control, candidate, environment switch)
SELECTIONS = {"qknorm": ("control", "fp32", "SGLANG_PREFILL_QKNORM"), "dense": ("control", "schedule", "SGLANG_PREFILL_DENSE_GEMM")}
PROJECTIONS = {"QKV": (Q_HEADS * DIM + 2 * KV_HEADS * DIM, HIDDEN), "O": (HIDDEN, Q_HEADS * DIM)}


def cases():
    return {"qknorm": [f"{rows}-{dtype}" for rows in ROWS for dtype in ("int64", "int32")], "dense": [str(rows) for rows in ROWS]}


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
    if variant == "control":
        kernels["control"](*args, None, 0)
    else:
        kernels["fp32"](*args)


def qknorm_reference(torch, inputs):
    """FP64 RMSNorm (weights after the scale) and NeoX RoPE of Q and K, one rounding to E4M3 via FP32."""
    qkv = inputs["qkv"].double()
    trig = inputs["trig"].double()[inputs["positions"].long()]
    cos, sin = trig[:, None, :DIM // 2], trig[:, None, DIM // 2:]
    result = []
    for x, w in ((qkv[:, :Q_HEADS], inputs["weights"][0]), (qkv[:, Q_HEADS:Q_HEADS + KV_HEADS], inputs["weights"][1])):
        n = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * w.double()
        a, b = n[..., :DIM // 2], n[..., DIM // 2:]
        result.append(torch.cat((a * cos - b * sin, a * sin + b * cos), -1).float().to(torch.float8_e4m3fn))
    return result


def e4m3_steps(torch, a, b):
    """Distance in E4M3 codes between finite values (sign-magnitude to a monotone integer line)."""
    def line(x):
        bits = x.view(torch.uint8).int()
        return torch.where(bits >= 128, -(bits - 128), bits)
    return (line(a) - line(b)).abs()


def check_qknorm(torch, kernels, inputs, outs):
    """Run both paths on the same inputs and judge fp32 against control and the FP64 reference."""
    for variant in ("control", "fp32"):
        qknorm_call(kernels, variant, inputs, outs[variant])
    torch.cuda.synchronize()
    ctrl, cand = outs["control"], outs["fp32"]
    def u8(t):
        return t.view(torch.uint8)

    slots = inputs["slots"].long()
    used = slots[slots != 0]
    untouched = torch.ones(inputs["pool"], dtype=torch.bool, device=slots.device)
    untouched[used] = False
    result = {
        "v_bitwise": torch.equal(u8(cand["v"]), u8(ctrl["v"])) and torch.equal(u8(cand["v_cache"]), u8(ctrl["v_cache"])),
        "paged_k_matches_contiguous": torch.equal(u8(cand["k_cache"][used]), u8(cand["k"][slots != 0])),
        "untouched_slots": bool((u8(cand["k_cache"][untouched]) == SENTINEL).all().item()) and bool((u8(cand["v_cache"][untouched]) == SENTINEL).all().item()),
        "nan_pattern_equal": all(torch.equal(cand[name].float().isnan(), ctrl[name].float().isnan()) for name in ("q", "k", "v")),
    }
    for name, ref in zip(("q", "k"), qknorm_reference(torch, inputs)):
        finite = ref.float().isfinite() & ctrl[name].float().isfinite()
        steps = e4m3_steps(torch, cand[name][finite], ref[finite])
        control_steps = e4m3_steps(torch, ctrl[name][finite], ref[finite])
        r = ref.float()[finite]
        errors = [((x.float()[finite] - r).norm() / r.norm().clamp_min(1e-30)).item() for x in (cand[name], ctrl[name])]
        result[name] = {"exact_fraction": (steps == 0).float().mean().item(), "max_steps": int(steps.max().item()), "control_exact_fraction": (control_steps == 0).float().mean().item(), "control_max_steps": int(control_steps.max().item()), "relative_l2": errors[0], "control_relative_l2": errors[1]}
    result["passed"] = result["v_bitwise"] and result["paged_k_matches_contiguous"] and result["untouched_slots"] and result["nan_pattern_equal"] and all(
        result[name]["exact_fraction"] >= MIN_EXACT and result[name]["max_steps"] <= 1 and result[name]["relative_l2"] <= result[name]["control_relative_l2"] for name in ("q", "k"))
    return result


def make_dense_inputs(torch, device, generator, rows, quantize):
    """FP8 activations with column-major TMA-aligned group scales (as the input norm and FA3 write them) and FP8 weights."""
    def activation(k):
        return quantize((torch.randn(rows, k, device=device, generator=generator) * 2).to(torch.bfloat16), 128, column_major_scales=True, scale_tma_aligned=True)

    def weight(n, k):
        w = (torch.randn(n, k, device=device, generator=generator) * 0.05).to(torch.float8_e4m3fn)
        return w, torch.rand(n // 128, k // 128, device=device, generator=generator) * 0.01 + 0.001

    return {name: (activation(k), weight(n, k), torch.empty(rows, n, device=device, dtype=torch.bfloat16)) for name, (n, k) in PROJECTIONS.items()}


def dense_call(deep_gemm, variant, inputs):
    for a, w, out in inputs.values():
        deep_gemm.fp8_gemm_nt(a, w, out, compiled_dims="n" if variant == "schedule" else "nk")


def check_dense(torch, deep_gemm, inputs):
    outputs = {}
    for variant in ("control", "schedule"):
        for _, _, out in inputs.values():
            out.fill_(float("nan"))
        dense_call(deep_gemm, variant, inputs)
        outputs[variant] = {name: out.clone() for name, (_, _, out) in inputs.items()}
    torch.cuda.synchronize()
    result = {name: {"bitwise": torch.equal(outputs["schedule"][name].view(torch.int16), outputs["control"][name].view(torch.int16)), "finite": bool(outputs["control"][name].isfinite().all().item())} for name in PROJECTIONS}
    result["passed"] = all(result[name]["bitwise"] and result[name]["finite"] for name in PROJECTIONS)
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
    import deep_gemm
    import torch
    from sglang.kernels.ops.attention.rope import (
        apply_qknorm_rope_fp8_prefill,
        apply_qknorm_rope_fp8_prefill_fp32,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    device = torch.device("cuda", rank)
    generator = torch.Generator(device=device)
    generator.manual_seed(1030 + rank)
    kernels = {"control": apply_qknorm_rope_fp8_prefill, "fp32": apply_qknorm_rope_fp8_prefill_fp32}
    local = {"rank": rank, "device": torch.cuda.get_device_name(rank), "uuid": str(torch.cuda.get_device_properties(rank).uuid), "cases": {"qknorm": {}, "dense": {}}, "us": {}}
    for rows in ROWS:
        for id_dtype in (torch.int64, torch.int32):
            inputs = make_qknorm_inputs(torch, device, generator, rows, id_dtype)
            outs = {variant: make_qknorm_outputs(torch, device, rows, inputs["pool"]) for variant in ("control", "fp32")}
            key = f"{rows}-{str(id_dtype).split('.')[-1]}"
            local["cases"]["qknorm"][key] = check_qknorm(torch, kernels, inputs, outs)
            print(json.dumps({"rank": rank, "qknorm": key, **local["cases"]["qknorm"][key]}), flush=True)
        dense = make_dense_inputs(torch, device, generator, rows, sglang_per_token_group_quant_fp8)
        local["cases"]["dense"][str(rows)] = check_dense(torch, deep_gemm, dense)
        print(json.dumps({"rank": rank, "dense": rows, **local["cases"]["dense"][str(rows)]}), flush=True)
    # Timing on 7988-row int64 inputs.
    inputs = make_qknorm_inputs(torch, device, generator, TIMED_ROWS, torch.int64)
    outs = {variant: make_qknorm_outputs(torch, device, TIMED_ROWS, inputs["pool"]) for variant in ("control", "fp32")}
    local["us"]["qknorm"] = time_graphs(torch, {variant: (lambda v=variant: qknorm_call(kernels, v, inputs, outs[v])) for variant in ("control", "fp32")})
    dense = make_dense_inputs(torch, device, generator, TIMED_ROWS, sglang_per_token_group_quant_fp8)
    local["us"]["dense"] = time_graphs(torch, {variant: (lambda v=variant: dense_call(deep_gemm, v, dense)) for variant in ("control", "schedule")})
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
