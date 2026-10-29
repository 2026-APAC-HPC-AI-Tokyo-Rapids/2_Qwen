"""Gate and select the Prefill MegaMoE pipeline variant on every Prefill GPU before model load.

Run under torchrun on one Prefill worker's four GPUs, with the serving image,
host extension and DeepGEMM header overlays. --ranks 1 makes every GPU its own
one-rank EP group (PP4/EP1 Prefill, DG_SM90_PREFILL_LOCAL_EP1); --ranks 4 runs
the four GPUs as one EP4 group. Variants: control (three stages), four-stage
(DG_SM90_PREFILL_4STAGE=1: fourth A/B stage, per-warp FC2 staging) and
four-stage-split (also DG_SM90_PREFILL_TAIL_SPLIT=1: pool blocks with at most
64 rows on both warpgroups as m64n128 halves). For two routings the control
must match an FP32 reference on sampled tokens; four-stage must reproduce the
control bit for bit; four-stage-split must do so or stay within FP-order
tolerance of the control on every token and pass the reference. All three are
timed in rotating order. --aggregate adopts the fastest variant that passed on
every GPU of every worker and was faster than the control by the margin on
both routings; anything missing keeps the control. A control reference failure
stops the job.
"""
import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

HIDDEN, INTERMEDIATE, EXPERTS, TOPK = 4096, 1536, 128, 8
VARIANTS = {
    "control": {"DG_SM90_PREFILL_4STAGE": "0", "DG_SM90_PREFILL_TAIL_SPLIT": "0"},
    "four-stage": {"DG_SM90_PREFILL_4STAGE": "1", "DG_SM90_PREFILL_TAIL_SPLIT": "0"},
    "four-stage-split": {"DG_SM90_PREFILL_4STAGE": "1", "DG_SM90_PREFILL_TAIL_SPLIT": "1"},
}
CANDIDATES = ("four-stage", "four-stage-split")
# The split changes the WGMMA width only; anything beyond FP-order noise is a defect.
INEXACT_ALLOWED = {"four-stage": False, "four-stage-split": True}
ROUTINGS = ("balanced", "hot")
ROUNDS = 12
SAMPLED_TOKENS = 32
MARGIN_US, MARGIN_FRACTION = 5.0, 0.005
# Same tolerances as the serving gate (check-firmus-megamoe.py).
MAX_RELATIVE_L2, MAX_SCALED_ERROR = 0.025, 0.08
# Candidate against the control on every token, when not bit-identical.
MAX_RELATIVE_L2_VS_CONTROL, MAX_SCALED_ERROR_VS_CONTROL = 0.002, 0.02


def quantized(torch, x, group):
    blocks = x.float().reshape(x.shape[0], -1, group)
    scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape)


def dequantized(w, scale):
    return w.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)


def reference_rows(torch, x, ids, weights, w13, s13, w2, s2, first_expert):
    """Locally owned experts' FP32 contribution: FP8 activation groups of 128, BF16 terms, FP32 top-k sum."""
    out = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    xq = quantized(torch, x, 128)
    for local in range(w13.shape[0]):
        rows, slots = torch.where(ids == first_expert + local)
        if rows.numel() == 0:
            continue
        gate, up = (xq[rows] @ dequantized(w13[local], s13[local]).T).chunk(2, dim=-1)
        term = quantized(torch, torch.nn.functional.silu(gate) * up, 128) @ dequantized(w2[local], s2[local]).T
        out.index_add_(0, rows, term.to(torch.bfloat16).float() * weights[rows, slots, None])
    return out


def routing(torch, kind, count, generator, device):
    scores = torch.rand(count, EXPERTS, generator=generator, device=device)
    if kind == "hot":
        # Serving captures route almost every token of a batch to one expert
        # (kernels/probes/megamoe-local-ep1-routing.npz: up to 7975 of 8002).
        scores[:, 0] += 1.0
    weights, ids = scores.topk(TOPK, dim=1)
    return ids.to(torch.int32), weights / weights.sum(-1, keepdim=True)


def errors(torch, observed, expected):
    observed, expected = observed.float(), expected.float()
    return ((observed - expected).norm() / expected.norm().clamp_min(1e-10)).item(), ((observed - expected).abs().max() / expected.abs().max().clamp_min(1e-10)).item()


def run_selection(output, ranks):
    begin = time.monotonic()
    import deep_gemm
    import torch
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=240), device_id=device)
    assert dist.get_world_size() == 4
    # Every rank must create every one-rank group, in the same order.
    singles = [dist.new_group([r]) for r in range(4)] if ranks == 1 else None
    group = singles[rank] if ranks == 1 else dist.group.WORLD
    group_rank = 0 if ranks == 1 else rank
    os.environ["DG_SM90_PREFILL_2WG"] = "1"
    os.environ["DG_SM90_PREFILL_LOCAL_EP1"] = "1" if ranks == 1 else "0"
    tokens = 8064 if ranks == 1 else 4096
    local_experts = EXPERTS // ranks
    generator = torch.Generator(device=device)
    generator.manual_seed(1029 + rank)
    w13 = (torch.randn(local_experts, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=device, generator=generator) * 4).to(torch.float8_e4m3fn)
    w2 = (torch.randn(local_experts, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device=device, generator=generator) * 4).to(torch.float8_e4m3fn)
    s13 = torch.rand(local_experts, 2 * INTERMEDIATE // 128, HIDDEN // 128, device=device, generator=generator) * 0.005 + 0.005
    s2 = torch.rand(local_experts, HIDDEN // 128, INTERMEDIATE // 128, device=device, generator=generator) * 0.005 + 0.005
    l1, l2 = deep_gemm.transform_weights_for_mega_moe_sm90((w13, s13), (w2, s2))
    buf = deep_gemm.get_symm_buffer_for_mega_moe(group, EXPERTS, 8192 if ranks == 1 else 4096, TOPK, HIDDEN, INTERMEDIATE, use_fp8_dispatch=True, activation="swiglu")

    def forward(variant, hidden, ids, weights, out, start=None):
        os.environ.update(VARIANTS[variant])
        deep_gemm.mega_moe_pre_dispatch_sm90(hidden, ids, weights, buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights, num_tokens=tokens, group_size=128, routed_scaling_factor=1.0)
        if start is not None:
            start.record()
        deep_gemm.fp8_mega_moe(out, l1, l2, buf, recipe=(128, 128, 128), activation="swiglu", fast_math=True)

    def timed(variant, hidden, ids, weights, out):
        """MegaMoE kernel time only: a GPU sleep keeps host launch work off the timed span."""
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if ranks == 4:
            dist.barrier()
        # About 0.7 ms at the capped clock: covers the host launch path, keeps the duty cycle near serving.
        torch.cuda._sleep(1_000_000)
        forward(variant, hidden, ids, weights, out, start)
        end.record()
        end.synchronize()
        value = torch.tensor([start.elapsed_time(end) * 1000.0], device=device)
        if ranks == 4:
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return value.item()

    local = {"rank": rank, "ranks_per_group": ranks, "tokens": tokens, "device": torch.cuda.get_device_name(rank), "uuid": str(torch.cuda.get_device_properties(rank).uuid), "routings": {}}
    for kind in ROUTINGS:
        hidden = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device=device, generator=generator) * 0.25
        ids, weights = routing(torch, kind, tokens, generator, device)
        outputs = {variant: torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=device) for variant in VARIANTS}
        for variant, out in outputs.items():
            forward(variant, hidden, ids, weights, out)
        torch.cuda.synchronize()
        # Reference on sampled tokens: every rank adds its experts' terms for every rank's samples.
        sample = torch.randperm(tokens, device=device, generator=generator)[:SAMPLED_TOKENS]
        parts = [hidden[sample], ids[sample], weights[sample]]
        if ranks == 4:
            for i, part in enumerate(parts):
                bucket = [torch.empty_like(part) for _ in range(4)]
                dist.all_gather(bucket, part)
                parts[i] = torch.cat(bucket)
        reference = reference_rows(torch, *parts, w13, s13, w2, s2, group_rank * local_experts)
        if ranks == 4:
            dist.all_reduce(reference)
            reference = reference[rank * SAMPLED_TOKENS:(rank + 1) * SAMPLED_TOKENS]
        expected = reference.to(torch.bfloat16)
        result = {}
        for variant, out in outputs.items():
            relative_l2, max_scaled_error = errors(torch, out[sample], expected)
            entry = {"finite": bool(torch.isfinite(out).all().item()), "relative_l2": relative_l2, "max_scaled_error": max_scaled_error, "us": []}
            entry["reference_passed"] = entry["finite"] and relative_l2 < MAX_RELATIVE_L2 and max_scaled_error < MAX_SCALED_ERROR
            if variant != "control":
                entry["bitwise_equal"] = bool(torch.equal(out, outputs["control"]))
                entry["relative_l2_vs_control"], entry["max_scaled_error_vs_control"] = errors(torch, out, outputs["control"])
            result[variant] = entry
        names = list(VARIANTS)
        for variant in names:
            forward(variant, hidden, ids, weights, outputs[variant])
        for round_index in range(ROUNDS):
            for offset in range(len(names)):
                variant = names[(round_index + offset) % len(names)]
                result[variant]["us"].append(timed(variant, hidden, ids, weights, outputs[variant]))
        local["routings"][kind] = result
        print(json.dumps({"rank": rank, "routing": kind, **{v: {**{k: x for k, x in e.items() if k != "us"}, "median_us": statistics.median(e["us"])} for v, e in result.items()}}), flush=True)
    local["elapsed_seconds"] = time.monotonic() - begin
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(local, indent=2, allow_nan=False) + "\n")
    buf.destroy()
    dist.destroy_process_group()


def candidate_passed(name, entry):
    if not entry["reference_passed"] or len(entry["us"]) != ROUNDS:
        return False
    if entry["bitwise_equal"]:
        return True
    return INEXACT_ALLOWED[name] and entry["relative_l2_vs_control"] < MAX_RELATIVE_L2_VS_CONTROL and entry["max_scaled_error_vs_control"] < MAX_SCALED_ERROR_VS_CONTROL


def choose(entries, expected):
    """Adopt the fastest variant that passed and beat the control by the margin on every GPU and routing."""
    if any(e is None for e in entries) or len(entries) != expected:
        return {"winner": "control", "reason": "missing-results"}
    for entry in entries:
        for kind in ROUTINGS:
            control = entry["routings"][kind]["control"]
            if not control["reference_passed"]:
                raise SystemExit(f"Prefill MegaMoE control failed the FP32 reference: rank {entry['global_rank']} routing {kind}: {control}")
    choices = {}
    for name in CANDIDATES:
        savings, reason = [], None
        for entry in entries:
            for kind in ROUTINGS:
                result = entry["routings"][kind]
                if len(result["control"]["us"]) != ROUNDS or not candidate_passed(name, result[name]):
                    reason = f"failed-or-untimed rank {entry['global_rank']} {kind}"
                    continue
                control, candidate = statistics.median(result["control"]["us"]), statistics.median(result[name]["us"])
                savings.append({"rank": entry["global_rank"], "routing": kind, "control_median_us": control, "candidate_median_us": candidate, "saving_us": control - candidate, "required_us": max(MARGIN_US, MARGIN_FRACTION * control), "bitwise_equal": result[name]["bitwise_equal"]})
        if reason is None and not all(s["saving_us"] >= s["required_us"] for s in savings):
            reason = "below-margin"
        choices[name] = {"eligible": reason is None, "reason": reason or "all-gpus-faster", "total_median_us": sum(s["candidate_median_us"] for s in savings), "savings": savings}
    eligible = [name for name in CANDIDATES if choices[name]["eligible"]]
    winner = min(eligible, key=lambda name: (choices[name]["total_median_us"], CANDIDATES.index(name))) if eligible else "control"
    return {"winner": winner, "reason": "fastest-eligible" if eligible else "no-eligible-candidate", "choices": choices}


def aggregate(directory, output, env_output, workers):
    entries = []
    for worker in range(workers):
        for rank in range(4):
            path = directory / f"prefill-{worker}" / f"rank-{rank}.json"
            entry = json.loads(path.read_text()) if path.is_file() else None
            entries.append(entry and {**entry, "global_rank": worker * 4 + rank})
    selection = choose(entries, workers * 4)
    selection["env"] = {k: v for k, v in VARIANTS[selection["winner"]].items() if v != "0"}
    result = {"schema": "firmus-prefill-megamoe-selection/v2", "rule": {"rounds": ROUNDS, "margin_us": MARGIN_US, "margin_fraction": MARGIN_FRACTION, "routings": ROUTINGS, "inexact_allowed": INEXACT_ALLOWED, "max_relative_l2_vs_control": MAX_RELATIVE_L2_VS_CONTROL, "max_scaled_error_vs_control": MAX_SCALED_ERROR_VS_CONTROL}, **selection}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with env_output.open("a") as env:
        env.write("".join(f"{k}={v}\n" for k, v in selection["env"].items()))
    print(json.dumps({"winner": result["winner"], "reason": result["reason"], "env": result["env"], **{name: {k: v for k, v in choice.items() if k != "savings"} for name, choice in result.get("choices", {}).items()}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="worker directory (selection) or selection JSON (aggregate)")
    parser.add_argument("--ranks", type=int, choices=(1, 4), default=1)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--env-output", type=Path)
    parser.add_argument("--workers", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    if args.aggregate:
        assert args.env_output is not None
        aggregate(args.aggregate, args.output, args.env_output, args.workers)
    else:
        run_selection(args.output, args.ranks)
