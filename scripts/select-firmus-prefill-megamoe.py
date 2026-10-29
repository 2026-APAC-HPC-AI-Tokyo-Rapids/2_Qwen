"""Gate and select the four-stage Prefill MegaMoE on every Prefill GPU before model load.

Run under torchrun on one Prefill worker's four GPUs, with the serving image,
host extension and DeepGEMM header overlays. --ranks 1 makes every GPU its own
one-rank EP group (PP4/EP1 Prefill, DG_SM90_PREFILL_LOCAL_EP1); --ranks 4 runs
the four GPUs as one EP4 group. For two routings the control (three stages)
must match an FP32 reference on sampled tokens, the candidate
(DG_SM90_PREFILL_4STAGE=1: fourth A/B stage, per-warp FC2 staging) must
reproduce the control output bit for bit, and both are timed in alternating
pairs. --aggregate adopts the candidate only if every GPU of every worker
passed and was faster by the margin on both routings; anything missing keeps
the control. A control reference failure stops the job.
"""
import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

HIDDEN, INTERMEDIATE, EXPERTS, TOPK = 4096, 1536, 128, 8
SWITCH = "DG_SM90_PREFILL_4STAGE"
ROUTINGS = ("balanced", "hot")
PAIRS = 12
SAMPLED_TOKENS = 32
MARGIN_US, MARGIN_FRACTION = 5.0, 0.005
# Same tolerances as the serving gate (check-firmus-megamoe.py).
MAX_RELATIVE_L2, MAX_SCALED_ERROR = 0.025, 0.08


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
        os.environ[SWITCH] = "1" if variant == "candidate" else "0"
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
        control, candidate = (torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=device) for _ in range(2))
        forward("control", hidden, ids, weights, control)
        forward("candidate", hidden, ids, weights, candidate)
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
        observed = control[sample].float()
        expected = reference.to(torch.bfloat16).float()
        result = {
            "finite": bool(torch.isfinite(control).all().item()),
            "relative_l2": ((observed - expected).norm() / expected.norm().clamp_min(1e-10)).item(),
            "max_scaled_error": ((observed - expected).abs().max() / expected.abs().max().clamp_min(1e-10)).item(),
            "bitwise_equal": bool(torch.equal(control, candidate)),
            "control_us": [],
            "candidate_us": [],
        }
        result["control_reference_passed"] = result["finite"] and result["relative_l2"] < MAX_RELATIVE_L2 and result["max_scaled_error"] < MAX_SCALED_ERROR
        for variant in ("control", "candidate", "control", "candidate"):
            forward(variant, hidden, ids, weights, control if variant == "control" else candidate)
        for pair in range(PAIRS):
            order = ("control", "candidate") if pair % 2 == 0 else ("candidate", "control")
            for variant in order:
                result[f"{variant}_us"].append(timed(variant, hidden, ids, weights, control if variant == "control" else candidate))
        local["routings"][kind] = result
        print(json.dumps({"rank": rank, "routing": kind, **{k: v for k, v in result.items() if not k.endswith("_us")}, "control_median_us": statistics.median(result["control_us"]), "candidate_median_us": statistics.median(result["candidate_us"])}), flush=True)
    local["elapsed_seconds"] = time.monotonic() - begin
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(local, indent=2, allow_nan=False) + "\n")
    buf.destroy()
    dist.destroy_process_group()


def choose(entries, expected):
    """Adopt only with every expected GPU present, exact, and faster on every routing."""
    if any(e is None for e in entries) or len(entries) != expected:
        return {"winner": "control", "reason": "missing-results"}
    for entry in entries:
        for kind in ROUTINGS:
            result = entry["routings"][kind]
            if not result["control_reference_passed"]:
                raise SystemExit(f"Prefill MegaMoE control failed the FP32 reference: rank {entry['global_rank']} routing {kind}: {result}")
    savings = []
    for entry in entries:
        for kind in ROUTINGS:
            result = entry["routings"][kind]
            if not result["bitwise_equal"] or len(result["control_us"]) != PAIRS or len(result["candidate_us"]) != PAIRS:
                return {"winner": "control", "reason": f"candidate-not-exact-or-untimed rank {entry['global_rank']} {kind}"}
            control, candidate = statistics.median(result["control_us"]), statistics.median(result["candidate_us"])
            savings.append({"rank": entry["global_rank"], "routing": kind, "control_median_us": control, "candidate_median_us": candidate, "saving_us": control - candidate, "required_us": max(MARGIN_US, MARGIN_FRACTION * control)})
    adopted = all(s["saving_us"] >= s["required_us"] for s in savings)
    return {"winner": "four-stage" if adopted else "control", "reason": "all-gpus-faster" if adopted else "below-margin", "savings": savings}


def aggregate(directory, output, env_output, workers):
    entries = []
    for worker in range(workers):
        for rank in range(4):
            path = directory / f"prefill-{worker}" / f"rank-{rank}.json"
            entry = json.loads(path.read_text()) if path.is_file() else None
            entries.append(entry and {**entry, "global_rank": worker * 4 + rank})
    selection = choose(entries, workers * 4)
    selection["env"] = {SWITCH: "1"} if selection["winner"] == "four-stage" else {}
    result = {"schema": "firmus-prefill-megamoe-selection/v1", "rule": {"pairs": PAIRS, "margin_us": MARGIN_US, "margin_fraction": MARGIN_FRACTION, "routings": ROUTINGS, "bitwise_equal_to_control": True}, **selection}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with env_output.open("a") as env:
        env.write("".join(f"{k}={v}\n" for k, v in selection["env"].items()))
    print(json.dumps({k: v for k, v in result.items() if k != "savings"}), flush=True)


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
