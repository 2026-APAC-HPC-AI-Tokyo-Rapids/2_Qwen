"""Gate and select the Prefill MegaMoE pipeline variant on every Prefill GPU before model load.

Run under torchrun on one Prefill worker's four GPUs, with the serving image,
host extension and DeepGEMM header overlays. --ranks 1 makes every GPU its own
one-rank EP group (PP4/EP1 Prefill, DG_SM90_PREFILL_LOCAL_EP1); --ranks 4 runs
the four GPUs as one EP4 group. Variants: control (three stages), four-stage
(DG_SM90_PREFILL_4STAGE=1: fourth A/B stage, per-warp FC2 staging),
four-stage-lean (also DG_SM90_PREFILL_LEAN=1, EP1 only: no cross-rank count
publication, rank round robin or post-signal grid sync, FC2 staging with
stmatrix), four-stage-lean-pair (also DG_SM90_PREFILL_TAIL_PAIR=1: pool
blocks with at most 64 rows run their N blocks in pairs on both warpgroups),
four-stage-lean-split (lean plus DG_SM90_PREFILL_SPLIT_ACCUM=1: each warpgroup's
FC1/FC2 accumulator is issued as two m64n128 halves so that the carried-scale
rescale of one half overlaps the other half's WGMMA) and four-stage-lean-split-ldsm
(also DG_SM90_PREFILL_SPLIT_LDSM=1: the split halves take A from registers loaded
once per K block by ldmatrix, so the WGMMAs read A once instead of twice).
The round-2 direct-FC2 and A-gather variants lost (+1.6% and +44% in
40911121619) and the tail split stayed below the margin twice; they are gone.
For the two timed routings and the untimed edge routing (experts with 0, 1-64,
65, 127-129 and 255-256 rows) the control must match an FP32 reference on
sampled tokens and every candidate must reproduce the control bit for bit.
All six are timed in rotating order. --aggregate adopts the fastest variant that passed on
every GPU of every worker and saved at least the summed per-case margin over
all GPUs and routings; anything missing keeps the control. A control reference
failure stops the job. Neither per-case nor per-GPU margins are used: in runs
40849459077 and 40883389891 four-stage was bit-exact and saved 2.3% and 2.1%
in total, while single cases (+3.6, -12.9 us) and single GPUs (GPU 3: +108 us,
then -33 us) moved within the timing noise of one GPU.

Endurance gate (--endurance, one torchrun per Prefill worker after every
worker's selection): run 40991744289 adopted four-stage-lean-split, and all
three Prefill workers stopped after about 13 microbatches (about 300 MegaMoE
calls per GPU) with no CUDA error; the selection makes only about 30 calls per
GPU. Each GPU recomputes the winner the aggregate will pick from the same rank
files and, if it is a variant that has not served yet, runs it for
ENDURANCE_CALLS consecutive calls rotating over the balanced, hot and edge
routings, then checks that the last output of every routing equals the
control's bit for bit. --aggregate keeps such a winner only if all twelve GPUs
ran it to the end and matched; otherwise it takes the fastest eligible variant
that has served (UNGATED) and records why.
"""
import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

HIDDEN, INTERMEDIATE, EXPERTS, TOPK = 4096, 1536, 128, 8
SWITCHES = ("DG_SM90_PREFILL_4STAGE", "DG_SM90_PREFILL_LEAN", "DG_SM90_PREFILL_TAIL_PAIR", "DG_SM90_PREFILL_SPLIT_ACCUM", "DG_SM90_PREFILL_SPLIT_LDSM")
VARIANTS = {name: dict(zip(SWITCHES, values, strict=True)) for name, values in {
    "control": ("0", "0", "0", "0", "0"),
    "four-stage": ("1", "0", "0", "0", "0"),
    "four-stage-lean": ("1", "1", "0", "0", "0"),
    "four-stage-lean-pair": ("1", "1", "1", "0", "0"),
    "four-stage-lean-split": ("1", "1", "0", "1", "0"),
    "four-stage-lean-split-ldsm": ("1", "1", "0", "1", "1"),
}.items()}
# Every candidate changes staging, synchronisation, the work assignment or the WGMMA
# issue order only (same products, K order and multipliers per element): bit-exact.
CANDIDATES = ("four-stage", "four-stage-lean", "four-stage-lean-pair", "four-stage-lean-split", "four-stage-lean-split-ldsm")
# Variants that have served whole runs need no endurance gate: four-stage-lean-split passed the
# gate and served 6212 requests in 41005872470 (9832.74), four-stage-lean served 12k+ Prefill
# batches in 40964714831 (9668.27), four-stage and the control before them.
UNGATED = ("control", "four-stage", "four-stage-lean", "four-stage-lean-split")
# Over 300 calls per GPU stopped serving in 40991744289; 600 at about 2.5 ms is about 1.5 s.
ENDURANCE_CALLS = 600
ROUTINGS = ("balanced", "hot")
# Checked for exactness only (not timed): the pool-block shapes the timed routings rarely hit.
EXACT_ROUTINGS = ("edges",)
# Rows of experts 16..47 in the edge routing; experts 0..15 get none.
EDGE_EXPERT_ROWS = (1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 66, 127, 128, 129, 191, 192, 193, 255, 256, 1, 63, 64, 65, 1, 2, 3, 4)
ROUNDS = 12
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
    if kind == "edges":
        # Experts 0..47 are never in the top-k; experts 16..47 then replace the last slot
        # of disjoint token ranges (distinct from the other seven) with EDGE_EXPERT_ROWS rows.
        scores[:, :48] = -1.0
    weights, ids = scores.topk(TOPK, dim=1)
    if kind == "edges":
        row = 0
        for expert, rows in enumerate(EDGE_EXPERT_ROWS, start=16):
            ids[row:row + rows, TOPK - 1] = expert
            row += rows
        assert row <= count
    return ids.to(torch.int32), weights / weights.sum(-1, keepdim=True)


def errors(torch, observed, expected):
    observed, expected = observed.float(), expected.float()
    return ((observed - expected).norm() / expected.norm().clamp_min(1e-10)).item(), ((observed - expected).abs().max() / expected.abs().max().clamp_min(1e-10)).item()


def prepare(ranks):
    """Process group, the seeded FP8 weights, the symmetric buffer and one MegaMoE call per variant."""
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

    return torch, dist, rank, device, tokens, local_experts, generator, group_rank, (w13, s13, w2, s2), buf, forward


def run_selection(output, ranks):
    begin = time.monotonic()
    torch, dist, rank, device, tokens, local_experts, generator, group_rank, (w13, s13, w2, s2), buf, forward = prepare(ranks)

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
    for kind in ROUTINGS + EXACT_ROUTINGS:
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
            result[variant] = entry
        if kind in EXACT_ROUTINGS:
            local["routings"][kind] = result
            print(json.dumps({"rank": rank, "routing": kind, **{v: {k: x for k, x in e.items() if k != "us"} for v, e in result.items()}}), flush=True)
            continue
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


def load_entries(directory, workers):
    entries = []
    for worker in range(workers):
        for rank in range(4):
            path = directory / f"prefill-{worker}" / f"rank-{rank}.json"
            entry = json.loads(path.read_text()) if path.is_file() else None
            entries.append(entry and {**entry, "global_rank": worker * 4 + rank})
    return entries


def run_endurance(selection, output, workers):
    """Run the winner the aggregate would pick, if it has not served, for ENDURANCE_CALLS calls."""
    begin = time.monotonic()
    rank = int(os.environ["LOCAL_RANK"])
    winner = choose(load_entries(selection, workers), workers * 4)["winner"]
    local = {"rank": rank, "variant": winner, "calls": 0, "needed": winner not in UNGATED}
    if local["needed"]:
        torch, dist, rank, device, tokens, _, generator, _, _, buf, forward = prepare(1)
        kinds = ROUTINGS + EXACT_ROUTINGS
        inputs = {kind: (torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device=device, generator=generator) * 0.25, *routing(torch, kind, tokens, generator, device)) for kind in kinds}
        reference = {kind: torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=device) for kind in kinds}
        outputs = {kind: torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=device) for kind in kinds}
        for kind in kinds:
            forward("control", *inputs[kind], reference[kind])
        torch.cuda.synchronize()
        start = time.monotonic()
        # Back to back as in serving, rotating the routings; each output buffer keeps its routing's last call.
        for call in range(ENDURANCE_CALLS):
            kind = kinds[call % len(kinds)]
            forward(winner, *inputs[kind], outputs[kind])
        torch.cuda.synchronize()
        local["calls"] = ENDURANCE_CALLS
        local["calls_seconds"] = time.monotonic() - start
        local["bitwise_equal"] = {kind: bool(torch.equal(outputs[kind], reference[kind])) for kind in kinds}
        buf.destroy()
        dist.destroy_process_group()
    local["passed"] = not local["needed"] or all(local["bitwise_equal"].values())
    local["elapsed_seconds"] = time.monotonic() - begin
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(local, indent=2, allow_nan=False) + "\n")
    print(json.dumps(local), flush=True)


def endurance_failure(gate, winner, expected):
    """None if every GPU ran the winner for ENDURANCE_CALLS calls and matched the control, else why not."""
    if gate is None or len(gate) != expected:
        return "no endurance results"
    for global_rank, entry in enumerate(gate):
        if entry is None:
            return f"no endurance result from rank {global_rank} (late, hung or failed)"
        if entry["variant"] != winner:
            return f"rank {global_rank} gated {entry['variant']}, not {winner}"
        if entry["calls"] != ENDURANCE_CALLS or not entry["passed"]:
            return f"rank {global_rank} failed the endurance gate: {entry.get('calls')} calls, bitwise_equal {entry.get('bitwise_equal')}"
    return None


def candidate_passed(entry, timed):
    return entry["reference_passed"] and entry["bitwise_equal"] and (len(entry["us"]) == ROUNDS or not timed)


def choose(entries, expected, gate=None):
    """Adopt the fastest variant that passed everywhere and beat the summed margin.

    A winner outside UNGATED also needs the endurance gate on every GPU (gate: the
    per-GPU endurance results, or None before the gate has run)."""
    if any(e is None for e in entries) or len(entries) != expected:
        return {"winner": "control", "reason": "missing-results"}
    for entry in entries:
        for kind in ROUTINGS + EXACT_ROUTINGS:
            control = entry["routings"][kind]["control"]
            if not control["reference_passed"]:
                raise SystemExit(f"Prefill MegaMoE control failed the FP32 reference: rank {entry['global_rank']} routing {kind}: {control}")
    choices = {}
    for name in CANDIDATES:
        savings, reason = [], None
        for entry in entries:
            for kind in EXACT_ROUTINGS:
                if not candidate_passed(entry["routings"][kind][name], False):
                    reason = f"failed rank {entry['global_rank']} {kind}"
            for kind in ROUTINGS:
                result = entry["routings"][kind]
                if len(result["control"]["us"]) != ROUNDS or not candidate_passed(result[name], True):
                    reason = f"failed-or-untimed rank {entry['global_rank']} {kind}"
                    continue
                control, candidate = statistics.median(result["control"]["us"]), statistics.median(result[name]["us"])
                savings.append({"rank": entry["global_rank"], "routing": kind, "control_median_us": control, "candidate_median_us": candidate, "saving_us": control - candidate, "required_us": max(MARGIN_US, MARGIN_FRACTION * control)})
        if reason is None and sum(s["saving_us"] for s in savings) < sum(s["required_us"] for s in savings):
            reason = "below-margin"
        choices[name] = {"eligible": reason is None, "reason": reason or "faster-in-total", "total_median_us": sum(s["candidate_median_us"] for s in savings), "savings": savings}
    eligible = sorted((name for name in CANDIDATES if choices[name]["eligible"]), key=lambda name: (choices[name]["total_median_us"], CANDIDATES.index(name)))
    if not eligible:
        return {"winner": "control", "reason": "no-eligible-candidate", "choices": choices}
    if gate is None or eligible[0] in UNGATED:
        return {"winner": eligible[0], "reason": "fastest-eligible", "choices": choices}
    failure = endurance_failure(gate, eligible[0], expected)
    if failure is None:
        return {"winner": eligible[0], "reason": "fastest-eligible, passed the endurance gate", "choices": choices}
    choices[eligible[0]]["endurance"] = failure
    served = [name for name in eligible if name in UNGATED]
    return {"winner": served[0] if served else "control", "reason": f"{eligible[0]} not adopted: {failure}", "choices": choices}


def aggregate(directory, output, env_output, workers, endurance):
    gate = [entry and {k: v for k, v in entry.items() if k != "global_rank"} for entry in load_entries(endurance, workers)]
    selection = choose(load_entries(directory, workers), workers * 4, gate)
    selection["env"] = {k: v for k, v in VARIANTS[selection["winner"]].items() if v != "0"}
    result = {"schema": "firmus-prefill-megamoe-selection/v4", "rule": {"rounds": ROUNDS, "margin_us": MARGIN_US, "margin_fraction": MARGIN_FRACTION, "routings": ROUTINGS, "exact_routings": EXACT_ROUTINGS, "bit_exact": True, "endurance_calls": ENDURANCE_CALLS, "ungated": UNGATED}, **selection, "endurance": gate}
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
    parser.add_argument("--endurance", type=Path, help="per-worker endurance results (aggregate), or the selection results to gate (endurance run)")
    parser.add_argument("--workers", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    if args.aggregate:
        assert args.env_output is not None and args.endurance is not None
        aggregate(args.aggregate, args.output, args.env_output, args.workers, args.endurance)
    elif args.endurance:
        run_endurance(args.endurance, args.output, args.workers)
    else:
        run_selection(args.output, args.ranks)
