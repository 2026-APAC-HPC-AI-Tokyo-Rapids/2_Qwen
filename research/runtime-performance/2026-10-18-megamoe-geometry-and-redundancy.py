"""Prefill-only redundant experts (Firmus plan Tier 3 item 3-5): desk calculation.

Input: the SGLang stat-mode expert distribution dumps of run 39051152456 (same as
2026-10-18-megamoe-expert-placement.py). For R redundant physical experts per layer:

1. replicate: repeatedly give one more copy to the logical expert with the largest load per copy
   (SGLang EPLB rule), at most one copy per rank (<= 4 copies);
2. pack the 128+R physical experts onto 4 ranks, (128+R)/4 per rank, copies of one expert on
   distinct ranks, greedy LPT on the aggregate load per copy (R=0 is exactly the current
   balanced_packing placement);
3. evaluate the token-weighted per-pass rank max/mean (same metric as the placement note) for
   - static: a replicated expert's tokens split evenly over its copies (any fixed split such as
     token-index parity or source-rank mapping is at best this),
   - static with every (copy, pass) padded to whole 128-row M tiles,
   - per-pass optimal: the best split any per-pass (dynamic) assignment could reach. For this
     transportation problem it equals max over rank subsets U of (load of unreplicated experts on
     U + load of replicated experts whose copies all lie in U) / |U| (Hall's condition),
   - static, out of sample: placement from two workers evaluated on the third.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

RANKS, EXPERTS, TILE = 4, 128, 128
SUBSETS = [list(u) for k in range(1, RANKS + 1) for u in itertools.combinations(range(RANKS), k)]


def load(paths):
    out = {}
    for p in paths:
        c = torch.load(p, map_location="cpu", weights_only=True)["logical_count"].numpy().astype(np.int64)
        out[p.parent.name] = c[c.sum((1, 2)) > 0]
    return out


def plan_layer(weight, r):
    copies = np.ones(EXPERTS, dtype=np.int64)
    for _ in range(r):
        copies[int(np.argmax(np.where(copies < RANKS, weight / copies, -1.0)))] += 1
    slots = (EXPERTS + r) // RANKS
    items = sorted(((weight[e] / copies[e], e) for e in range(EXPERTS) for _ in range(copies[e])), key=lambda t: (-t[0], t[1]))
    loads, held = [0.0] * RANKS, [set() for _ in range(RANKS)]
    for w, e in items:
        j = min((j for j in range(RANKS) if len(held[j]) < slots and e not in held[j]), key=loads.__getitem__)
        held[j].add(e)
        loads[j] += w
    member = np.zeros((RANKS, EXPERTS), dtype=bool)
    for j in range(RANKS):
        member[j, sorted(held[j])] = True
    assert (member.sum(0) == copies).all() and (member.sum(1) == slots).all()
    return member, copies


def weighted(maxload, total):
    """Token-weighted mean over passes of max/mean. maxload, total: (passes,)."""
    return float((maxload / (total / RANKS) * total).sum() / total.sum())


def evaluate(counts, plans):
    """counts (passes, layers, experts). Returns per-layer static, static-tile, optimal max/mean."""
    static, tiled, optimal = [], [], []
    for l, (member, copies) in enumerate(plans):
        c = counts[:, l, :].astype(np.float64)
        total = c.sum(1)
        m = member.T.astype(np.float64)
        share = c / copies
        static.append(weighted((share @ m).max(1), total))
        tiled_load = (np.ceil(np.ceil(share) / TILE) * TILE) @ m
        tiled.append(weighted(tiled_load.max(1), tiled_load.sum(1)))
        rep = copies > 1
        single = c[:, ~rep] @ m[~rep]
        best = np.zeros(c.shape[0])
        for u in SUBSETS:
            inside = rep & (member[u].sum(0) == copies)
            best = np.maximum(best, (single[:, u].sum(1) + c[:, inside].sum(1)) / len(u))
        optimal.append(weighted(best, total))
    return np.array(static), np.array(tiled), np.array(optimal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", type=Path)
    ap.add_argument("--report-out", type=Path)
    a = ap.parse_args()
    data = load(a.dumps)
    names = sorted(data)
    allc = np.concatenate([data[w] for w in names], axis=0)
    layers = allc.shape[1]
    agg = allc.sum(0).astype(np.float64)
    # One expert's FP8 weights + block128 FP32 scales: w13 [3072, 4096], w2 [4096, 1536].
    expert_bytes = 3072 * 4096 + 4096 * 1536 + 4 * (24 * 32 + 32 * 12)
    report = {"passes": int(allc.shape[0]), "layers": layers, "expert_bytes_per_layer": expert_bytes, "R": {}}
    for r in (0, 4, 8, 12):
        plans = [plan_layer(agg[l], r) for l in range(layers)]
        static, tiled, optimal = evaluate(allc, plans)
        heldout = []
        for held in names:
            train = sum(data[w].sum(0) for w in names if w != held).astype(np.float64)
            heldout.append(float(evaluate(data[held], [plan_layer(train[l], r) for l in range(layers)])[0].mean()))
        report["R"][r] = {
            "slots_per_rank": (EXPERTS + r) // RANKS,
            "extra_vram_gb_per_rank": round(r // RANKS * layers * expert_bytes / 1e9, 3),
            "replicated_experts_per_layer_mean": float(np.mean([(p[1] > 1).sum() for p in plans])),
            "static_mean": float(static.mean()), "static_p90": float(np.percentile(static, 90)), "static_max": float(static.max()),
            "static_tile128_mean": float(tiled.mean()),
            "static_heldout_mean": heldout,
            "per_pass_optimal_mean": float(optimal.mean()), "per_pass_optimal_max": float(optimal.max()),
            "per_layer_static": [round(float(x), 4) for x in static],
            "per_layer_per_pass_optimal": [round(float(x), 4) for x in optimal],
        }
        print(r, json.dumps({k: v for k, v in report["R"][r].items() if not k.startswith("per_layer")}), flush=True)
    if a.report_out:
        a.report_out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
