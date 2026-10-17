"""Prefill MegaMoE EP-rank imbalance from SGLang stat-mode expert distribution dumps."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

RANKS, EXPERTS, TOPK = 4, 128, 8
PER_RANK = EXPERTS // RANKS


def load(paths):
    out = {}
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=True)
        c = d["logical_count"].numpy().astype(np.int64)  # (passes, layers, experts)
        out[p.parent.name] = c
    return out


def balanced_packing(weight):
    """SGLang eplb deepseek.balanced_packing for one layer: greedy LPT, exactly 32 experts per rank."""
    order = np.argsort(-weight.astype(np.float64), kind="stable")
    load, items = [0.0] * RANKS, [0] * RANKS
    phys = [[] for _ in range(RANKS)]
    for e in order:
        r = min((j for j in range(RANKS) if items[j] < PER_RANK), key=load.__getitem__)
        phys[r].append(int(e))
        load[r] += float(weight[e])
        items[r] += 1
    return [e for r in range(RANKS) for e in phys[r]]  # physical_to_logical for this layer


def rank_loads(counts, p2l):
    """counts (passes, layers, experts) logical; p2l (layers, 128). Returns (passes, layers, ranks)."""
    phys = np.take_along_axis(counts, np.broadcast_to(p2l[None], counts.shape), axis=2)
    return phys.reshape(*phys.shape[:2], RANKS, PER_RANK).sum(-1)


def tile_loads(counts, p2l, tile):
    phys = np.take_along_axis(counts, np.broadcast_to(p2l[None], counts.shape), axis=2)
    padded = np.ceil(phys / tile) * tile
    return padded.reshape(*padded.shape[:2], RANKS, PER_RANK).sum(-1)


def ratio(loads):
    """Token-weighted mean over passes of max/mean, per layer. loads (passes, layers, ranks)."""
    total = loads.sum(-1)  # (passes, layers)
    mean = total / RANKS
    r = np.where(total > 0, loads.max(-1) / np.maximum(mean, 1e-9), 0.0)
    w = total
    return (r * w).sum(0) / w.sum(0)


def summarize(name, per_layer):
    return {"name": name, "mean": float(per_layer.mean()), "median": float(np.median(per_layer)), "p90": float(np.percentile(per_layer, 90)), "max": float(per_layer.max()), "argmax_layer": int(per_layer.argmax()), "min": float(per_layer.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", type=Path)
    ap.add_argument("--placement-out", type=Path)
    ap.add_argument("--report-out", type=Path)
    a = ap.parse_args()
    data = load(a.dumps)
    report = {"workers": {}}
    for w, c in data.items():
        tot = c.sum((1, 2))
        nz = tot > 0
        # Every layer sees the same tokens: per-pass sum over experts / 8 = tokens of the pass (all 4 ranks).
        tokens = c[nz].sum(2)[:, 0] // TOPK
        report["workers"][w] = {"passes_recorded": int(c.shape[0]), "passes_with_moe": int(nz.sum()), "layers": int(c.shape[1]), "tokens_total": int(tokens.sum()), "tokens_per_pass_mean": float(tokens.mean()), "tokens_per_pass_p50": float(np.median(tokens)), "layer_consistency": bool((c[nz].sum(2) == c[nz].sum(2)[:, :1]).all())}
        data[w] = c[nz]
    allc = np.concatenate(list(data.values()), axis=0)
    layers = allc.shape[1]
    trivial = np.tile(np.arange(EXPERTS), (layers, 1))
    agg = allc.sum(0)  # (layers, experts)
    placement = np.array([balanced_packing(agg[l]) for l in range(layers)])
    assert all(sorted(row) == list(range(EXPERTS)) for row in placement.tolist())
    r_triv = ratio(rank_loads(allc, trivial))
    r_new = ratio(rank_loads(allc, placement))
    agg_triv = rank_loads(agg[None], trivial)[0]
    agg_new = rank_loads(agg[None], placement)[0]
    report["per_pass_max_over_mean"] = {"trivial": summarize("trivial", r_triv), "permuted_in_sample": summarize("permuted", r_new)}
    report["aggregate_max_over_mean"] = {"trivial": summarize("trivial", agg_triv.max(-1) / agg_triv.mean(-1)), "permuted": summarize("permuted", agg_new.max(-1) / agg_new.mean(-1))}
    # Random permutation baseline: noise floor of per-pass imbalance with no systematic skew removal.
    rng = np.random.default_rng(0)
    rand = np.array([rng.permutation(EXPERTS) for _ in range(layers)])
    report["per_pass_max_over_mean"]["random_permutation"] = summarize("random", ratio(rank_loads(allc, rand)))
    # Out-of-sample checks: placement from other workers, and from the first half of passes.
    oos = {}
    names = sorted(data)
    for held in names:
        train = sum(data[w].sum(0) for w in names if w != held)
        pl = np.array([balanced_packing(train[l]) for l in range(layers)])
        oos[f"heldout_{held}"] = {"trivial": float(ratio(rank_loads(data[held], trivial)).mean()), "permuted": float(ratio(rank_loads(data[held], pl)).mean())}
    half = allc.shape[0] // 2
    pl = np.array([balanced_packing(allc[:half].sum(0)[l]) for l in range(layers)])
    oos["second_half_from_first_half"] = {"trivial": float(ratio(rank_loads(allc[half:], trivial)).mean()), "permuted": float(ratio(rank_loads(allc[half:], pl)).mean())}
    report["out_of_sample_mean_over_layers"] = oos
    # Per-expert skew (aggregate counts per layer).
    share = agg / agg.sum(1, keepdims=True)
    report["expert_skew"] = {"max_expert_over_mean": summarize("max/mean", share.max(1) * EXPERTS), "min_expert_over_mean": summarize("min/mean", share.min(1) * EXPERTS), "cv": summarize("cv", share.std(1) / share.mean(1)), "top8_share": summarize("top8", np.sort(share, 1)[:, -8:].sum(1))}
    # Tile-padded loads: a rank's work in 64/128-row M tiles of each local expert.
    for tile in (64, 128):
        report[f"tile{tile}_per_pass_max_over_mean"] = {"trivial": summarize("trivial", ratio(tile_loads(allc, trivial, tile))), "permuted": summarize("permuted", ratio(tile_loads(allc, placement, tile)))}
    gate_ms, serving_ms = 1.66, 2.17
    d = r_triv - r_new
    report["implied_saving_ms_per_layer"] = {"gate_scaled_mean": float(gate_ms * d.mean()), "serving_proportional_mean": float(serving_ms * (1 - r_new / r_triv).mean()), "per_forward_94_layers_gate_scaled_ms": float(gate_ms * d.sum())}
    report["per_layer"] = [{"layer": l, "trivial": round(float(r_triv[l]), 4), "permuted": round(float(r_new[l]), 4), "agg_trivial": round(float(agg_triv[l].max() / agg_triv[l].mean()), 4)} for l in range(layers)]
    if a.placement_out:
        a.placement_out.write_text(json.dumps({"physical_to_logical_map": placement.tolist()}, separators=(",", ":")) + "\n")
    text = json.dumps(report, indent=1)
    if a.report_out:
        a.report_out.write_text(text + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_layer"}, indent=1))


if __name__ == "__main__":
    main()
