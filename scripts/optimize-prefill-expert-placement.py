"""Predict locality-aware EP4 placements from raw, aligned Prefill captures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

RANKS, EXPERTS, SLOTS, TILE = 4, 128, 35, 128
FC1_US_PER_ROW = 0.020555343690719444
FC2_US_PER_ROW = 0.010501314843053878
BYTES_PER_PAIR, NVLINK_BYTES_PER_SECOND = 8192, 450_000_000_000
ALPHAS = (0, 0.5, 1)


def validate_placement(mapping, redundant, layers=94):
    if redundant != 12:
        raise ValueError("The deployed checker requires exactly 12 redundant experts")
    if mapping.shape != (layers, 140) or mapping.dtype.kind not in "iu":
        raise ValueError(f"Expected {layers}x140 integer physical_to_logical_map")
    for row in mapping:
        if set(row.tolist()) != set(range(EXPERTS)):
            raise ValueError("Every logical expert 0..127 must be present")
        counts = np.bincount(row, minlength=EXPERTS)
        hot = set(np.flatnonzero(counts == 4).tolist())
        if len(hot) != 4 or np.any((counts != 1) & (counts != 4)):
            raise ValueError("Checker requires four experts x four copies and 124 single copies")
        for local in row.reshape(RANKS, SLOTS):
            if len(set(local.tolist())) != SLOTS or not hot <= set(local.tolist()):
                raise ValueError("Checker requires unique local experts and every hot expert on every rank")


def owners(row):
    """State 4 means one local replica per source; 0..3 means a sole owner."""
    result = np.empty(EXPERTS, dtype=np.int64)
    for expert in range(EXPERTS):
        slots = np.flatnonzero(row == expert)
        result[expert] = 4 if len(slots) == 4 else slots[0] // SLOTS
    return result


def load_capture(directory, mapping):
    if not directory.is_dir():
        raise ValueError("Routing input must be a raw capture directory; summary JSON is unsupported")
    layers = len(mapping)
    expected_workers = {f"prefill-{w}" for w in range(3)}
    captures, provenance = {}, []
    states = [owners(row) for row in mapping]
    for path in sorted(directory.glob("*/*.npz")):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            ids = data["ids"]
        worker, rank, sample, step = (meta[k] for k in ("worker", "source_ep_rank", "sample", "target_forward_step"))
        if worker not in expected_workers or path.parent.name != worker:
            raise ValueError(f"Unexpected worker: {path}")
        if type(rank) is not int or rank not in range(4) or type(sample) is not int or sample not in range(8) or type(step) is not int:
            raise ValueError(f"Invalid rank/sample/forward ID: {path}")
        if path.name != f"rank-{rank}-sample-{sample}.npz":
            raise ValueError(f"Capture filename differs from metadata: {path}")
        geometry = {"schema": "prefill-routing/v1", "phase": "diagnostic", "physical_experts": 140, "experts_per_destination_rank": 35, "hidden_size": 4096, "topk": 8, "layers": list(range(layers))}
        if any(meta.get(k) != v for k, v in geometry.items()):
            raise ValueError(f"Unsupported capture geometry: {path}")
        live = meta["live_rows"]
        if type(live) is not int or not 0 <= live <= 4096 or ids.shape != (layers, live, 8) or ids.dtype.kind != "i":
            raise ValueError(f"Invalid top-k tensor: {path}")
        if np.any((ids < 0) | (ids >= 140)):
            raise ValueError(f"Invalid physical ID: {path}")
        counts = np.zeros((layers, EXPERTS), dtype=np.int64)
        for layer in range(layers):
            logical = mapping[layer, ids[layer]]
            if np.any(np.diff(np.sort(logical, axis=1), axis=1) == 0):
                raise ValueError(f"Duplicate logical expert in top-k: {path}, layer {layer}")
            destinations = np.where(states[layer][logical] == 4, rank, states[layer][logical])
            if np.any(destinations != ids[layer] // SLOTS):
                raise ValueError(f"Capture routing disagrees with current placement static dispatch: {path}, layer {layer}")
            counts[layer] = np.bincount(logical.ravel(), minlength=EXPERTS)
        key = (worker, rank, step)
        if key in captures:
            raise ValueError(f"Duplicate target forward: {key}")
        captures[key] = (sample, counts)
        provenance.append({"path": str(path.relative_to(directory)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    halves, split = [[], []], []
    for worker in sorted(expected_workers):
        steps = [sorted(s for w, r, s in captures if w == worker and r == rank) for rank in range(4)]
        if not 2 <= len(steps[0]) <= 8 or any(s != steps[0] for s in steps[1:]):
            raise ValueError(f"Need 2..8 aligned target forwards on all four ranks: {worker}")
        for rank in range(4):
            samples = sorted(captures[worker, rank, s][0] for s in steps[0])
            if samples != list(range(len(steps[0]))):
                raise ValueError(f"Incomplete sample sequence: {worker}, rank {rank}")
            if not any(captures[worker, rank, s][1].sum() for s in steps[0]):
                raise ValueError(f"Source rank has no nonempty data: {worker}, rank {rank}")
        half = len(steps[0]) // 2
        split.append({"worker": worker, "training": steps[0][:half], "held_out": steps[0][half:]})
        for i, step in enumerate(steps[0]):
            halves[int(i >= half)].append(np.stack([captures[worker, rank, step][1] for rank in range(4)], axis=1))
    return np.stack(halves[0]), np.stack(halves[1]), split, provenance


def contributions(counts):
    """[expert, owner state, forward, rank, metric: useful/padded/sent/received]."""
    result = np.zeros((EXPERTS, 5, len(counts), RANKS, 4), dtype=np.int64)
    for expert in range(EXPERTS):
        source = counts[:, :, expert]
        total = source.sum(axis=1)
        for rank in range(RANKS):
            result[expert, rank, :, rank, 0] = total
            result[expert, rank, :, rank, 1] = ((total + TILE - 1) // TILE) * TILE
            result[expert, rank, :, :, 3] = source * BYTES_PER_PAIR
            result[expert, rank, :, rank, 3] = 0
            result[expert, rank, :, rank, 2] = (total - source[:, rank]) * BYTES_PER_PAIR
        result[expert, 4, :, :, 0] = source
        result[expert, 4, :, :, 1] = ((source + TILE - 1) // TILE) * TILE
    return result


def times(metrics, alpha):
    rank_us = metrics[..., 1] * (FC1_US_PER_ROW + FC2_US_PER_ROW)
    rank_us = rank_us + alpha * np.maximum(metrics[..., 2], metrics[..., 3]) * 1e6 / NVLINK_BYTES_PER_SECOND
    return rank_us.max(axis=-1)


def optimize_layer(counts, row):
    costs = contributions(counts)
    state = owners(row)
    metrics = costs[np.arange(EXPERTS), state].sum(axis=0)
    score = times(metrics, 0.5).mean()
    moves = 0
    while True:
        # Cold/cold swaps and hot/cold replica-set exchanges both swap states.
        a, b = np.triu_indices(EXPERTS, 1)
        valid = state[a] != state[b]
        a, b = a[valid], b[valid]
        delta = costs[a, state[b]] + costs[b, state[a]] - costs[a, state[a]] - costs[b, state[b]]
        scores = times(metrics[None] + delta, 0.5).mean(axis=1)
        best = int(np.argmin(scores))
        if scores[best] >= score - 1e-10:
            break
        x, y = int(a[best]), int(b[best])
        if state[x] != 4 and state[y] != 4:
            px, py = int(np.flatnonzero(row == x)[0]), int(np.flatnonzero(row == y)[0])
            row[px], row[py] = y, x
        else:
            hot, cold = (x, y) if state[x] == 4 else (y, x)
            owner = state[cold]
            for slot in np.flatnonzero(row == hot):
                if slot // SLOTS != owner:
                    row[slot] = cold
        state[x], state[y] = state[y], state[x]
        metrics += delta[best]
        score = float(scores[best])
        moves += 1
    return row, moves


def evaluate(counts, mapping):
    metrics = np.stack([contributions(counts[:, layer])[np.arange(EXPERTS), owners(row)].sum(axis=0) for layer, row in enumerate(mapping)], axis=1)
    return {
        "layers": [{"layer": layer, "mean_useful_rows_by_rank": m[..., 0].mean(axis=0).tolist(), "mean_padded_rows_by_rank": m[..., 1].mean(axis=0).tolist(), "mean_fc2_sent_bytes_by_rank": m[..., 2].mean(axis=0).tolist(), "mean_fc2_received_bytes_by_rank": m[..., 3].mean(axis=0).tolist(), "predicted_time_us": {str(a): float(times(m, a).mean()) for a in ALPHAS}} for layer, m in enumerate(metrics.transpose(1, 0, 2, 3))],
        "predicted_iteration_us": {str(a): float(times(metrics, a).sum(axis=1).mean()) for a in ALPHAS},
    }


def plan(training, held_out, mapping):
    candidate = mapping.copy()
    moves = []
    for layer in range(len(mapping)):
        candidate[layer], count = optimize_layer(training[:, layer], candidate[layer])
        moves.append(count)
    report = {"schema": "prefill-placement-prediction/v1", "accepted": False, "moves_by_layer": moves}
    for name, counts in (("training", training), ("held_out", held_out)):
        before, after = evaluate(counts, mapping), evaluate(counts, candidate)
        gains = {str(a): before["predicted_iteration_us"][str(a)] - after["predicted_iteration_us"][str(a)] for a in ALPHAS}
        report[name] = {"forwards": len(counts), "current": before, "candidate": after, "predicted_saving_us_per_iteration": gains}
    train = report["training"]["predicted_saving_us_per_iteration"]["0.5"]
    held = report["held_out"]["predicted_saving_us_per_iteration"]["0.5"]
    reasons = []
    if train < 2000:
        reasons.append(f"alpha=0.5 predicted training saving {train:.3f} us is below 2000 us per iteration")
    if held < train / 2:
        reasons.append(f"alpha=0.5 held-out saving {held:.3f} us is below half of training saving {train:.3f} us")
    report["refusal_reasons"] = reasons
    report["accepted"] = not reasons
    return candidate, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--redundant-experts", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = [p.resolve() for p in (args.placement, args.output, args.report)]
        if len(set(paths)) != 3 or args.output.exists():
            raise ValueError("Use distinct placement/output/report paths and a new output path")
        placement_bytes = args.placement.read_bytes()
        mapping = np.asarray(json.loads(placement_bytes)["physical_to_logical_map"])
        validate_placement(mapping, args.redundant_experts)
        training, held_out, split, provenance = load_capture(args.directory, mapping)
        candidate, report = plan(training, held_out, mapping)
        validate_placement(candidate, args.redundant_experts)
        report.update({
            "placement_sha256": hashlib.sha256(placement_bytes).hexdigest(), "captures": provenance, "aligned_forward_split": split,
            "model": {"block_m": TILE, "fc1_us_per_row": FC1_US_PER_ROW, "fc2_us_per_row": FC2_US_PER_ROW, "fc2_bytes_per_pair": BYTES_PER_PAIR, "nvlink_one_direction_bytes_per_second": NVLINK_BYTES_PER_SECOND, "alphas": ALPHAS, "selection_alpha": 0.5, "minimum_training_saving_us": 2000, "minimum_held_out_fraction": 0.5,
                      "sources": ["research/runtime-performance/megamoe-phases-40057117293.json: WG0 fc1_loop/fc2_loop CTA medians divided by same-call recv_pairs", "research/runtime-performance/kernel-theory-model.json: model and hardware.nvlink_one_direction_bytes_per_second", "patches/deep-gemm-host/sm90-prefill-2wg.patch: BLOCK_M=128"],
                      "interpretation": "Conditional model predictions, not serving measurements. Useful-row loop coefficients are extrapolated to M128 padded rows; effective link bandwidth and non-overlap alpha are unmeasured. Per forward sum layers, then average forwards across workers; worker times are not summed. FC2 uses max(send, receive) for full duplex. Hot replicas select the source rank local slot; cold experts select their sole global slot."},
        })
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        if not report["accepted"]:
            parser.exit(1, "Refused placement: " + "; ".join(report["refusal_reasons"]) + "\n")
        args.output.write_text(json.dumps({"physical_to_logical_map": candidate.tolist()}, separators=(",", ":")) + "\n")
        print(f"Wrote model-predicted placement {args.output}; report {args.report}")
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Invalid placement/capture: {exc}\n")


if __name__ == "__main__":
    main()
