"""Validate complete EP4 Prefill captures and count actual FC2 output traffic."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(directory, placement, expected_workers=3, expected_ranks=4, samples_per_rank=8):
    import numpy as np

    if expected_ranks != 4 or not 1 <= samples_per_rank <= 8:
        raise ValueError("Expected EP4 and 1..8 samples per rank")
    placement_bytes = placement.read_bytes()
    mapping = np.asarray(json.loads(placement_bytes)["physical_to_logical_map"])
    if mapping.shape != (94, 140) or mapping.dtype.kind not in "iu":
        raise ValueError("Expected the deployed 94-layer physical-140 placement")
    if any(set(layer.tolist()) != set(range(128)) for layer in mapping):
        raise ValueError("Every logical expert must be represented in every layer")
    expected = {f"prefill-{w}" for w in range(expected_workers)}
    captures = {}
    for path in sorted(directory.glob("*/*.npz")):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
        worker, rank, sample = meta["worker"], meta["source_ep_rank"], meta["sample"]
        if worker not in expected or path.parent.name != worker:
            raise ValueError(f"Unexpected worker: {path}")
        if type(rank) is not int or rank not in range(4) or type(sample) is not int or sample not in range(samples_per_rank):
            raise ValueError(f"Invalid rank/sample: {path}")
        key = (worker, rank, sample)
        if key in captures:
            raise ValueError(f"Duplicate rank/sample: {key}")
        if meta["schema"] != "prefill-routing/v1" or meta["phase"] != "diagnostic" or meta["physical_experts"] != 140 or meta["experts_per_destination_rank"] != 35 or meta["hidden_size"] != 4096 or meta["topk"] != 8 or meta["layers"] != list(range(94)):
            raise ValueError(f"Unsupported capture geometry: {path}")
        captures[key] = (path, meta)
    required = {(w, r, s) for w in expected for r in range(4) for s in range(samples_per_rank)}
    if set(captures) != required:
        raise ValueError(f"Incomplete capture: {len(captures)}/{len(required)} files")
    # Sampling indices are not global step IDs; require the same actual forwards.
    for worker in expected:
        steps = [{captures[worker, rank, s][1]["target_forward_step"] for s in range(samples_per_rank)} for rank in range(4)]
        if len(steps[0]) != samples_per_rank or any(x != steps[0] for x in steps[1:]):
            raise ValueError(f"Source ranks captured different target forwards: {worker}")

    workers = []
    for worker in sorted(expected):
        matrix = np.zeros((94, 4, 4), dtype=np.int64)
        logical_counts = np.zeros((94, 4, 128), dtype=np.int64)
        token_counts = np.zeros(4, dtype=np.int64)
        remote_groups = np.zeros((94, 4), dtype=np.int64)
        for rank in range(4):
            for sample in range(samples_per_rank):
                path, meta = captures[worker, rank, sample]
                with np.load(path, allow_pickle=False) as data:
                    ids = data["ids"]
                if ids.dtype.kind != "i" or ids.shape != (94, meta["live_rows"], 8) or not 0 <= ids.shape[1] <= 4096:
                    raise ValueError(f"Invalid top-k tensor: {path}")
                if np.any((ids < 0) | (ids >= 140)):
                    raise ValueError(f"Invalid physical ID in a live row: {path}")
                token_counts[rank] += ids.shape[1]
                for layer in range(94):
                    logical = mapping[layer, ids[layer]]
                    if np.any(np.diff(np.sort(logical, axis=1), axis=1) == 0):
                        raise ValueError(f"Duplicate logical expert in top-k: {path} layer {layer}")
                    destinations = ids[layer] // 35
                    matrix[layer, rank] += np.bincount(destinations.ravel(), minlength=4)
                    logical_counts[layer, rank] += np.bincount(logical.ravel(), minlength=128)
                    for destination in range(4):
                        if destination != rank:
                            remote_groups[layer, rank] += np.any(destinations == destination, axis=1).sum()
        if np.any(token_counts == 0):
            raise ValueError(f"Every source rank needs nonempty data: {worker}")
        layers = []
        for layer in range(94):
            traffic = matrix[layer]
            local = np.diag(traffic)
            remote_by_source = traffic.sum(axis=1) - local
            remote_by_expert_rank = traffic.sum(axis=0) - local
            total = int(traffic.sum())
            remote = int(remote_by_source.sum())
            # An optimistic bound, not a deployable placement: cold experts may
            # each choose their best origin rank without a slot/load constraint.
            counts = logical_counts[layer]
            occurrences = np.bincount(mapping[layer], minlength=128)
            cold = np.where(occurrences == 1)[0]
            cold_owners = np.array([int(np.where(mapping[layer] == e)[0][0]) // 35 for e in cold])
            current_cold_local = sum(int(counts[owner, e]) for e, owner in zip(cold, cold_owners, strict=True))
            cold_local_upper = int(counts[:, cold].max(axis=0).sum())
            layers.append({
                "layer": layer, "token_count_by_source_rank": token_counts.tolist(),
                "slots_source_to_expert_rank": traffic.tolist(),
                "logical_counts_by_source_rank": counts.tolist(),
                "total_slots": total, "remote_slots": remote,
                "remote_slot_fraction": remote / total,
                "fc2_remote_output_bytes_by_expert_rank": (remote_by_expert_rank * 8192).tolist(),
                "fc2_remote_received_bytes_by_source_rank": (remote_by_source * 8192).tolist(),
                "combine_read_payload_bytes_by_source_rank": (traffic.sum(axis=1) * 8192).tolist(),
                "remote_token_destination_pairs": int(remote_groups[layer].sum()),
                "cold_relocation_saved_remote_slots_upper_bound": cold_local_upper - current_cold_local,
            })
        workers.append({"worker": worker, "layers": layers})
    return {
        "schema": "prefill-routing-summary/v1", "valid": True,
        "expected_workers": expected_workers, "expected_ranks": expected_ranks,
        "samples_per_rank": samples_per_rank, "placement_sha256": hashlib.sha256(placement_bytes).hexdigest(),
        "bytes_per_fc2_expert_output": 8192,
        "interpretation": "Measured physical routing; source-to-expert matrix reverses for FC2 scatter. Payload bytes exclude link/protocol overhead. Cold relocation bound ignores capacity, padding and load balance, and is not a speedup prediction. Token/destination deduplication is only a hypothetical output-row bound; current combine reads every expert output.",
        "workers": workers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-workers", type=int, default=3)
    parser.add_argument("--expected-ranks", type=int, default=4)
    parser.add_argument("--samples-per-rank", type=int, default=8)
    args = parser.parse_args()
    result = summarize(args.directory, args.placement, args.expected_workers, args.expected_ranks, args.samples_per_rank)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(f"Validated {args.expected_workers} workers x {args.expected_ranks} ranks x {args.samples_per_rank} forwards; output={args.output}")


if __name__ == "__main__":
    main()
