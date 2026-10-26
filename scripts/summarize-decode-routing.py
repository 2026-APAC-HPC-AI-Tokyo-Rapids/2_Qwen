#!/usr/bin/env python3
"""Combine sampled source-rank top-k histograms into destination expert rows."""
import argparse
import json
from pathlib import Path


def summarize(directory):
    samples = {}
    for path in sorted(directory.glob("rank-*-pid-*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["schema"] != "decode-routing/v1" or row["physical_experts"] != 128:
                raise ValueError(f"Unsupported routing schema in {path}")
            step, rank = row["target_forward_step"], row["source_ep_rank"]
            group = samples.setdefault(step, {})
            if rank in group:
                raise ValueError(f"Duplicate source rank {rank} at window step {step}")
            group[rank] = row
    complete, incomplete = [], []
    for step, group in sorted(samples.items()):
        if set(group) != set(range(4)):
            incomplete.append({"target_forward_step": step, "source_ranks": sorted(group)})
            continue
        layers = set(group[0]["counts_by_layer"])
        if any(set(row["counts_by_layer"]) != layers for row in group.values()):
            raise ValueError(f"Layer mismatch at window step {step}")
        received = {}
        for layer in sorted(layers, key=int):
            vectors = [group[rank]["counts_by_layer"][layer] for rank in range(4)]
            for rank, counts in enumerate(vectors):
                if len(counts) != 128 or any(count < 0 for count in counts):
                    raise ValueError("Expected 128 nonnegative physical-expert counts")
                if sum(counts) != group[rank]["valid_topk_slots_by_layer"][layer]:
                    raise ValueError("Histogram sum differs from recorded valid top-k slots")
            counts = [sum(vector[expert] for vector in vectors) for expert in range(128)]
            received[layer] = {
                "rows_by_physical_expert": counts,
                "rows_by_destination_rank": [sum(counts[start:start + 32]) for start in range(0, 128, 32)],
                "experts_over_64_rows": [expert for expert, count in enumerate(counts) if count > 64],
                "m64_blocks_by_destination_rank": [sum((count + 63) // 64 for count in counts[start:start + 32]) for start in range(0, 128, 32)],
            }
        complete.append({"target_forward_step": step, "layers": received})
    return {"schema": "decode-routing-received/v1", "samples": complete, "incomplete_samples": incomplete}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(f"complete={len(result['samples'])} incomplete={len(result['incomplete_samples'])}")
    if not result["samples"]:
        raise SystemExit("No complete EP4 samples")


if __name__ == "__main__":
    main()
