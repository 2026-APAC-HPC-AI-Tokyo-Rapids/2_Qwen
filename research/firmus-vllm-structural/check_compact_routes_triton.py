"""Compare actual metadata kernel bodies with the independent NumPy route plan.

CPU mode uses matching-version Triton interpreter, not CUDA compilation.
CUDA mode also replays changed inputs through a preallocated CUDA graph.
"""

import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.device == "cpu":
    os.environ["TRITON_INTERPRET"] = "1"
elif os.environ.get("TRITON_INTERPRET") == "1":
    raise RuntimeError("CUDA validation must not use the interpreter")

import numpy as np
import torch
import triton
from compact_routes_reference import plan_routes
from compact_routes_triton import RouteWorkspace, launch


def verify(ws, ids, valid):
    array, mask = ids.cpu().numpy(), valid.cpu().numpy()
    plan = plan_routes(array, mask, ws.rank)
    count = plan.slots.size
    offsets = ws.offsets.cpu().numpy()
    routes = ws.routes.cpu().numpy()
    assert offsets[-1] == count
    assert np.array_equal(routes[:count], plan.slots)
    assert np.all(routes[count:] == -1)
    assert np.array_equal(ws.inverse.cpu().numpy().reshape(array.shape), plan.reverse)
    expected = np.bincount(array.reshape(-1)[plan.slots] - ws.rank * 32, minlength=32)
    assert np.array_equal(ws.counts.cpu().numpy(), expected)
    assert np.array_equal(offsets, np.r_[0, np.cumsum(expected)])


def patterns(rng, tokens):
    ids = np.stack([rng.choice(128, 8, replace=False) for _ in range(tokens)]).astype(
        np.int64
    )
    yield "uniform", ids, np.ones(tokens, dtype=bool)
    yield "hot_rank", np.tile(np.arange(8), (tokens, 1)), np.ones(tokens, dtype=bool)
    yield (
        "boundary",
        np.tile([31, 32, 63, 64, 95, 96, 0, 127], (tokens, 1)),
        np.ones(tokens, dtype=bool),
    )
    yield "masked", ids, np.arange(tokens) % 3 != 0
    yield "empty", ids, np.zeros(tokens, dtype=bool)
    yield "changed", (ids + 32) % 128, np.ones(tokens, dtype=bool)


def main():
    rng = np.random.default_rng(20261014)
    comparisons, graph_replays, cases = 0, 0, []
    for tokens in [1, 7, 32, 97, 288, 384, 768]:
        for rank in range(4):
            ws = RouteWorkspace.allocate(tokens, rank, torch.device(args.device))
            ids = torch.empty((tokens, 8), dtype=torch.int64, device=args.device)
            valid = torch.empty(tokens, dtype=torch.bool, device=args.device)
            pointers = [
                v.data_ptr() for v in [ws.counts, ws.offsets, ws.inverse, ws.routes]
            ]
            for mode, array, mask in patterns(rng, tokens):
                ids.copy_(torch.from_numpy(array))
                valid.copy_(torch.from_numpy(mask))
                launch(ws, ids, valid)
                verify(ws, ids, valid)
                assert pointers == [
                    v.data_ptr() for v in [ws.counts, ws.offsets, ws.inverse, ws.routes]
                ]
                comparisons += 1
                cases.append({"tokens": tokens, "rank": rank, "mode": mode})
            if args.device == "cuda":
                # Warmup and synchronization are harness-only, outside serving.
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch(ws, ids, valid)
                for _, array, mask in patterns(rng, tokens):
                    ids.copy_(torch.from_numpy(array))
                    valid.copy_(torch.from_numpy(mask))
                    graph.replay()
                    verify(ws, ids, valid)
                    graph_replays += 1
    result = {
        "schema": "compact-route-device-metadata/v1",
        "device": args.device,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "comparisons": comparisons,
        "changed_cuda_graph_replays": graph_replays,
        "cases": cases,
        "limitations": "Metadata only. CPU interpreter does not certify CUDA compilation, memory masking, graph capture, native model FP8/BF16 numerics, GEMM integration, latency, official accuracy or speedup. Buffers retain worst-case full M*8 capacity. Three extra metadata launches may offset savings; no serving integration or extra GPU run.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["cases", "limitations"]}
        )
    )


if __name__ == "__main__":
    main()
