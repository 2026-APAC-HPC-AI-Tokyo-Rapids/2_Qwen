#!/usr/bin/env python3
"""Extract saved MegaMoE phases and work accounting without running a GPU job.

Usage: uv run --with numpy python research/runtime-performance/extract-megamoe-phases.py
       VARIANT_RESULTS --run RUN --image IMAGE --case CASE > OUTPUT.json
The results directory holds bench.jsonl, timeline.jsonl, and timeline/CASE.npy.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARSER = ROOT / "scripts/summarize-megamoe-kernel-bench.py"
spec = importlib.util.spec_from_file_location("megamoe_timeline", PARSER)
timeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timeline)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(path, case):
    return next(r for r in map(json.loads, path.read_text().splitlines()) if r["case"] == case)


def extract(directory, case, run, image):
    npy = directory / "timeline" / (case + ".npy")
    bench = row(directory / "bench.jsonl", case)
    instrumented = row(directory / "timeline.jsonl", case)
    assert bench["output_sha256"] == instrumented["output_sha256"]
    assert instrumented["bitwise_equal_to_reference"]
    data = np.load(npy)
    calls = [[timeline.parse(data[c, r]) for r in range(data.shape[1])] for c in range(data.shape[0])]
    kernel = np.array([max(r["end"] for r in c) for c in calls[1:]])
    selected = 1 + int(np.argsort(kernel)[len(kernel) // 2])
    chosen = calls[selected]
    assert all(r["header"]["version"] == 2 for r in chosen), "This extractor models v2 only"
    ranks = []
    counts = bench.get("source_rank_expert_counts")
    transfer = None
    padded_rows = None
    if counts is not None:
        experts = bench["physical_experts"]
        assert len(counts) == 4 and all(len(r) == experts for r in counts)
        assert all(sum(r) == bench["tokens_per_rank"] * 8 for r in counts)
        loads = [sum(r[e] for r in counts) for e in range(experts)]
        epr = experts // 4
        destinations = [[sum(r[d*epr:(d+1)*epr]) for d in range(4)] for r in counts]
        assert [sum(destinations[s][d] for s in range(4)) for d in range(4)] == bench["recv_pairs_per_rank"]
        # FC2 executes at destination expert rank and returns BF16 partials to source.
        remote = [sum(destinations[s][d] for s in range(4) if s != d) * 4096 * 2 for d in range(4)]
        receive = [sum(destinations[s][d] for d in range(4) if s != d) * 4096 * 2 for s in range(4)]
        assert sum(remote) == sum(receive)
        padded_rows = [sum(((v + 63)//64)*64 for v in loads[d*epr:(d+1)*epr]) for d in range(4)]
        transfer = {"source_to_expert_rank_pairs": destinations, "fc2_remote_payload_bytes_per_producer_rank": remote,
                    "fc2_remote_receive_bytes_per_source_rank": receive,
                    "fc2_remote_payload_at_450GBs_us": [b / 450e9 * 1e6 for b in remote],
                    "fc2_send_receive_peak_reference_us_per_rank": [max(tx, rx) / 450e9 * 1e6 for tx, rx in zip(remote, receive)],
                    "scope": "logical BF16 payload; excludes protocol/transactions and is not measured link utilization"}
    for i, rank in enumerate(chosen):
        ghz = float(np.nanmedian(rank["ghz"]))
        pairs = bench["recv_pairs_per_rank"][i]
        work = {"recv_pairs": pairs, "fc1_useful_flops": pairs*4*4096*1536,
                "fc2_useful_flops": pairs*2*4096*1536, "peak_fp8_flops_per_second": 8192*132*ghz*1e9,
                "fc2_partial_bytes": pairs*4096*2,
                "combine_logical_read_write_bytes": bench["tokens_per_rank"]*4096*2*9,
                "combine_fp32_fma_flops": bench["tokens_per_rank"]*4096*8*2,
                "padded_rows_64": None if padded_rows is None else padded_rows[i]}
        work["combine_logical_hbm_peak_us"] = work["combine_logical_read_write_bytes"]/4.8e12*1e6
        for fc in ("fc1", "fc2"):
            work[fc + "_compute_only_us"] = work[fc + "_useful_flops"]/work["peak_fp8_flops_per_second"]*1e6
            if padded_rows is not None:
                work[fc + "_padded_flops_64"] = work[fc + "_useful_flops"]*padded_rows[i]//pairs
        wgs = []
        for wg in range(rank["header"]["math_warpgroups"]):
            def event(name):
                return rank["times"][:, timeline.V2_EVENTS.index(f"WG{wg}{name}")]

            def counter(name):
                return rank["counters"][:, timeline.V2_COUNTERS.index(f"WG{wg}{name}")]

            phases = {"first_stage_wait": event("FirstFC1")-event("Start"),
                      "fc1_loop": counter("FC1Loop"), "fc1_stage_wait_subset": counter("FC1Wait"),
                      "fc1_epilogue": counter("FC1Epilogue"), "fc2_loop": counter("FC2Loop"),
                      "fc2_stage_wait_subset": counter("FC2Wait"), "fc2_epilogue_scatter": counter("FC2Epilogue"),
                      "scheduler_tail": event("LoopExit")-event("LastFC2"),
                      "pre_combine_grid_nvlink_wait": event("BarrierEnd")-event("LoopExit"),
                      "combine_cleanup": event("CombineEnd")-event("BarrierEnd")}
            wgs.append({"wg": wg, "phases_us": {name: {"cta_median": float(np.nanmedian(v)), "cta_max": float(np.nanmax(v))} for name, v in phases.items()},
                        "blocks_cta_median": {name: float(np.nanmedian(counter(name))) for name in ("FC1Blocks", "FC2Blocks")},
                        "useful_to_peak_over_median_cta_loop": {fc: work[fc + "_compute_only_us"]/float(np.nanmedian(phases[fc + "_loop"])) for fc in ("fc1", "fc2")}})
        ranks.append({"rank": i, "sm_clock_ghz_cta_median": ghz, "work": work, "warpgroups": wgs})
    return {"schema_version": "megamoe-phases/1", "run": run, "image_run": image, "case": case,
            "source_files": {str(p.relative_to(directory)): digest(p) for p in (npy, directory/"bench.jsonl", directory/"timeline.jsonl")},
            "parser_sha256": digest(PARSER), "extractor_sha256": digest(Path(__file__)),
            "geometry": chosen[0]["header"], "bench_median_us": bench["median_ms"]*1000,
            "instrumented_bench_median_us": instrumented["median_ms"]*1000,
            "timeline_calls_2_end_median_us": float(np.median(kernel)), "representative_call_zero_based": selected,
            "transfer": transfer, "remote_fraction_all_ranks": 1-bench["local_slot_fraction"], "ranks": ranks,
            "limitations": ["Uninstrumented bench times pre_dispatch plus MegaMoE; the raw timeline describes the fused MegaMoE only.",
                            "CTA medians are not additive critical-path intervals or device utilization counters.",
                            "Stage waits are subsets of mainloop. FC1 deferred publication remains in later loop/scheduler work.",
                            "FC2 epilogue includes BF16 packing, address issue, stores and barriers; standalone transfer completion is unknown.",
                            "Grid/NVLink pre-combine waiting includes other ranks finishing computation, not only wire transfer.",
                            "Useful/peak over median CTA loop is a diagnostic proxy including padding, setup, stage waits and FP32 promotion.",
                            "64-row padded work applies to the current adopted full-padding-WG skip only; older runs leave padded_rows_64 null.",
                            "Logical bytes/peak bandwidth are resource references, not measured HBM/NVLink bandwidth.",
                            "Real-layer cases are sampled fixtures from archived pooled expert totals, not production source-conditioned token routing. Their matrix cannot justify a production placement change."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--case", default="prefill-real43-replica4-4096")
    p.add_argument("--run", required=True)
    p.add_argument("--image", required=True)
    a = p.parse_args()
    print(json.dumps(extract(a.directory, a.case, a.run, a.image), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
