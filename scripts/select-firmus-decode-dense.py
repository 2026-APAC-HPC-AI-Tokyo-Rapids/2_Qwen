"""Select Decode dense GEMMs inside the serving allocation, before model launch.

Run with torch.distributed.run --standalone --nproc-per-node=4. Rank 0 warms
JITs first; all four ranks validate, then measure synchronized round-robin
blocks. Only Python stdlib is imported for CPU selection-rule tests.
"""

import argparse
import json
import math
import os
import socket
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from runpy import run_path

BLOCKS = 2
REPLAYS = 25
REPS = 20
ENV_KEYS = ("DG_SM90_1D2D_DECODE_CONFIG", "SGLANG_DECODE_DENSE_GEMM_QKV", "SGLANG_DECODE_DENSE_GEMM_O")


def distribution(samples):
    assert samples and all(math.isfinite(x) and x > 0 for x in samples), samples
    values = sorted(samples)
    return {"median_us": statistics.median(values), "p10_us": values[len(values) // 10], "p90_us": values[(len(values) * 9) // 10]}


def choose_candidate(records):
    """Pool all four GPUs, require >=0.5us overall and a win in BOTH blocks."""
    assert len(records) == 4 and {r["rank"] for r in records} == set(range(4))
    names = [c["name"] for c in records[0]["candidates"]]
    assert names[0] == "stock" and len(names) == len(set(names))
    assert all([c["name"] for c in r["candidates"]] == names for r in records)
    summary = {}
    for name in names:
        entries = [r["results"][name] for r in records]
        if not all(e["numeric_passed"] for e in entries):
            summary[name] = {"eligible": False, "reason": "numeric-check-failed", "ranks": entries}
            continue
        samples = [[x for entry in entries for x in entry["blocks"][block]] for block in range(BLOCKS)]
        blocks = [distribution(values) for values in samples]
        pooled = distribution([x for values in samples for x in values])
        medians = [b["median_us"] for b in blocks]
        summary[name] = {**pooled, "blocks": blocks, "block_median_range_us": max(medians) - min(medians), "block_median_range_fraction": (max(medians) - min(medians)) / pooled["median_us"], "per_rank": [{"rank": r["rank"], "blocks": [distribution(values) for values in e["blocks"]]} for r, e in zip(records, entries)]}
    assert "median_us" in summary["stock"], "stock numerical check must pass on every GPU"
    stock = summary["stock"]
    eligible = []
    for name, result in summary.items():
        if "median_us" not in result:
            continue
        saving = stock["median_us"] - result["median_us"]
        wins = [b["median_us"] < ref["median_us"] for b, ref in zip(result["blocks"], stock["blocks"])]
        # Record rank-local regressions separately; eligibility follows the pooled rule.
        all_ranks_win = all(all(b["median_us"] < ref["median_us"] for b, ref in zip(rank["blocks"], stock_rank["blocks"])) for rank, stock_rank in zip(result["per_rank"], stock["per_rank"]))
        result.update(saving_us=saving, wins_by_block=wins, all_ranks_win=all_ranks_win, eligible=name != "stock" and saving >= 0.5 and all(wins))
        if result["eligible"]:
            eligible.append(name)
    winner = min(eligible, key=lambda name: (summary[name]["median_us"], name)) if eligible else "stock"
    return {"winner": winner, "summary": summary, "rule": {"minimum_saving_us": 0.5, "blocks": BLOCKS, "both_blocks_faster": True}}


def worker_environment(selections, candidates):
    """Empty values clear inherited settings; stock is represented by no override."""
    env = dict.fromkeys(ENV_KEYS, "")
    direct = []
    for op in ("qkv", "o"):
        name = selections[op]["winner"]
        candidate = next(c for c in candidates[op] if c["name"] == name)
        if name == "stock":
            continue
        if candidate["backend"] == "deepgemm":
            direct.append(op.upper() + ":" + ",".join(map(str, candidate["config"])))
        else:
            env["SGLANG_DECODE_DENSE_GEMM_" + op.upper()] = name
    env[ENV_KEYS[0]] = ";".join(direct)
    return env


def round_robin(names, block):
    """Alternate stock/candidate; reverse candidates in block two to vary order."""
    assert names[0] == "stock"
    others = names[1:] if block % 2 == 0 else list(reversed(names[1:]))
    return [item for name in others for item in ("stock", name)] if others else ["stock"]


@contextmanager
def candidate_mode(checks, op, candidate):
    previous = {key: os.environ.get(key) for key in ENV_KEYS}
    try:
        for key in ENV_KEYS[1:]:
            os.environ.pop(key, None)
        if candidate["backend"] != "deepgemm":
            os.environ["SGLANG_DECODE_DENSE_GEMM_" + op.upper()] = candidate["name"]
        with checks["dense_gemm_mode"](op, candidate):
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def capture_graph(torch, calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for call in calls:
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for rep in range(REPS):
            calls[rep % len(calls)]()
    return graph


def measure_graph(torch, graph):
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(REPLAYS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / REPS)
    return samples


def prepare_projection(torch, checks, op, n, k, candidates, cap_rows, boundary_rows=()):
    """Compile and screen outside measurement. All candidates share four weights."""
    started = time.monotonic()
    sets = [checks["dense_inputs"](cap_rows, n, k, 71200 + n + i) for i in range(4)]
    references = [checks["dense_reference"](s[0], s[1]) for s in sets]
    with candidate_mode(checks, op, candidates[0]):
        for s in sets:
            checks["dense_call"](candidates[0], s)()
    stock = [s[2].clone() for s in sets]
    boundaries = []
    for rows in boundary_rows:
        values = checks["dense_inputs"](rows, n, k, 71300 + n + rows)
        reference = checks["dense_reference"](values[0], values[1])
        with candidate_mode(checks, op, candidates[0]):
            checks["dense_call"](candidates[0], values)()
        boundaries.append((rows, values, reference, values[2].clone()))
    graphs, results = {}, {}
    for candidate in candidates:
        begin = time.monotonic()
        try:
            with candidate_mode(checks, op, candidate):
                metrics = checks["dense_validate_candidate"](op, candidate, sets[0], references[0], stock[0])
                boundary_metrics = [{"m": rows, **checks["dense_validate_candidate"](op, candidate, values, reference, baseline)} for rows, values, reference, baseline in boundaries]
                calls = [checks["dense_call"](candidate, s) for s in sets]
                for call, s, reference, baseline in zip(calls, sets, references, stock):
                    s[2].fill_(float("nan"))
                    call()
                    checks["dense_errors"](s[2], reference)
                    checks["dense_errors"](s[2], baseline.float())
                graphs[candidate["name"]] = capture_graph(torch, calls)
            results[candidate["name"]] = {"numeric_passed": True, "numeric": metrics, "boundary_numeric": boundary_metrics, "blocks": [[] for _ in range(BLOCKS)], "prepare_seconds": time.monotonic() - begin}
        except Exception as error:
            # Reject this candidate only; stock failures still abort the probe.
            if candidate["name"] == "stock":
                raise
            results[candidate["name"]] = {"numeric_passed": False, "error": str(error), "blocks": [[] for _ in range(BLOCKS)], "prepare_seconds": time.monotonic() - begin}
        print(json.dumps({"phase": "prepare", "rank": int(os.environ.get("LOCAL_RANK", "0")), "op": op, "candidate": candidate["name"], **results[candidate["name"]]}), flush=True)
    # Keep graph inputs alive across all round-robin blocks.
    return {"graphs": graphs, "sets": sets, "results": results, "prepare_seconds": time.monotonic() - started}


def run_selection(output, env_output):
    begin = time.monotonic()
    import torch
    import torch.distributed as dist
    from sglang.srt.layers.quantization.decode_dense_gemm import run_decode_dense_gemm

    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 4, "run on the four Decode GPUs in one torchrun"
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    assert torch.cuda.get_device_properties(rank).multi_processor_count == 132
    dist.init_process_group("gloo")
    checks = run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
    host_sha = checks["require_dense_host"]()
    original_call = checks["dense_call"]

    def serving_call(candidate, inputs):
        if candidate["backend"] == "deepgemm":
            return original_call(candidate, inputs)
        lhs, rhs, out = inputs
        op = "QKV" if rhs[0].shape == (9216, 4096) else "O"
        return lambda: run_decode_dense_gemm(op, candidate["name"], lhs, rhs, out)

    # run_path functions hold their own globals; inject into that original
    # namespace too so changed-input replay validates the actual overlay code.
    checks["dense_call"] = serving_call
    checks["dense_validate_candidate"].__globals__["dense_call"] = serving_call
    rows = checks["decode_cap_tokens"]()
    candidates = {op: checks["dense_candidates"](op) for op, _, _ in checks["DENSE_PROJECTIONS"]}
    prepared = {}
    with checks["dense_fp32_reference"]():
        # One cold compiler warms the shared container HOME caches first;
        # subsequent ranks import/load those kernels, avoiding four cold builds.
        if rank == 0:
            for op, n, k in checks["DENSE_PROJECTIONS"]:
                prepared[op] = prepare_projection(torch, checks, op, n, k, candidates[op], rows, boundary_rows=tuple(sorted({4, 64, 65, 128, 129, 160, 176, rows})))
        dist.barrier()
        if rank != 0:
            for op, n, k in checks["DENSE_PROJECTIONS"]:
                prepared[op] = prepare_projection(torch, checks, op, n, k, candidates[op], rows)
        dist.barrier()
    local = {"rank": rank, "device": torch.cuda.get_device_name(rank), "projections": {op: {"rank": rank, "candidates": choices, "results": prepared[op]["results"], "prepare_seconds": prepared[op]["prepare_seconds"]} for op, choices in candidates.items()}}
    all_records = [None] * 4
    dist.all_gather_object(all_records, local)
    for op, choices in candidates.items():
        active = [c["name"] for c in choices if all(record["projections"][op]["results"][c["name"]]["numeric_passed"] for record in all_records)]
        assert active and active[0] == "stock"
        for block in range(BLOCKS):
            for name in round_robin(active, block):
                dist.barrier()
                entry = prepared[op]["results"][name]
                if not entry["numeric_passed"]:
                    continue
                try:
                    samples = measure_graph(torch, prepared[op]["graphs"][name])
                    entry["blocks"][block].extend(samples)
                    print(json.dumps({"phase": "measure", "rank": rank, "op": op, "candidate": name, "block": block, **distribution(samples)}), flush=True)
                except Exception as error:
                    if name == "stock":
                        raise
                    entry.update(numeric_passed=False, error=f"timing {type(error).__name__}: {error}")
                    print(json.dumps({"phase": "candidate-rejected", "rank": rank, "op": op, "candidate": name, "error": entry["error"]}), flush=True)
    local["elapsed_seconds"] = time.monotonic() - begin
    dist.all_gather_object(all_records, local)
    if rank == 0:
        selections = {op: choose_candidate([r["projections"][op] for r in all_records]) for op in candidates}
        env = worker_environment(selections, candidates)
        result = {"schema": "firmus-decode-dense-selection/v1", "hostname": socket.gethostname(), "host_patch_sha256": host_sha, "shapes": {op: {"m": rows, "n": n, "k": k, "weight_bytes": 4 * n * k} for op, n, k in checks["DENSE_PROJECTIONS"]}, "blocks": BLOCKS, "replays": REPLAYS, "calls_per_replay": REPS, "weight_sets": 4, "timing": "CUDA graph replay; prequantized inputs; rotating weights", "elapsed_seconds": max(r["elapsed_seconds"] for r in all_records), "selections": selections, "worker_environment": env, "ranks": all_records}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        env_output.parent.mkdir(parents=True, exist_ok=True)
        env_output.write_text("".join(f"{key}={value}\n" for key, value in env.items()))
        print(json.dumps({"phase": "selected", "env": env, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-output", type=Path, required=True)
    args = parser.parse_args()
    run_selection(args.output, args.env_output)


if __name__ == "__main__":
    main()
