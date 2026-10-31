"""Default-off PP4 QKV gate; CPU aggregation imports only the standard library.

Three four-GPU workers run concurrently before model load. The unmodified
probe supplies identical operands, FP64 reference, five-output checks and calls.
Only all-twelve success may replace the pinned Prefill fp32 environment value.
"""

import argparse
import contextlib
import io
import json
import math
import os
import runpy
import statistics
import time
from pathlib import Path

KEY = "SGLANG_PREFILL_QKNORM"
SCHEMA = "firmus-prefill-qkv-selection/v1"
ROWS = (3994, 7988)
OUTPUTS = ("q", "k", "v", "k_cache", "v_cache")
ROUNDS, REPS, ENDURANCE = 12, 12, 300
MARGIN = 0.01


def reject_nonfinite(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def timing_summary(samples):
    medians = {}
    for order in ("forward", "reverse"):
        medians[order] = {}
        for mode in ("stock", "fused", "gemm", "post"):
            values = samples[order][mode]
            assert len(values) == ROUNDS, "incomplete timing"
            assert all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values), "invalid timing"
            medians[order][mode] = statistics.median(values)
    pooled = {mode: statistics.median(samples["forward"][mode] + samples["reverse"][mode]) for mode in medians["forward"]}
    noise = {mode: {"forward_median_us": medians["forward"][mode], "reverse_median_us": medians["reverse"][mode], "absolute_order_delta_us": abs(medians["forward"][mode] - medians["reverse"][mode]), "order_delta_fraction": abs(medians["forward"][mode] - medians["reverse"][mode]) / pooled[mode]} for mode in pooled}
    ratios = {order: values["fused"] / values["stock"] for order, values in medians.items()}
    ratios["pooled"] = pooled["fused"] / pooled["stock"]
    return {"median_us": pooled, "order_median_us": medians, "noise": noise, "fused_over_stock": ratios}


def numeric_pass(check):
    assert check["passed"] is True and set(check["outputs"]) == set(OUTPUTS), "missing or failed output check"
    for name, metrics in check["outputs"].items():
        assert metrics["passed"] is True, f"{name}: failed"
        for suffix in ("max_abs", "mean_abs", "fp64_code_mismatch"):
            fused, stock = metrics[f"fused_{suffix}"], metrics[f"fp32_{suffix}"]
            assert all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in (fused, stock)), "invalid numeric metric"
            assert fused <= stock, f"{name}: worse {suffix}"


def aggregate(directory, env_output, run_id, deadline, workers_ok):
    result = {"schema": SCHEMA, "adopted": False, "env": {KEY: "fp32"}, "rule": {"rows": ROWS, "outputs": OUTPUTS, "endurance_calls_per_shape": ENDURANCE, "timed_rows": 7988, "margin_fraction_of_stock_gemm_plus_post": MARGIN, "rounds_per_order": ROUNDS, "calls_per_sample": REPS, "require_both_orders": True}, "gpus": []}
    try:
        assert workers_ok, "selector process failed or timed out"
        assert time.time() <= deadline, "selection deadline expired"
        uuids = set()
        for worker in range(3):
            for rank in range(4):
                record = json.loads((directory / f"prefill-{worker}" / f"rank-{rank}.json").read_text(), parse_constant=reject_nonfinite)
                result["gpus"].append(record)
                assert record["schema"] == SCHEMA and record["run_id"] == run_id, "stale result"
                assert record["worker"] == f"prefill-{worker}" and record["rank"] == rank, "wrong GPU result"
                assert "H200" in record["device"] and record["uuid"] and record["uuid"] not in uuids, "wrong or repeated GPU"
                uuids.add(record["uuid"])
                assert record["finished_at"] <= deadline and record["elapsed_seconds"] < 60, "late result"
                assert record["passed"] is True and set(record["rows"]) == {str(m) for m in ROWS}, "failed or incomplete result"
                assert record["norm_weights"]["source"] == "staged-model", "missing model norm weights"
                for m in ROWS:
                    row = record["rows"][str(m)]
                    numeric_pass(row["initial"])
                    numeric_pass(row["after_300_calls"])
                    assert row["endurance_calls"] == ENDURANCE, "incomplete endurance"
                    row["timing_summary"] = timing_summary(row["samples_us"])
                ratios = record["rows"]["7988"]["timing_summary"]["fused_over_stock"]
                assert all(value <= 1 - MARGIN for value in ratios.values()), f"prefill-{worker} GPU {rank}: below 1% margin"
        original = env_output.read_text()
        lines = original.splitlines(keepends=True)
        assert [line.rstrip("\n") for line in lines if line.startswith(KEY + "=")] == [KEY + "=fp32"], "expected exactly one fp32 pin"
        assert time.time() <= deadline, "selection deadline expired"
        temporary = env_output.with_name(env_output.name + ".qkv.tmp")
        try:
            temporary.write_text("".join(KEY + "=fused\n" if line.startswith(KEY + "=") else line for line in lines))
            temporary.replace(env_output)
        finally:
            temporary.unlink(missing_ok=True)
        result.update(adopted=True, env={KEY: "fused"}, reason="all twelve H200 GPUs passed numerics/endurance and saved at least 1% in both timing orders")
    except Exception as error:  # noqa: BLE001 - gate failure keeps the existing fp32 pin
        result["reason"] = f"{type(error).__name__}: {error}"
    # Include all available raw records even when a worker timed out or the first GPU failed.
    if not result["adopted"]:
        result["gpus"] = []
        for path in sorted(directory.glob("prefill-*/rank-*.json")):
            try:
                record = json.loads(path.read_text(), parse_constant=reject_nonfinite)
                for row in record.get("rows", {}).values():
                    if "samples_us" in row:
                        row["timing_summary"] = timing_summary(row["samples_us"])
                result["gpus"].append(record)
            except Exception as error:  # noqa: BLE001 - malformed evidence must never abort serving
                result["gpus"].append({"path": str(path), "error": str(error)})
    return result


def model_norms(torch, model, rank):
    from safetensors import safe_open

    layer = (0, 24, 48, 72)[rank]
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    weights, sources = [], []
    for kind in ("q", "k"):
        key = f"model.layers.{layer}.self_attn.{kind}_norm.weight"
        shard = index[key]
        with safe_open(model / shard, framework="pt", device="cpu") as tensors:
            weight = tensors.get_tensor(key).to(device="cuda", dtype=torch.bfloat16)
        assert weight.shape == (128,) and bool(weight.isfinite().all()), "invalid model norm weight"
        weights.append(weight)
        sources.append({"key": key, "shard": shard, "min": weight.min().item(), "max": weight.max().item()})
    return weights, {"source": "staged-model", "model": str(model), "dtype": "bfloat16", "tensors": sources}


def measure(torch, probe, data, out):
    modes = ("stock", "fused", "gemm", "post")
    graphs = {mode: probe["capture"](lambda mode=mode: probe["call"](data, out, mode)) for mode in modes}
    samples = {order: {mode: [] for mode in modes} for order in ("forward", "reverse")}
    for _ in range(ROUNDS):
        for order, sequence in (("forward", modes), ("reverse", modes[::-1])):
            for mode in sequence:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[mode].replay()
                end.record()
                end.synchronize()
                samples[order][mode].append(start.elapsed_time(end) * 1000 / REPS)
    return samples


def run(output, run_id, deadline, model):
    begin = time.monotonic()
    import torch
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    result = {"schema": SCHEMA, "worker": output.name, "rank": rank, "run_id": run_id, "passed": False, "rows": {}}
    output.mkdir(parents=True, exist_ok=True)
    try:
        assert int(os.environ["WORLD_SIZE"]) == 4
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        result.update(device=torch.cuda.get_device_name(rank), uuid=str(torch.cuda.get_device_properties(rank).uuid))
        assert "H200" in result["device"] and torch.cuda.get_device_capability() == (9, 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        probe = runpy.run_path(str(Path(__file__).with_name("qkv-fused.py")))
        weights, result["norm_weights"] = model_norms(torch, model, rank)
        dist.init_process_group("gloo")
        # Compile once in the shared node-local caches, not four contending builds.
        if rank == 0:
            for m in ROWS:
                data = probe["make_case"](m)
                out = probe["outputs"](m)
                for mode in ("stock", "fused"):
                    probe["call"](data, out, mode)
                torch.cuda.synchronize()
                del data, out
        dist.barrier()
        for m in ROWS:
            assert time.time() <= deadline, "selection deadline expired"
            data = list(probe["make_case"](m))
            data[4:6] = weights
            fused, stock = probe["outputs"](m), probe["outputs"](m)
            probe["call"](data, stock, "stock")
            probe["call"](data, fused, "fused")
            exact = probe["reference"](data)
            row = result["rows"][str(m)] = {}
            for phase in ("initial", "after_300_calls"):
                if phase == "after_300_calls":
                    for _ in range(ENDURANCE):
                        probe["call"](data, fused, "fused")
                    torch.cuda.synchronize()
                    row["endurance_calls"] = ENDURANCE
                # check() emits all five tensor metrics and enforces tail/slot sentinels.
                with contextlib.redirect_stdout(io.StringIO()) as captured:
                    probe["check"](data, fused, stock, exact, phase)
                row[phase] = json.loads(captured.getvalue())
                numeric_pass(row[phase])
            row["samples_us"] = measure(torch, probe, data, fused)
            row["timing_summary"] = timing_summary(row["samples_us"])
            del data, fused, stock, exact
        assert time.time() <= deadline, "selection deadline expired"
        result["passed"] = True
    except Exception as error:  # noqa: BLE001 - persist error evidence, aggregate keeps fp32
        result["error"] = f"{type(error).__name__}: {error}"
    result.update(elapsed_seconds=time.monotonic() - begin, finished_at=time.time())
    temporary = output / f"rank-{rank}.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / f"rank-{rank}.json")
    print(json.dumps(result, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deadline", required=True, type=float)
    parser.add_argument("--workers-ok", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.aggregate:
        result = aggregate(args.aggregate, args.env_output, args.run_id, args.deadline, bool(args.workers_ok))
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "gpus"}), flush=True)
    else:
        run(args.output, args.run_id, args.deadline, args.model)


if __name__ == "__main__":
    main()
