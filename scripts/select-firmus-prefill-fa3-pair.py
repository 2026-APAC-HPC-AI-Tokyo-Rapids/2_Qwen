"""Default-off two-request FA3 gate, before model load on three parallel workers.

CPU aggregation needs only the standard library. Any incomplete or timed-out
screen retains the served pipe row. Native rows are built ahead of the job.
"""

import argparse
import json
import math
import os
import runpy
import statistics
import time
from pathlib import Path

KEY = "SGLANG_PREFILL_FA3_VERIFY"
CONTROL = "n192-direct-o-varlen-pipe"
CANDIDATE = "m192n96-direct-o-varlen-pipe"
DISPATCH = CONTROL + "+m192n96-pair"
SCHEMA = "firmus-prefill-fa3-pair/v1"
SHAPES = ((3994, 3994), (3986, 4027), (3986, 3986), (4027, 4027), (4096, 4096), (3900, 4096), (4096, 3901))
ROUNDS, REPS, ENDURANCE = 10, 12, 1200
SECONDS = 88


def reject_nonfinite(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def timing_summary(samples):
    savings, medians = {}, {}
    for order in ("forward", "reverse"):
        medians[order] = {}
        for mode in ("control", "candidate"):
            values = samples[order][mode]
            assert len(values) == ROUNDS, "incomplete timing"
            assert all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values), "invalid timing"
            medians[order][mode] = statistics.median(values)
        savings[order] = medians[order]["control"] - medians[order]["candidate"]
    savings["pooled"] = statistics.median(samples["forward"]["control"] + samples["reverse"]["control"]) - statistics.median(samples["forward"]["candidate"] + samples["reverse"]["candidate"])
    return {"order_median_us": medians, "savings_us": savings, "order_delta_us": {mode: abs(medians["forward"][mode] - medians["reverse"][mode]) for mode in ("control", "candidate")}}


def numeric_pass(check):
    assert check["numeric_passed"] is True and check["changed_input_replays"] == 2, "failed numeric/replay check"
    for key in ("metrics", "precision_reference_metrics"):
        assert len(check[key]) == 3, "incomplete numerical check"
        for metric in check[key]:
            for name, limit in (("relative_l2", .05), ("max_scaled_error", .25)):
                value = metric[name]
                assert type(value) in (float, int) and math.isfinite(value) and 0 <= value <= limit, "numerical tolerance exceeded"
    assert check["precision_reference"] == CONTROL, "wrong reference row"


def aggregate(directory, env_output, run_id, deadline, workers_ok):
    result = {"schema": SCHEMA, "adopted": False, "env": {KEY: CONTROL}, "rule": {"bounds_inclusive": (3900, 4096), "sequences": 2, "shapes": SHAPES, "rounds_per_order": ROUNDS, "calls_per_sample": REPS, "minimum_saving_us": 5, "require_both_orders": True, "endurance_calls": ENDURANCE, "worker_seconds": SECONDS}, "gpus": []}
    try:
        assert workers_ok, "selector process failed or timed out"
        assert time.time() <= deadline, "selection deadline expired"
        uuids = set()
        for worker in range(3):
            for rank in range(4):
                entry = json.loads((directory / f"prefill-{worker}" / f"rank-{rank}.json").read_text(), parse_constant=reject_nonfinite)
                result["gpus"].append(entry)
                assert entry["schema"] == SCHEMA and entry["run_id"] == run_id, "stale result"
                assert entry["worker"] == f"prefill-{worker}" and entry["rank"] == rank, "wrong GPU result"
                assert "H200" in entry["device"] and entry["uuid"] and entry["uuid"] not in uuids, "wrong or repeated GPU"
                uuids.add(entry["uuid"])
                assert entry["finished_at"] <= deadline and 0 < entry["elapsed_seconds"] < SECONDS, "late result"
                assert entry["passed"] is True and set(entry["shapes"]) == {"+".join(map(str, s)) for s in SHAPES}, "incomplete screen"
                for shape in entry["shapes"].values():
                    numeric_pass(shape["numeric"])
                    shape["timing_summary"] = timing_summary(shape["samples_us"])
                    assert all(s >= 5 for s in shape["timing_summary"]["savings_us"].values()), "below 5 us margin"
                endurance = entry["endurance"]
                assert endurance["calls"] == ENDURANCE and endurance["bytes_equal"] is True and endurance["direct_equal"] is True, "failed endurance"
                for metric in (endurance["initial_fp32"], endurance["final_fp32"]):
                    assert all(type(metric[k]) in (int, float) and math.isfinite(metric[k]) and 0 <= metric[k] <= bound for k, bound in (("relative_l2", .05), ("max_scaled_error", .25))), "endurance numerical check failed"
        lines = env_output.read_text().splitlines(keepends=True)
        assert [line.rstrip("\n") for line in lines if line.startswith(KEY + "=")] == [KEY + "=" + CONTROL], "expected exactly one control pin"
        assert time.time() <= deadline, "selection deadline expired"
        temporary = env_output.with_name(env_output.name + ".fa3-pair.tmp")
        try:
            temporary.write_text("".join(KEY + "=" + DISPATCH + "\n" if line.startswith(KEY + "=") else line for line in lines))
            temporary.replace(env_output)
        finally:
            temporary.unlink(missing_ok=True)
        result.update(adopted=True, env={KEY: DISPATCH}, reason="all twelve H200 GPUs passed numerics/endurance and saved at least 5 us in both orders on every pair")
    except Exception as error:  # noqa: BLE001 - retain the control env on any gate failure
        result["reason"] = f"{type(error).__name__}: {error}"
    if not result["adopted"]:
        result["gpus"] = []
        for path in sorted(directory.glob("prefill-*/rank-*.json")):
            try:
                result["gpus"].append(json.loads(path.read_text(), parse_constant=reject_nonfinite))
            except Exception as error:  # noqa: BLE001 - malformed evidence cannot abort serving
                result["gpus"].append({"path": str(path), "error": str(error)})
    return result


def measure(torch, checks, common):
    # The outputs/metadata are allocated before capture. Retain both closures so
    # the caching allocator cannot recycle their storage while a graph uses it.
    steps = {mode: checks["prefill_selection_step"](common, row, REPS) for mode, row in (("control", CONTROL), ("candidate", CANDIDATE))}
    graphs = {mode: checks["prefill_selection_capture"](step) for mode, step in steps.items()}
    samples = {order: {mode: [] for mode in graphs} for order in ("forward", "reverse")}
    for _ in range(ROUNDS):
        for order, modes in (("forward", ("control", "candidate")), ("reverse", ("candidate", "control"))):
            for mode in modes:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[mode].replay()
                end.record()
                end.synchronize()
                samples[order][mode].append(start.elapsed_time(end) * 1000 / REPS)
    return samples


def endurance(torch, checks):
    from sglang.srt.layers.attention.prefill_fp8 import (
        attention,
        pair_candidate,
        prefill_candidate,
    )

    common = checks["prefill_selection_inputs"]((3994, 3994), padding=0)
    common["fp32_producer"] = True
    direct = checks["prefill_selection_step"](common, CANDIDATE, 1)
    direct()  # Includes the real Q/K-norm/RoPE producer.
    lengths = common["lengths"]
    os.environ[KEY] = DISPATCH
    prefill_candidate.cache_clear()
    assert pair_candidate(lengths)["name"] == CANDIDATE

    def call():
        return attention(common["q"], common["k"], common["v"], common["cu"], max(lengths), 128**-.5, common["rows"], fixed=True, lengths=lengths)

    first = tuple(x.clone() for x in call())
    direct_equal = torch.equal(first[0].view(torch.uint8), direct.output[0].view(common["rows"], 8192).view(torch.uint8)) and torch.equal(first[1].contiguous().view(torch.uint8), direct.output[1].contiguous().view(torch.uint8))
    assert direct_equal, "serving dispatch differs from direct native row"
    indices, reference = checks["prefill_fp32_reference"](common)

    def validate(output):
        return checks["prefill_assert_errors"](output[0].view(common["rows"], 64, 128).float()[indices] * output[1][indices, :, None], reference, "serving_fp32")

    initial = validate(first)
    for _ in range(ENDURANCE):
        last = call()
    torch.cuda.synchronize()
    final = validate(last)
    equal = torch.equal(first[0].view(torch.uint8), last[0].view(torch.uint8)) and torch.equal(first[1].contiguous().view(torch.uint8), last[1].contiguous().view(torch.uint8))
    assert equal, "serving endurance output changed"
    return {"calls": ENDURANCE, "direct_equal": bool(direct_equal), "bytes_equal": bool(equal), "initial_fp32": initial, "final_fp32": final}


def run(output, run_id, deadline):
    begin = time.monotonic()
    import torch
    from sglang.srt.layers.attention.decode_fa3_verify import load

    rank = int(os.environ["LOCAL_RANK"])
    result = {"schema": SCHEMA, "worker": output.name, "rank": rank, "run_id": run_id, "passed": False, "shapes": {}}
    output.mkdir(parents=True, exist_ok=True)
    try:
        assert int(os.environ["WORLD_SIZE"]) == 4
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        result.update(device=torch.cuda.get_device_name(rank), uuid=str(torch.cuda.get_device_properties(rank).uuid))
        assert "H200" in result["device"] and torch.cuda.get_device_capability() == (9, 0)
        load()
        manifest = json.loads(Path("/opt/apac/fa3-verify/manifest.json").read_text())
        result["binary_sha256"] = manifest["binary_sha256"]
        checks = runpy.run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
        assert [{k: v for k, v in row.items() if k != "shared_bytes"} for row in manifest["prefill_candidates"]] == checks["prefill_specs"](), "native table mismatch"
        for lengths in SHAPES:
            assert time.time() <= deadline, "selection deadline expired"
            common = checks["prefill_selection_inputs"](lengths)
            common["fp32_producer"] = True
            row = result["shapes"]["+".join(map(str, lengths))] = {}
            row["numeric"] = checks["prefill_validate_candidate"](common, CANDIDATE)
            numeric_pass(row["numeric"])
            row["samples_us"] = measure(torch, checks, common)
            row["timing_summary"] = timing_summary(row["samples_us"])
            del common
        assert all(s >= 5 for row in result["shapes"].values() for s in row["timing_summary"]["savings_us"].values()), "below 5 us margin"
        result["endurance"] = endurance(torch, checks)
        assert time.time() <= deadline and time.monotonic() - begin < SECONDS, "selection deadline expired"
        result["passed"] = True
    except Exception as error:  # noqa: BLE001 - persist evidence; aggregate retains control
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deadline", required=True, type=float)
    parser.add_argument("--workers-ok", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.aggregate:
        result = aggregate(args.aggregate, args.env_output, args.run_id, args.deadline, bool(args.workers_ok))
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "gpus"}), flush=True)
    else:
        run(args.output, args.run_id, args.deadline)


if __name__ == "__main__":
    main()
