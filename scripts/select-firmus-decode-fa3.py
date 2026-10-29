"""Screen/tune the AOT verify companion on four GPUs within 180 seconds.

No model, compiler, profiler or scheduler submission. The configured cap/4
selects a worker setting; cap/4-4 and cap/4-2 are retained as diagnostics.

With --pinned-env (PP4, which keeps the prior Decode settings), the pinned
candidate is the timing reference ("stock" in the records) and only the
two-CTA-per-SM rows compete; the written environment holds only a winner.
"""

import argparse
import json
import math
import os
import shlex
import statistics
import time
from datetime import timedelta
from pathlib import Path
from runpy import run_path

BLOCKS = 2
PAIRS = 20
LAYERS = 94
_candidate_table = Path(__file__).resolve().parents[1] / "kernels/fa3-verify/candidates.json"
if not _candidate_table.is_file():
    _candidate_table = Path("/opt/apac/fa3-verify/candidates.json")
FA3_VERIFY_CANDIDATES = json.loads(_candidate_table.read_text())
NAMES = tuple(c["name"] for c in FA3_VERIFY_CANDIDATES)
SPECS = {c["name"]: c for c in FA3_VERIFY_CANDIDATES}
# PP4 screen: the rows that change only the CTAs per SM (and so the split layout) of the pinned kernel.
PINNED_SCREEN = tuple(c["name"] for c in FA3_VERIFY_CANDIDATES if c.get("ctas", 1) == 2)
PINNED_MINIMUM_SAVING_US = 1.0


def summary(values):
    assert values and all(math.isfinite(x) and x > 0 for x in values)
    values = sorted(values)
    return {"median_us": statistics.median(values), "p10_us": values[len(values) // 10], "p90_us": values[len(values) * 9 // 10]}


def choose_candidate(ranks, selection_batch):
    """All four GPUs, configured cap/4 only, both blocks and >=3.2us/layer on every GPU."""
    assert len(ranks) == 4 and {r["rank"] for r in ranks} == set(range(4))
    choices = {}
    for name in NAMES:
        entries = [r["batches"][str(selection_batch)][name] for r in ranks]
        if any(not entry.get("numeric_passed") or entry.get("error") or len(entry.get("blocks", [])) != BLOCKS for entry in entries):
            choices[name] = {"eligible": False, "reason": "candidate-failed", "ranks": entries}
            continue
        blocks = [block for entry in entries for block in entry["blocks"]]
        if any(not b.get("stock") or not b.get("candidate") for b in blocks):
            choices[name] = {"eligible": False, "reason": "missing-timing"}
            continue
        savings = [statistics.median(b["stock"]) - statistics.median(b["candidate"]) for b in blocks]
        timing = summary([x for b in blocks for x in b["candidate"]])
        choices[name] = {**timing, "savings_us_by_rank_block": savings, "eligible": name != "control96" and min(savings) >= 3.2, "minimum_saving_us": min(savings)}
    eligible = [name for name, result in choices.items() if result["eligible"]]
    winner = min(eligible, key=lambda name: (choices[name]["median_us"], NAMES.index(name))) if eligible else None
    return {"winner": winner, "choices": choices, "rule": {"selection_batch": selection_batch, "minimum_saving_us_per_layer": 3.2, "blocks": BLOCKS, "gpus": 4, "all_rank_blocks_faster": True}, "env": {"SGLANG_DECODE_FA3_VERIFY": winner or ""}}


def choose_pinned(ranks, selection_batch, control, names):
    """PP4: replace the pinned candidate only with a row that passed the numeric gate, wrote the
    pinned kernel's exact bytes where its arithmetic is the same (equal splits), and saved
    >=1us/layer in both blocks on every GPU at cap/4."""
    assert len(ranks) == 4 and {r["rank"] for r in ranks} == set(range(4))
    choices = {}
    for name in names:
        entries = [r["batches"][str(selection_batch)][name] for r in ranks]
        if any(not entry.get("numeric_passed") or entry.get("error") or len(entry.get("blocks", [])) != BLOCKS for entry in entries):
            choices[name] = {"eligible": False, "reason": "candidate-failed", "ranks": entries}
            continue
        if any(entry.get("bytes_equal_to_control") is False for entry in entries):
            choices[name] = {"eligible": False, "reason": "not-bit-exact"}
            continue
        blocks = [block for entry in entries for block in entry["blocks"]]
        if any(not b.get("stock") or not b.get("candidate") for b in blocks):
            choices[name] = {"eligible": False, "reason": "missing-timing"}
            continue
        savings = [statistics.median(b["stock"]) - statistics.median(b["candidate"]) for b in blocks]
        choices[name] = {**summary([x for b in blocks for x in b["candidate"]]), "control": summary([x for b in blocks for x in b["stock"]]), "savings_us_by_rank_block": savings, "eligible": min(savings) >= PINNED_MINIMUM_SAVING_US, "minimum_saving_us": min(savings)}
    eligible = [name for name, result in choices.items() if result["eligible"]]
    winner = min(eligible, key=lambda name: (choices[name]["median_us"], NAMES.index(name))) if eligible else None
    return {"winner": winner, "control": control, "choices": choices, "rule": {"selection_batch": selection_batch, "minimum_saving_us_per_layer": PINNED_MINIMUM_SAVING_US, "blocks": BLOCKS, "gpus": 4, "all_rank_blocks_faster": True, "equal_split_bytes_exact": True}, "env": {"SGLANG_DECODE_FA3_VERIFY": winner} if winner else {}}


def read_env(path):
    """KEY=VALUE lines as written by prepare-firmus-pp4-settings.py (shell-quoted values)."""
    env = {}
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        env[key] = "".join(shlex.split(value))
    return env


def exact_reference(control, name):
    """Equal splits and no uneven metadata: the same per-tile arithmetic, so the bytes must match."""
    if control == "stock":
        return False
    a, b = SPECS[control], SPECS[name]
    return a["splits"] == b["splits"] and not a.get("uneven") and not b.get("uneven") and a["output"] == b["output"]


def bytes_equal(torch, actual, expected):
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    as_bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
    return len(actual) == len(expected) and all(torch.equal(x.view(as_bits[x.element_size()]), y.view(as_bits[y.element_size()])) for x, y in zip(actual, expected))


def measure(torch, graph):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / LAYERS


def prepare_candidate(torch, checks, batch, name, common_sets):
    result = {"numeric_passed": False, "blocks": []}
    try:
        result.update(checks["verify_validate_candidate"](batch, name))
        step = checks["verify_step"](common_sets, name)
        graph = checks["verify_capture"](step)
        graph.replay()
        torch.cuda.synchronize()
        return result, step, graph
    except Exception as error:  # noqa: BLE001 - the caller proceeds to the next candidate
        result.update(numeric_passed=False, error=f"{type(error).__name__}: {error}")
        return result, None, None


def run_selection(output, env_output, pinned_env=None):
    begin = time.monotonic()
    import torch
    import torch.distributed as dist
    from sglang.srt.layers.attention import decode_fa3_verify

    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 4
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    assert torch.cuda.get_device_properties(rank).multi_processor_count == 132
    dist.init_process_group("gloo", timeout=timedelta(seconds=175))
    checks = run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
    decode_fa3_verify.load()
    manifest = json.loads(Path("/opt/apac/fa3-verify/manifest.json").read_text())
    local = {"rank": rank, "device": torch.cuda.get_device_name(rank), "batches": {}, "binary_sha256": manifest["binary_sha256"], "uuid": str(torch.cuda.get_device_properties(rank).uuid), "clock_mhz_start": torch.cuda.clock_rate(rank), "power_w_start": torch.cuda.power_draw(rank) / 1000}
    gathered = [None] * 4
    config = json.loads((Path(__file__).resolve().parents[1] / "source/firmus-framework.json").read_text())
    cap = config["decode_max_running_requests"]
    selection_batch = cap // config["decode_dp_size"]
    batches = (selection_batch - 4, selection_batch - 2, selection_batch)
    control = (read_env(pinned_env).get("SGLANG_DECODE_FA3_VERIFY") or "stock") if pinned_env else "stock"
    screened = PINNED_SCREEN if pinned_env else NAMES
    for batch in batches:
        common_sets = [checks["verify_inputs"](batch, 93101 + batch + i)[0] for i in range(4)]
        # Same live metadata tensors for every layer, as in the B4 serving path.
        for c in common_sets[1:]:
            c["cache_seqlens"] = common_sets[0]["cache_seqlens"]
            c["cu_seqlens_q"] = common_sets[0]["cu_seqlens_q"]
        # The timing reference: stock, or the pinned candidate in PP4 (kept under the "stock" key).
        stock_step = checks["verify_step"](common_sets, control)
        stock_graph = checks["verify_capture"](stock_step)
        local["batches"][str(batch)] = {}
        names = screened if batch % 2 == 0 else tuple(reversed(screened))
        for name in names:
            result, step, graph = prepare_candidate(torch, checks, batch, name, common_sets)
            if graph is not None and exact_reference(control, name):
                stock_graph.replay()
                torch.cuda.synchronize()
                result["bytes_equal_to_control"] = bytes_equal(torch, step.output, stock_step.output)
            local["batches"][str(batch)][name] = result
            dist.all_gather_object(gathered, {"passed": result["numeric_passed"] and "error" not in result})
            if all(r["passed"] for r in gathered):
                for block in range(BLOCKS):
                    times = {"stock": [], "candidate": []}
                    try:
                        dist.barrier()
                        stock_graph.replay()
                        graph.replay()
                        for pair in range(PAIRS):
                            order = (("stock", stock_graph), ("candidate", graph))
                            if (pair + block) % 2:
                                order = tuple(reversed(order))
                            for kind, current in order:
                                times[kind].append(measure(torch, current))
                        result["blocks"].append(times)
                    except Exception as error:  # noqa: BLE001 - one failed candidate must not stop selection
                        result.update(numeric_passed=False, error=f"timing {type(error).__name__}: {error}")
                    dist.all_gather_object(gathered, {"passed": result["numeric_passed"]})
                    if not all(r["passed"] for r in gathered):
                        result.setdefault("error", "timing failed on another rank")
                        break
            else:
                result.setdefault("error", "candidate rejected on another rank")
            print(json.dumps({"rank": rank, "batch": batch, "candidate": name, **result}), flush=True)
            del graph, step
        del stock_graph, stock_step, common_sets
    local.update(clock_mhz_end=torch.cuda.clock_rate(rank), power_w_end=torch.cuda.power_draw(rank) / 1000)
    local["elapsed_seconds"] = time.monotonic() - begin
    dist.all_gather_object(gathered, local)
    if rank == 0:
        chosen = choose_pinned(gathered, selection_batch, control, PINNED_SCREEN) if pinned_env else choose_candidate(gathered, selection_batch)
        result = {"schema": "firmus-decode-fa3-selection/v1", **chosen, "ranks": gathered, "elapsed_seconds": max(r["elapsed_seconds"] for r in gathered), "layers": LAYERS, "kv_sets": 4, "pairs_per_block": PAIRS, "cap": cap, "selection_batches": [selection_batch], "diagnostic_batches": list(batches[:-1])}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        env_output.write_text("".join(f"{key}={value}\n" for key, value in result["env"].items()))
        print(json.dumps({"phase": "selected", "env": result["env"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-output", type=Path, required=True)
    parser.add_argument("--pinned-env", type=Path, help="PP4: pinned decode-kernels.env; screen only the two-CTA rows against its FA3 candidate")
    args = parser.parse_args()
    run_selection(args.output, args.env_output, args.pinned_env)
