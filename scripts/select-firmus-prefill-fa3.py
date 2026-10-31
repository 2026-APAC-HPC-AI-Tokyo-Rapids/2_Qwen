"""Select fixed Prefill on every assigned worker GPU before loading the model.

Endurance gate (--endurance, one torchrun per Prefill worker after every
worker's screen): run 40991744289 adopted n192-direct-o-varlen-pipe after
about 2,000 graph-replayed calls per GPU on 3986, 4027, 4096 and 3986+4027,
and all three Prefill workers stopped after 7 to 26 microbatches of two
4001-token requests (about 170-620 eager calls per GPU) with no CUDA error.
Each GPU recomputes the row the aggregate would adopt from the twelve screen
results and, if that row has not served (UNGATED), runs it through the serving
call path (prefill_fp8.attention, eager, a fresh metadata tensor and tile
counter memset per call) for ENDURANCE_CALLS calls rotating over
ENDURANCE_SHAPES, with the Q write of the QK-norm prologue before and an
unrelated GEMM after every call and a copy-engine copy on a side stream, as in
a PP4 stage. The last output of every shape must equal the pinned row's bytes.
--aggregate --endurance keeps such a row only if all twelve GPUs finished and
matched; otherwise it keeps the pinned row and records why.
"""
import argparse
import json
import math
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path
from runpy import run_path

LAYERS = 94
PAIRS = 20
LENGTHS = (3986, 4027, 4096)
# Two packed requests, timed only for varlen rows; control is then the prepared N224 path.
PAIR = (3986, 4027)
# Rows that have served whole runs: n192 (40201493911 and later), the PP4 pin n192-direct-o-varlen
# (40883389891 and later) and the prepared N224 control. Any other winner needs the endurance gate.
UNGATED = ("control", "n192", "n192-direct-o-varlen")
# 40991744289 stopped after about 170-620 calls per GPU; 1200 calls of 0.4-0.8 ms is about 2 s with the GEMMs.
ENDURANCE_CALLS = 1200
# The serving microbatch (two 4001-token requests in every batch of 40991744289), two other packed pairs,
# and single requests at the serving length and at both ends of the fixed range (3585 leaves a one-token
# last M-tile, as 4001 does).
ENDURANCE_SHAPES = ((4001, 4001), (4032, 4032), (3986, 4027), (4001,), (3585,), (4096,))
_table = Path(__file__).resolve().parents[1] / "kernels/fa3-verify/prefill-candidates.json"
if not _table.is_file():
    _table = Path(__file__).resolve().parents[1] / "source/prefill-candidates.json"
if not _table.is_file():
    _table = Path("/opt/apac/fa3-verify/prefill-candidates.json")
CANDIDATES = json.loads(_table.read_text())
# Native V rows require their own producer, independent dump and composed gate.
NAMES = tuple(c["name"] for c in CANDIDATES if not c.get("native_v_n"))
VARLEN = {c["name"] for c in CANDIDATES if c.get("varlen")}


def shapes(name):
    return [str(length) for length in LENGTHS] + (["+".join(map(str, PAIR))] if name in VARLEN else [])


def choose_candidate(ranks, expected_ranks=4):
    """Rows screened on rank 0 compete; varlen rows must also win on the pair."""
    assert len(ranks) == expected_ranks and {r["rank"] for r in ranks} == set(range(expected_ranks))
    screened = [name for name in NAMES[1:] if name in ranks[0]["lengths"][str(LENGTHS[0])]]
    choices = {}
    for name in screened:
        entries = [r["lengths"].get(key, {}).get(name, {}) for r in ranks for key in shapes(name)]
        if any(not e.get("numeric_passed") or e.get("error") or len(e.get("control", [])) != PAIRS or len(e.get("candidate", [])) != PAIRS or any(not math.isfinite(x) or x <= 0 for x in e.get("control", []) + e.get("candidate", [])) for e in entries):
            choices[name] = {"eligible": False, "reason": "gate-or-timing-failed"}
            continue
        savings = [statistics.median(e["control"]) - statistics.median(e["candidate"]) for e in entries]
        medians = [statistics.median(r["lengths"][str(length)][name]["candidate"]) for r in ranks for length in LENGTHS]
        choices[name] = {"eligible": min(savings) >= 5.0, "savings_us_by_rank_length": savings, "minimum_saving_us": min(savings), "median_us": statistics.median(medians)}
    eligible = [n for n, e in choices.items() if e["eligible"]]
    winner = min(eligible, key=lambda n: (choices[n]["median_us"], NAMES.index(n))) if eligible else "control"
    return {"winner": winner, "choices": choices, "env": {"SGLANG_PREFILL_FA3_VERIFY": winner}, "rule": {"all_ranks_and_lengths": True, "minimum_saving_us_per_layer": 5.0, "lengths": LENGTHS, "varlen_pair": PAIR, "screened": screened, "gpus": expected_ranks}}


def measure(torch, graph):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / LAYERS


def run_selection(output, names, control_row="control"):
    begin = time.monotonic()
    import torch
    import torch.distributed as dist
    from sglang.srt.layers.attention.decode_fa3_verify import load

    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 4
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    assert torch.cuda.get_device_properties(rank).multi_processor_count == 132
    dist.init_process_group("gloo", timeout=timedelta(seconds=175))
    load()
    checks = run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
    manifest = json.loads(Path("/opt/apac/fa3-verify/manifest.json").read_text())
    assert [{k: v for k, v in c.items() if k != "shared_bytes"} for c in manifest["prefill_candidates"]] == CANDIDATES
    local = {"rank": rank, "control_row": control_row, "lengths": {}, "device": torch.cuda.get_device_name(rank), "uuid": str(torch.cuda.get_device_properties(rank).uuid), "binary_sha256": manifest["binary_sha256"]}
    gathered = [None] * 4
    for key in dict.fromkeys(key for name in names for key in shapes(name)):
        common = checks["prefill_selection_inputs"](tuple(map(int, key.split("+"))))
        original_inputs = common["packed"].clone()
        control = checks["prefill_selection_step"](common, control_row, LAYERS)
        control_graph = checks["prefill_selection_capture"](control)
        local["lengths"][key] = {}
        for name in [n for n in names if key in shapes(n)]:
            result = {"numeric_passed": False, "control": [], "candidate": []}
            graph = step = None
            common["packed"].copy_(original_inputs)
            try:
                result.update(checks["prefill_validate_candidate"](common, name))
                step = checks["prefill_selection_step"](common, name, LAYERS)
                graph = checks["prefill_selection_capture"](step)
            except Exception as error:  # noqa: BLE001 - failed candidates retain control
                result["error"] = f"{type(error).__name__}: {error}"
            common["packed"].copy_(original_inputs)
            dist.all_gather_object(gathered, {"passed": result["numeric_passed"] and graph is not None})
            if all(r["passed"] for r in gathered):
                try:
                    for pair in range(PAIRS):
                        order = (("control", control_graph), ("candidate", graph))
                        if pair % 2:
                            order = tuple(reversed(order))
                        for kind, current in order:
                            result[kind].append(measure(torch, current))
                except Exception as error:  # noqa: BLE001 - failed timing retains control
                    result.update(numeric_passed=False, error=f"timing {type(error).__name__}: {error}")
            else:
                result.setdefault("error", "candidate rejected on another rank")
            local["lengths"][key][name] = result
            print(json.dumps({"rank": rank, "length": key, "name": name, **result}), flush=True)
            del graph, step
        del common, control, control_graph, original_inputs
    local["elapsed_seconds"] = time.monotonic() - begin
    dist.all_gather_object(gathered, local)
    if rank == 0:
        result = {"schema": "firmus-prefill-fa3-selection/v1", "ranks": gathered, "layers": LAYERS, "pairs": PAIRS, "includes_qkv_producer": True, "elapsed_seconds": max(r["elapsed_seconds"] for r in gathered)}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    dist.destroy_process_group()


def screen_selection(directory, workers):
    """choose_candidate over every worker's screen; an incomplete worker retains control."""
    ranks = []
    failed = []
    for worker in range(workers):
        path = directory / f"prefill-{worker}"
        if not (path / "status.txt").is_file() or (path / "status.txt").read_text().strip() != "fa3=0":
            failed.append(worker)
            continue
        fragment = json.loads((path / "fa3.json").read_text())
        for entry in fragment["ranks"]:
            ranks.append({**entry, "local_rank": entry["rank"], "rank": worker*4 + entry["rank"], "worker": worker})
    if failed:
        return {"winner": "control", "env": {"SGLANG_PREFILL_FA3_VERIFY": "control"}, "failed_workers": failed}
    return choose_candidate(ranks, expected_ranks=workers * 4)


def endurance_inputs(torch, lengths, device):
    rows = sum(lengths)
    generator = torch.Generator(device=device).manual_seed(93601 + rows)
    q, k, v = [(torch.randn(rows, heads, 128, device=device, generator=generator) * .5).to(torch.float8_e4m3fn) for heads in (64, 4, 4)]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
    return {"q": q, "q_source": q.clone(), "k": k, "v": v, "cu": cu, "maxlen": max(lengths), "rows": rows}


def run_endurance(selection, output, workers):
    """Run the row the aggregate would adopt, if it has not served, for ENDURANCE_CALLS serving-path calls."""
    begin = time.monotonic()
    rank = int(os.environ["LOCAL_RANK"])
    row = screen_selection(selection, workers)["winner"]
    local = {"rank": rank, "row": row, "calls": 0, "needed": row not in UNGATED}
    if local["needed"]:
        # The serving call path reads the adopted row from this variable once (lru_cache).
        os.environ["SGLANG_PREFILL_FA3_VERIFY"] = row
        import torch
        from sglang.srt.layers.attention.decode_fa3_verify import load
        from sglang.srt.layers.attention.prefill_fp8 import attention, prefill_candidate

        torch.cuda.set_device(rank)
        load()
        device = torch.device("cuda", rank)
        assert torch.cuda.get_device_capability() == (9, 0) and prefill_candidate()["name"] == row
        # The gated rows are varlen rows that must write n192's bytes per request, which the served pin
        # n192-direct-o-varlen also writes; its output on the same packed input is the reference.
        spec = CANDIDATES[NAMES.index(row)]
        assert spec.get("varlen") and spec.get("bitwise_reference") == "n192", spec
        pin = NAMES.index("n192-direct-o-varlen")
        shapes = {"+".join(map(str, lengths)): endurance_inputs(torch, lengths, device) for lengths in ENDURANCE_SHAPES}
        reference = {}
        for key, x in shapes.items():
            out = torch.empty((x["rows"], 64, 128), device=device, dtype=torch.float8_e4m3fn)
            scales = torch.empty((64, (x["rows"] + 3) // 4 * 4), device=device, dtype=torch.float32).T[:x["rows"]]
            lse = torch.empty((64, x["rows"]), device=device, dtype=torch.float32)
            metadata = torch.empty(17, device=device, dtype=torch.int32)
            empty = torch.empty(0, device=device, dtype=torch.uint8)
            torch.ops.apac_fa3_verify.prefill_variant_fwd_out(x["q"], x["k"], x["v"], empty, x["cu"], out, scales, lse, metadata, x["maxlen"], 128**-.5, pin)
            reference[key] = (out.view(x["rows"], 8192), scales)
        # Serving context of a PP4 stage: the O projection and MoE GEMMs between attention calls, and
        # copy-engine traffic (proxy copies, KV transfer) beside them.
        a = torch.randn(8192, 4096, device=device, dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
        c = torch.empty(8192, 4096, device=device, dtype=torch.bfloat16)
        side = torch.cuda.Stream(device)
        copy_source = torch.empty(64 << 20, device=device, dtype=torch.uint8)
        copy_target = torch.empty_like(copy_source)
        torch.cuda.synchronize()
        last = {}
        start = time.monotonic()
        keys = list(shapes)
        for call in range(ENDURANCE_CALLS):
            key = keys[call % len(keys)]
            x = shapes[key]
            with torch.cuda.stream(side):
                copy_target.copy_(copy_source)
            x["q"].copy_(x["q_source"])
            last[key] = attention(x["q"], x["k"], x["v"], x["cu"], x["maxlen"], 128**-.5, x["rows"], fixed=True)
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        local["calls"] = ENDURANCE_CALLS
        local["calls_seconds"] = time.monotonic() - start
        local["bytes_equal"] = {key: bool(torch.equal(last[key][0].view(torch.uint8), reference[key][0].view(torch.uint8)) and torch.equal(last[key][1], reference[key][1])) for key in keys}
    local["passed"] = not local["needed"] or all(local["bytes_equal"].values())
    local["elapsed_seconds"] = time.monotonic() - begin
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(local, indent=2, allow_nan=False) + "\n")
    print(json.dumps(local), flush=True)


def endurance_failure(endurance, row, workers):
    """None if every GPU ran row for ENDURANCE_CALLS calls and matched the pinned row's bytes, else why not."""
    for worker in range(workers):
        for rank in range(4):
            path = endurance / f"prefill-{worker}" / f"rank-{rank}.json"
            if not path.is_file():
                return f"no endurance result from prefill-{worker} rank {rank} (late, hung or failed)"
            entry = json.loads(path.read_text())
            if entry["row"] != row:
                return f"prefill-{worker} rank {rank} gated {entry['row']}, not {row}"
            if entry["calls"] != ENDURANCE_CALLS or not entry["passed"]:
                return f"prefill-{worker} rank {rank} failed the endurance gate: {entry['calls']} calls, bytes_equal {entry.get('bytes_equal')}"
    return None


def aggregate_selection(directory, output, env_output, workers=3, endurance=None):
    """All assigned Prefill GPUs must pass; an incomplete worker retains control.

    Other lines of env_output (the PP4 MegaMoE switch) are kept. When env_output
    already names a row (the PP4 pinned n192), a losing or failed screen keeps
    that row instead of control. With endurance (the PP4 gate results), a winner
    outside UNGATED is adopted only if every GPU passed the endurance gate."""
    kept = env_output.read_text().splitlines() if env_output.is_file() else []
    pinned = [line.split("=", 1)[1] for line in kept if line.startswith("SGLANG_PREFILL_FA3_VERIFY=")]
    kept = [line + "\n" for line in kept if not line.startswith("SGLANG_PREFILL_FA3_VERIFY=")]
    selection = screen_selection(directory, workers)
    if endurance is not None and selection["winner"] not in UNGATED:
        failure = endurance_failure(endurance, selection["winner"], workers)
        selection["endurance"] = failure or f"passed: {ENDURANCE_CALLS} calls on every GPU"
        if failure is not None:
            selection.update(winner="control", env={"SGLANG_PREFILL_FA3_VERIFY": "control"})
    if selection["winner"] == "control" and pinned:
        selection["env"] = {"SGLANG_PREFILL_FA3_VERIFY": pinned[-1]}
    result = {"schema": "firmus-prefill-fa3-selection-all-workers/v2", **selection, "fragments": [f"prefill-{i}/fa3.json" for i in range(workers)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    env_output.write_text("".join(kept) + "".join(f"{k}={v}\n" for k, v in result["env"].items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-output", type=Path)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--endurance", type=Path, help="per-worker endurance results (aggregate), or the screen results to gate (endurance run)")
    parser.add_argument("--workers", type=int, choices=(2, 3), default=3)
    parser.add_argument("--names", default=",".join(NAMES[1:]), help="comma-separated rows to screen against control")
    parser.add_argument("--control", default="control", choices=NAMES, help="row the screened rows are timed against (PP4: the pinned row)")
    args = parser.parse_args()
    if args.aggregate:
        assert args.env_output is not None
        aggregate_selection(args.aggregate, args.output, args.env_output, args.workers, args.endurance)
    elif args.endurance:
        run_endurance(args.endurance, args.output, args.workers)
    else:
        names = args.names.split(",")
        assert names and set(names) <= set(NAMES[1:]) and args.control not in names, names
        run_selection(args.output, names, args.control)
