"""Select fixed Prefill on every assigned worker GPU before loading the model."""
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
_table = Path(__file__).resolve().parents[1] / "kernels/fa3-verify/prefill-candidates.json"
if not _table.is_file():
    _table = Path(__file__).resolve().parents[1] / "source/prefill-candidates.json"
if not _table.is_file():
    _table = Path("/opt/apac/fa3-verify/prefill-candidates.json")
CANDIDATES = json.loads(_table.read_text())
NAMES = tuple(c["name"] for c in CANDIDATES)
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


def run_selection(output, names):
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
    local = {"rank": rank, "lengths": {}, "device": torch.cuda.get_device_name(rank), "uuid": str(torch.cuda.get_device_properties(rank).uuid), "binary_sha256": manifest["binary_sha256"]}
    gathered = [None] * 4
    for key in dict.fromkeys(key for name in names for key in shapes(name)):
        common = checks["prefill_selection_inputs"](tuple(map(int, key.split("+"))))
        original_inputs = common["packed"].clone()
        control = checks["prefill_selection_step"](common, "control", LAYERS)
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


def aggregate_selection(directory, output, env_output, workers=3):
    """All assigned Prefill GPUs must pass; an incomplete worker retains control.

    Other lines of env_output (the PP4 MegaMoE switch) are kept. When env_output
    already names a row (the PP4 pinned n192), a losing or failed screen keeps
    that row instead of control."""
    kept = env_output.read_text().splitlines() if env_output.is_file() else []
    pinned = [line.split("=", 1)[1] for line in kept if line.startswith("SGLANG_PREFILL_FA3_VERIFY=")]
    kept = [line + "\n" for line in kept if not line.startswith("SGLANG_PREFILL_FA3_VERIFY=")]
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
        selection = {"winner": "control", "env": {"SGLANG_PREFILL_FA3_VERIFY": "control"}, "failed_workers": failed}
    else:
        selection = choose_candidate(ranks, expected_ranks=workers * 4)
    if selection["winner"] == "control" and pinned:
        selection["env"] = {"SGLANG_PREFILL_FA3_VERIFY": pinned[-1]}
    result = {"schema": "firmus-prefill-fa3-selection-all-workers/v1", **selection, "fragments": [f"prefill-{i}/fa3.json" for i in range(workers)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    env_output.write_text("".join(kept) + "".join(f"{k}={v}\n" for k, v in result["env"].items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-output", type=Path)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--workers", type=int, choices=(2, 3), default=3)
    parser.add_argument("--names", default=",".join(NAMES[1:]), help="comma-separated rows to screen against control")
    args = parser.parse_args()
    if args.aggregate:
        assert args.env_output is not None
        aggregate_selection(args.aggregate, args.output, args.env_output, args.workers)
    else:
        names = args.names.split(",")
        assert names and set(names) <= set(NAMES[1:]), names
        run_selection(args.output, names)
