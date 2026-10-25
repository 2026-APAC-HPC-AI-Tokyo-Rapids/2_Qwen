"""Select fixed Prefill on every assigned worker GPU before loading the model."""
import argparse
import json
import math
import os
import signal
import statistics
import time
from datetime import timedelta
from pathlib import Path
from runpy import run_path

LAYERS = 94
PAIRS = 20
LENGTHS = (3986, 4027, 4096)
QKNORM_PAIRS = 50
QKNORM_ID_TYPES = ("int32", "int64")
_table = Path(__file__).resolve().parents[1] / "kernels/fa3-verify/prefill-candidates.json"
if not _table.is_file():
    _table = Path(__file__).resolve().parents[1] / "source/prefill-candidates.json"
if not _table.is_file():
    _table = Path("/opt/apac/fa3-verify/prefill-candidates.json")
CANDIDATES = json.loads(_table.read_text())
NAMES = tuple(c["name"] for c in CANDIDATES)


def choose_candidate(ranks, expected_ranks=4):
    assert len(ranks) == expected_ranks and {r["rank"] for r in ranks} == set(range(expected_ranks))
    choices = {}
    for name in NAMES[1:]:
        entries = [r["lengths"][str(length)][name] for r in ranks for length in LENGTHS]
        if any(not e.get("numeric_passed") or e.get("error") or len(e.get("control", [])) != PAIRS or len(e.get("candidate", [])) != PAIRS or any(not math.isfinite(x) or x <= 0 for x in e.get("control", []) + e.get("candidate", [])) for e in entries):
            choices[name] = {"eligible": False, "reason": "gate-or-timing-failed"}
            continue
        savings = [statistics.median(e["control"]) - statistics.median(e["candidate"]) for e in entries]
        medians = [statistics.median(e["candidate"]) for e in entries]
        choices[name] = {"eligible": min(savings) >= 5.0, "savings_us_by_rank_length": savings, "minimum_saving_us": min(savings), "median_us": statistics.median(medians)}
    eligible = [n for n, e in choices.items() if e["eligible"]]
    winner = min(eligible, key=lambda n: (choices[n]["median_us"], NAMES.index(n))) if eligible else "control"
    return {"winner": winner, "choices": choices, "env": {"SGLANG_PREFILL_FA3_VERIFY": winner}, "rule": {"all_ranks_and_lengths": True, "minimum_saving_us_per_layer": 5.0, "lengths": LENGTHS, "gpus": expected_ranks}}


def measure(torch, graph):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / LAYERS


def choose_qknorm(ranks, expected_ranks=4):
    """Fail closed, including missing, malformed or nonfinite timing fragments."""
    result = {"winner": "control", "rule": {"all_ranks_lengths_and_id_types": True, "minimum_saving_us_per_call": 8.0, "pairs": QKNORM_PAIRS, "lengths": LENGTHS, "id_types": QKNORM_ID_TYPES, "gpus": expected_ranks}}
    try:
        assert len(ranks) == expected_ranks and {r["rank"] for r in ranks} == set(range(expected_ranks))
        savings = []
        for rank in ranks:
            assert not rank["qknorm"].get("error")
            for length in LENGTHS:
                for id_type in QKNORM_ID_TYPES:
                    entry = rank["qknorm"]["lengths"][str(length)][id_type]
                    assert entry["bitwise_passed"] is True and not entry.get("error")
                    control, wide = entry["control"], entry["wide"]
                    assert len(control) == len(wide) == QKNORM_PAIRS
                    assert all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in control + wide)
                    savings.append(statistics.median(control) - statistics.median(wide))
        result.update(savings_us_by_rank_length_id_type=savings, minimum_saving_us=min(savings))
        if min(savings) >= 8.0:
            result["winner"] = "wide"
    except Exception as error:  # noqa: BLE001 - never select on incomplete evidence
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def qknorm_selection(torch):
    """Use the already-compiled serving module; add at most a 14-second screen.

    No distributed collectives inside the deadline: each rank can fail closed
    independently, then the existing final all_gather collects all four gates.
    """
    from sglang.kernels.ops.attention.rope import apply_qknorm_rope_fp8_prefill

    begin = time.monotonic()
    result = {"lengths": {}}
    previous = os.environ.get("SGLANG_PREFILL_QKNORM")
    old_handler = signal.getsignal(signal.SIGALRM)

    def deadline(_signum, _frame):
        raise TimeoutError("QK-norm screen exceeded 14 seconds")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 14)
    try:
        # Reproduce serving's packed QKV strides. Large weights and V values
        # exercise FP8 overflow; injected BF16 bits include both signed zeros,
        # both signed subnormals, finite extremes and the FP8 boundary.
        torch.manual_seed(9261026)
        angles = torch.arange(8192, device="cuda").float()[:, None] * (1000000.0 ** (-torch.arange(0, 128, 2, device="cuda").float() / 128))
        trig = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
        qw = torch.ones(128, device="cuda", dtype=torch.bfloat16)
        kw = torch.full((128,), .75, device="cuda", dtype=torch.bfloat16)
        qw[::4], qw[1::4] = 512, -2048
        kw[::4], kw[1::4] = -1024, 256
        special = torch.tensor([0, -32768, 1, -32767, 127, -32641, 0x43e8, -15384, 0x5d80, -8832], device="cuda", dtype=torch.int16)
        for length in LENGTHS:
            packed = torch.randn(length, 72, 128, device="cuda", dtype=torch.bfloat16) * 512
            bits = packed.view(torch.int16).view(-1)
            count = bits[::257].numel()
            bits[::257] = special.repeat((count + special.numel() - 1) // special.numel())[:count]
            q, k, v = packed[:, :64], packed[:, 64:68], packed[:, 68:]
            # Unique destinations avoid store races. Slot zero and untouched
            # sentinel-filled slots are compared along with every written row.
            loc64 = torch.randperm(8192, device="cuda")[:length] + 1
            loc64[0] = 0
            pos64 = torch.randint(8192, (length,), device="cuda", dtype=torch.int64)
            pos64[0], pos64[-1] = 0, 8191
            outputs = {}
            for name in ("control", "wide"):
                shapes = ((length, 64, 128), (8193, 4, 128), (8193, 4, 128), (length, 4, 128), (length, 4, 128))
                outputs[name] = [torch.full(shape, 0xa5, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn) for shape in shapes]
            result["lengths"][str(length)] = {}
            for id_type in QKNORM_ID_TYPES:
                dtype = getattr(torch, id_type)
                pos, loc = pos64.to(dtype), loc64.to(dtype)
                graphs = {}
                for name in ("control", "wide"):
                    os.environ["SGLANG_PREFILL_QKNORM"] = name
                    out, kc, vc, ck, cv = outputs[name]
                    args = (q, k, qw, kw, trig, pos, 1e-6, out, v, kc, vc, loc, ck, cv)
                    for tensor in outputs[name]:
                        tensor.view(torch.uint8).fill_(0xa5)
                    apply_qknorm_rope_fp8_prefill(*args)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    # Amortize Python replay overhead without including other
                    # kernels or changing the prologue's memory footprint.
                    with torch.cuda.graph(graph):
                        for _ in range(8):
                            apply_qknorm_rope_fp8_prefill(*args)
                    graphs[name] = graph
                assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(outputs["control"], outputs["wide"])), f"byte mismatch at {length}/{id_type}"
                entry = {"bitwise_passed": True, "control": [], "wide": []}
                for graph in graphs.values():
                    for _ in range(3):
                        graph.replay()
                torch.cuda.synchronize()
                for pair in range(QKNORM_PAIRS):
                    order = ("control", "wide") if pair % 2 == 0 else ("wide", "control")
                    for name in order:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        graphs[name].replay()
                        end.record()
                        end.synchronize()
                        entry[name].append(start.elapsed_time(end) * 1000 / 8)
                result["lengths"][str(length)][id_type] = entry
                del graphs
            del outputs, packed, q, k, v
    except Exception as error:  # noqa: BLE001 - any gate/timing failure retains control
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if previous is None:
            os.environ.pop("SGLANG_PREFILL_QKNORM", None)
        else:
            os.environ["SGLANG_PREFILL_QKNORM"] = previous
    result["elapsed_seconds"] = time.monotonic() - begin
    if result["elapsed_seconds"] >= 15:
        result["error"] = "QK-norm screen exceeded 15-second total budget"
    return result


def run_selection(output):
    begin = time.monotonic()
    import torch
    import torch.distributed as dist
    from sglang.srt.layers.attention.decode_fa3_verify import load

    rank = int(os.environ["LOCAL_RANK"])
    # FA3 candidates must see the unchanged control prologue, independent of
    # inherited environment. QK-norm is measured only after FA3 finishes.
    os.environ["SGLANG_PREFILL_QKNORM"] = "control"
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
    for length in LENGTHS:
        common = checks["prefill_selection_inputs"](length)
        original_inputs = common["packed"].clone()
        control = checks["prefill_selection_step"](common, "control", LAYERS)
        control_graph = checks["prefill_selection_capture"](control)
        local["lengths"][str(length)] = {}
        for name in NAMES[1:]:
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
            local["lengths"][str(length)][name] = result
            print(json.dumps({"rank": rank, "length": length, "name": name, **result}), flush=True)
            del graph, step
        del common, control, control_graph, original_inputs
    local["qknorm"] = qknorm_selection(torch)
    print(json.dumps({"rank": rank, "qknorm": local["qknorm"]}), flush=True)
    local["elapsed_seconds"] = time.monotonic() - begin
    dist.all_gather_object(gathered, local)
    if rank == 0:
        result = {"schema": "firmus-prefill-fa3-selection/v1", "ranks": gathered, "layers": LAYERS, "pairs": PAIRS, "includes_qkv_producer": True, "elapsed_seconds": max(r["elapsed_seconds"] for r in gathered)}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    dist.destroy_process_group()


def aggregate_selection(directory, output, env_output):
    """All twelve Prefill GPUs must pass; an incomplete worker retains control."""
    ranks = []
    failed = []
    for worker in range(3):
        path = directory / f"prefill-{worker}"
        try:
            assert (path / "status.txt").read_text().strip() == "fa3=0"
            fragment = json.loads((path / "fa3.json").read_text())
            entries = fragment["ranks"]
            assert len(entries) == 4 and {r["rank"] for r in entries} == set(range(4))
            ranks.extend({**entry, "local_rank": entry["rank"], "rank": worker*4 + entry["rank"], "worker": worker} for entry in entries)
        except Exception:  # noqa: BLE001 - failed or incomplete workers retain control
            failed.append(worker)
    if failed:
        selection = {"winner": "control", "env": {"SGLANG_PREFILL_FA3_VERIFY": "control"}, "failed_workers": failed}
    else:
        try:
            selection = choose_candidate(ranks, expected_ranks=12)
        except Exception as error:  # noqa: BLE001 - malformed timing retains control
            selection = {"winner": "control", "env": {"SGLANG_PREFILL_FA3_VERIFY": "control"}, "error": f"{type(error).__name__}: {error}"}
    selection["qknorm"] = choose_qknorm(ranks, expected_ranks=12) if not failed else {"winner": "control", "failed_workers": failed}
    selection["env"]["SGLANG_PREFILL_QKNORM"] = selection["qknorm"]["winner"]
    result = {"schema": "firmus-prefill-fa3-selection-all-workers/v1", **selection, "fragments": [f"prefill-{i}/fa3.json" for i in range(3)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    env_output.write_text("".join(f"{k}={v}\n" for k, v in result["env"].items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-output", type=Path)
    parser.add_argument("--aggregate", type=Path)
    args = parser.parse_args()
    if args.aggregate:
        assert args.env_output is not None
        aggregate_selection(args.aggregate, args.output, args.env_output)
    else:
        run_selection(args.output)
