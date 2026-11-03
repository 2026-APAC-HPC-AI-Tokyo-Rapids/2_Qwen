"""Strict post-P gate for the r26 Prefill swap tail, in one resident process/GPU.

Snapshots read ordinary kernel buffers after completion. Timing and endurance
use the ordinary uninstrumented serving API. No diagnostic kernel is compiled.
"""
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

SCHEMA = "firmus-prefill-megamoe-swap-tail/v1"
KEY = "DG_SM90_PREFILL_SWAP_TAIL"
CONTROL = "four-stage-lean-split"
CALLS, ROUNDS, SECONDS = 1200, 12, 120
TIMED = ("balanced", "hot", "historical")
CASES = (*TIMED, "edges-low", "edges-high", "edges-pool", "zero", "wide", *(f"saved-{i}" for i in range(36)))
COMPONENTS = ("output", "fc1_fp8", "fc1_scale", "fc2_bf16_partial")
MARGIN_US, MARGIN_FRACTION = 5.0, 0.005


def reject_constant(value):
    raise ValueError(f"non-finite JSON: {value}")


def read(path):
    return json.loads(path.read_text(), parse_constant=reject_constant)


def margin(samples):
    result = {}
    for order in ("forward", "reverse"):
        row = samples[order]
        for values in row.values():
            assert len(values) == ROUNDS and all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in values), "missing timing"
        c, t = statistics.median(row["control"]), statistics.median(row["candidate"])
        assert c - t >= max(MARGIN_US, MARGIN_FRACTION * c), "below selector margin"
        result[order] = c - t
    return result


def aggregate(directory, env_output, pair_selection, run_id, deadline, workers_ok, image):
    result = {"schema": SCHEMA, "adopted": False, "run_id": run_id, "gpus": [], "rule": {"control": CONTROL, "calls": CALLS, "rounds_per_order": ROUNDS, "minimum_us": MARGIN_US, "minimum_fraction": MARGIN_FRACTION, "seconds": SECONDS, "components": COMPONENTS, "cases": CASES}}
    try:
        assert workers_ok and time.time() <= deadline, "failed workers or expired budget"
        prior = read(pair_selection)
        assert prior["adopted"] is True, "P prerequisite missing"
        identities = {g["uuid"] for g in prior["gpus"]}
        assert len(identities) == len(prior["gpus"]) == 12, "incomplete prerequisite UUIDs"
        assert all(g["run_id"].split("-", 1)[0] == run_id.split("-", 1)[0] for g in prior["gpus"]), "wrong allocation"
        uuids, headers = set(), set()
        for worker in range(3):
            for rank in range(4):
                r = read(directory / f"prefill-{worker}" / f"rank-{rank}.json")
                result["gpus"].append(r)
                assert r["schema"] == SCHEMA and r["run_id"] == run_id and r["rank"] == rank, "stale rank"
                assert r["worker"] == f"prefill-{worker}" and r["uuid"] in identities and r["uuid"] not in uuids and "H200" in r["device"], "wrong GPU"
                assert r["image_sha256"] == image and r["healthy"] is True, "wrong image or failed worker"
                assert 0 < r["elapsed_seconds"] < SECONDS and r["finished_at"] <= deadline, "late worker"
                assert len(r["header_sha256"]) == 64 and len(r["routing_sha256"]) == 64, "missing provenance"
                headers.add((r["header_sha256"], r["routing_sha256"]))
                uuids.add(r["uuid"])
                assert set(r["exact"]) == set(CASES), "missing cases"
                assert all(set(v) == set(COMPONENTS) and all(x is True for x in v.values()) for v in r["exact"].values()), "byte mismatch"
                assert set(r["timing"]) == set(TIMED), "missing timing routing"
                for samples in r["timing"].values():
                    margin(samples)
                e = r["endurance"]
                assert e["calls"] == CALLS and e["buffers"] == 2 and e["input_versions"] == 2, "incomplete endurance"
                assert set(e["outputs"]) == set(TIMED + ("edges-low", "zero", "wide")) and all(v is True for v in e["outputs"].values()), "endurance mismatch"
                assert len(e["restored"]) == 2 and all(set(v) == set(COMPONENTS) and all(x is True for x in v.values()) for v in e["restored"]), "restore mismatch"
        assert uuids == identities and len(headers) == 1, "incomplete GPUs or inconsistent source"
        lines = env_output.read_text().splitlines()
        for key in ("DG_SM90_PREFILL_4STAGE", "DG_SM90_PREFILL_LEAN", "DG_SM90_PREFILL_SPLIT_ACCUM"):
            assert [v for v in lines if v.startswith(key + "=")] == [key + "=1"], "wrong serving control"
        assert all(not line.startswith(KEY + "=") or line == KEY + "=0" for line in lines), "unexpected swap pin"
        assert time.time() <= deadline, "aggregate deadline expired"
        temporary = env_output.with_name(env_output.name + ".swap-tail.tmp")
        try:
            temporary.write_text("\n".join([line for line in lines if not line.startswith(KEY + "=")] + [KEY + "=1"]) + "\n")
            temporary.replace(env_output)
        finally:
            temporary.unlink(missing_ok=True)
        result.update(adopted=True, reason="all twelve UUIDs: all bytes, 1200 calls, both order medians exceed prior selector margin")
    except (OSError, AssertionError, KeyError, ValueError, TypeError) as error:
        result["reason"] = f"{type(error).__name__}: {error}"
    return result


def saved_routings(path):
    import numpy as np

    ids = np.load(path)["logical_ids"]
    assert ids.shape == (2, 3, 4, 4001, 8)
    cases = []
    for sample in range(2):
        for layer in range(3):
            for group in ((0,), (1,), (2,), (3,), (0, 1), (2, 3)):
                cases.append(np.concatenate([ids[sample, layer, rank] for rank in group]).astype(np.int32))
    historical = np.concatenate([ids[s, 0, 0, :3994] for s in range(2)]).astype(np.int32)
    assert all((v >= 0).all() for v in [historical, *cases])
    return historical, cases


def canonical_indices(torch, buf, ids):
    """Pool order is nondeterministic: restore source token/top-k order from metadata."""
    counts = torch.bincount(ids.flatten().long(), minlength=128).cpu().tolist()
    indices, base = [], 0
    for count in counts:
        indices.extend(range(base, base + count))
        base += ((count + 127) // 128) * 128
    indices = torch.tensor(indices, device=ids.device, dtype=torch.long)
    pool = buf.l1_acts.shape[0]
    blocks = pool // 8
    # SM90Workspace, from layout/mega_moe.cuh; view only, not instrumentation.
    metadata_offset = 32 + 128 * 8 * 3 + ((blocks + 1) // 2 * 2) * 4 + blocks * 8 + 128 * buf.num_max_tokens_per_rank * 4
    meta = buf.buffer[metadata_offset:metadata_offset + pool * 12].view(torch.int32).reshape(pool, 3)[indices].long()
    keys = meta[:, 1] * 8 + meta[:, 2]
    order = keys.argsort()
    assert torch.equal(keys[order], torch.arange(ids.numel(), device=ids.device)), "metadata not a permutation"
    assert bool((meta[:, 0] == 0).all().item()), "not EP1 metadata"
    return indices[order]


def snapshot(torch, buf, ids, out):
    torch.cuda.synchronize()
    indices = canonical_indices(torch, buf, ids)
    partial_bytes = 8 * buf.num_max_tokens_per_rank * 4096 * 2
    partial = buf.buffer[-partial_bytes:].view(torch.uint8).reshape(8, buf.num_max_tokens_per_rank, 8192)[:, :ids.shape[0]]
    return {"output": out.view(torch.uint8).clone(), "fc1_fp8": buf.l2_acts.view(torch.uint8)[indices].clone(), "fc1_scale": buf.l2_acts_sf[indices, :12].contiguous().view(torch.uint8).clone(), "fc2_bf16_partial": partial.clone()}


def equal(torch, left, right):
    return {k: bool(torch.equal(left[k], right[k])) for k in COMPONENTS}


def run(prepare, output, routing_path, run_id, deadline, image):
    begin = time.monotonic()
    import deep_gemm

    historical, saved = saved_routings(routing_path)
    torch, dist, rank, device, _, _, generator, _, _, buf, forward = prepare(1, tokens=7988)
    other = deep_gemm.get_symm_buffer_for_mega_moe(buf.group, 128, 8192, 8, 4096, 1536, use_fp8_dispatch=True, activation="swiglu")
    buffers = (buf, other)
    import importlib.util

    helper = Path(importlib.util.find_spec("deep_gemm").origin).parent / "include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"
    result = {"schema": SCHEMA, "run_id": run_id, "rank": rank, "worker": output.name, "uuid": str(torch.cuda.get_device_properties(rank).uuid), "device": torch.cuda.get_device_name(rank), "image_sha256": image, "header_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(), "routing_sha256": hashlib.sha256(routing_path.read_bytes()).hexdigest(), "exact": {}, "timing": {}, "healthy": False}

    def call(mode, case, out, buffer=buf, start=None):
        os.environ[KEY] = "1" if mode == "candidate" else "0"
        forward(CONTROL, *case, out, start=start, buffer=buffer)

    def make_case(name):
        count = len(saved[int(name[6:])]) if name.startswith("saved-") else 7988
        hidden = torch.randn(count, 4096, dtype=torch.bfloat16, device=device, generator=generator) * (10.0 if name == "wide" else 0.25)
        if name == "zero":
            hidden.zero_()
        if name.startswith("saved-") or name == "historical":
            ids = torch.tensor(saved[int(name[6:])] if name.startswith("saved-") else historical, device=device, dtype=torch.int32)
        else:
            scores = torch.rand(count, 128, device=device, generator=generator)
            if name == "hot":
                scores[:, 0] += 1.0
            if name.startswith("edges-"):
                scores[:, :64] = -1
            ids = scores.topk(8, -1).indices.to(torch.int32)
            if name.startswith("edges-"):
                sizes = range(1, 65) if name == "edges-low" else range(65, 129) if name == "edges-high" else (129, 191, 192, 193, 255, 256)
                row = 0
                for expert, size in enumerate(sizes):
                    ids[row:row + size, -1] = expert
                    row += size
                assert row <= count
        weights = torch.rand(count, 8, device=device, generator=generator)
        weights /= weights.sum(-1, keepdim=True)
        return hidden, ids, weights

    cached = {}
    # Byte checks finish before timing, with no copies or probes inside timed spans.
    for name in CASES:
        assert time.time() < deadline, "worker budget expired"
        case = make_case(name)
        out = torch.empty_like(case[0])
        call("control", case, out)
        expected = snapshot(torch, buf, case[1], out)
        call("candidate", case, out)
        result["exact"][name] = equal(torch, expected, snapshot(torch, buf, case[1], out))
        assert all(result["exact"][name].values()), f"byte mismatch: {name}"
        if name == "historical":
            restoration = expected
        del expected
        if name in TIMED + ("edges-low", "zero", "wide"):
            cached[name] = case
    for name in TIMED:
        case = cached[name]
        out = torch.empty_like(case[0])
        samples = {order: {m: [] for m in ("control", "candidate")} for order in ("forward", "reverse")}
        for i in range(ROUNDS):
            for order, modes in (("forward", ("control", "candidate")), ("reverse", ("candidate", "control"))):
                for mode in modes:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda._sleep(1_000_000)
                    call(mode, case, out, start=start)
                    end.record()
                    end.synchronize()
                    samples[order][mode].append(start.elapsed_time(end) * 1000)
        result["timing"][name] = samples
    kinds = tuple(cached)
    # Two input versions and two buffers; no synchronization between 1200 calls.
    changed = {name: (-v[0], v[1].roll(1, 0), v[2].roll(1, 0)) for name, v in cached.items()}
    reference, observed = {}, {}
    for name in kinds:
        out = torch.empty_like(cached[name][0])
        call("control", changed[name], out)
        reference[name] = out.clone()
        observed[name] = torch.empty_like(out)
    torch.cuda.synchronize()
    for i in range(CALLS):
        name = kinds[i % len(kinds)]
        call("candidate", changed[name] if i >= CALLS // 2 else cached[name], observed[name], buffer=buffers[(i // len(kinds)) % 2])
    torch.cuda.synchronize()
    endurance = {"calls": CALLS, "buffers": 2, "input_versions": 2, "outputs": {name: bool(torch.equal(reference[name].view(torch.uint8), observed[name].view(torch.uint8))) for name in kinds}, "restored": []}
    case = cached["historical"]
    out = torch.empty_like(case[0])
    for buffer in buffers:
        call("candidate", case, out, buffer=buffer)
        endurance["restored"].append(equal(torch, restoration, snapshot(torch, buffer, case[1], out)))
    result["endurance"] = endurance
    result.update(healthy=True, elapsed_seconds=time.monotonic() - begin, finished_at=time.time())
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("exact", "timing")}), flush=True)
    for buffer in buffers:
        buffer.destroy()
    dist.destroy_process_group()
