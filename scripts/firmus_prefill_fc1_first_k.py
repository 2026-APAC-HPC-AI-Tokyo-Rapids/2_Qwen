"""Post-greedy gate for the FC1 first-K128, in one resident process/GPU.

Snapshots read ordinary kernel buffers after completion. Timing and endurance
use the ordinary uninstrumented serving API. No diagnostic kernel is compiled.
"""
import faulthandler
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

SCHEMA = "firmus-prefill-megamoe-fc1-first-k/v1"
KEY = "DG_SM90_PREFILL_FC1_FIRST_K"
CONTROL = "four-stage-lean-split"
CALLS, ROUNDS, SECONDS = 1200, 12, 95
AGGREGATE_SECONDS = 105
TIMED = ("balanced", "hot", "historical")
CASES = (*TIMED, "edges-low", "edges-high", "edges-pool", "zero", "wide", *(f"saved-{i}" for i in range(36)))
COMPONENTS = ("output", "fc1_fp8", "fc1_scale", "fc2_bf16_partial")
MARGIN_US, MARGIN_FRACTION = 5.0, 0.005
MAX_RELATIVE_L2, MAX_SCALED_ERROR, SAMPLED_TOKENS = 0.025, 0.08, 32
ENDURANCE_KINDS = TIMED + ("edges-low", "zero", "wide")
ENDURANCE_CELLS = tuple(f"{name}/{version}/{buffer}" for name in ENDURANCE_KINDS for version in range(2) for buffer in range(2))


def reject_constant(value):
    raise ValueError(f"non-finite JSON: {value}")


def read(path):
    return json.loads(path.read_text(), parse_constant=reject_constant)


def timing_medians(samples):
    result = {}
    assert set(samples) == {"forward", "reverse"}, "missing timing order"
    for order in ("forward", "reverse"):
        row = samples[order]
        assert set(row) == {"control", "candidate"}, "missing timing mode"
        for values in row.values():
            assert len(values) == ROUNDS and all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in values), "missing timing"
        result[order] = {mode: statistics.median(values) for mode, values in row.items()}
    result["pooled"] = {mode: statistics.median(samples["forward"][mode] + samples["reverse"][mode]) for mode in ("control", "candidate")}
    return result


def pooled_margin(timings):
    """Post-r32 rule: median of twelve per-GPU savings, never pool GPU samples."""
    assert len(timings) == 12, "incomplete timing GPUs"
    rows = [timing_medians(samples) for samples in timings]
    savings = {order: [r[order]["control"] - r[order]["candidate"] for r in rows] for order in ("forward", "reverse", "pooled")}
    controls = [r["pooled"]["control"] for r in rows]
    summary = {"control_median_us": statistics.median(controls), "saving_median_us": {order: statistics.median(values) for order, values in savings.items()}, "saving_median_fraction": statistics.median(v / c for v, c in zip(savings["pooled"], controls, strict=True)), "positive_gpus": sum(v > 0 for v in savings["pooled"]), "minimum_saving_us": min(savings["pooled"]), "minimum_saving_fraction": min(v / c for v, c in zip(savings["pooled"], controls, strict=True))}
    assert summary["saving_median_us"]["pooled"] >= MARGIN_US and summary["saving_median_fraction"] >= MARGIN_FRACTION, "below pooled selector margin"
    assert all(summary["saving_median_us"][order] > 0 for order in ("forward", "reverse")), "nonpositive order median"
    assert summary["positive_gpus"] >= 9, "fewer than nine positive GPUs"
    assert summary["minimum_saving_fraction"] >= -0.01, "GPU loss exceeds one percent"
    return summary


def check_reference(record):
    assert set(record) == {"control", "candidate"}, "missing reference mode"
    for row in record.values():
        assert row["samples"] == SAMPLED_TOKENS and row["finite"] is True, "invalid FP32 reference"
        for key, limit in (("relative_l2", MAX_RELATIVE_L2), ("max_scaled_error", MAX_SCALED_ERROR)):
            assert type(row[key]) in (int, float) and math.isfinite(row[key]) and 0 <= row[key] < limit, "FP32 reference threshold"


def aggregate(directory, env_output, pair_selection, run_id, deadline, worker_exit_codes, image):
    result = {"schema": SCHEMA, "adopted": False, "run_id": run_id, "worker_exit_codes": {f"prefill-{i}": code for i, code in enumerate(worker_exit_codes)}, "gpus": [], "rule": {"control": CONTROL, "calls": CALLS, "rounds_per_order": ROUNDS, "minimum_us": MARGIN_US, "minimum_fraction": MARGIN_FRACTION, "seconds": SECONDS, "components": COMPONENTS, "cases": CASES, "timing": "post-r32 pooled per-GPU medians", "minimum_positive_gpus": 9, "maximum_gpu_loss_fraction": 0.01}}
    try:
        assert time.time() <= deadline, "expired budget"
        worker_deadline = deadline - (AGGREGATE_SECONDS - SECONDS)
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
                assert 0 < r["elapsed_seconds"] < SECONDS and r["finished_at"] <= worker_deadline, "late worker"
                assert len(r["header_sha256"]) == 64 and len(r["routing_sha256"]) == 64, "missing provenance"
                headers.add((r["header_sha256"], r["routing_sha256"]))
                uuids.add(r["uuid"])
                assert set(r["exact"]) == set(CASES), "missing cases"
                assert all(set(v) == set(COMPONENTS) and all(x is True for x in v.values()) for v in r["exact"].values()), "byte mismatch"
                assert set(r["reference"]) == set(CASES), "missing FP32 reference case"
                for reference in r["reference"].values():
                    check_reference(reference)
                assert set(r["timing"]) == set(TIMED), "missing timing routing"
                e = r["endurance"]
                assert e["calls"] == CALLS and e["buffers"] == 2 and e["input_versions"] == 2, "incomplete endurance"
                assert set(e["checks"]) == {"600", "1200"}, "missing endurance checkpoint"
                assert all(set(v) == set(ENDURANCE_CELLS) and all(x is True for x in v.values()) for v in e["checks"].values()), "endurance mismatch"
                assert len(e["restored"]) == 2 and all(set(v) == set(COMPONENTS) and all(x is True for x in v.values()) for v in e["restored"]), "restore mismatch"
        assert uuids == identities and len(headers) == 1, "incomplete GPUs or inconsistent source"
        result["timing_summary"] = {name: pooled_margin([r["timing"][name] for r in result["gpus"]]) for name in TIMED}
        lines = env_output.read_text().splitlines()
        for key in ("DG_SM90_PREFILL_4STAGE", "DG_SM90_PREFILL_LEAN", "DG_SM90_PREFILL_SPLIT_ACCUM"):
            assert [v for v in lines if v.startswith(key + "=")] == [key + "=1"], "wrong serving control"
        assert all(not line.startswith(KEY + "=") or line == KEY + "=0" for line in lines), "unexpected first-K pin"
        assert time.time() <= deadline, "aggregate deadline expired"
        temporary = env_output.with_name(env_output.name + ".fc1-first-k.tmp")
        try:
            temporary.write_text("\n".join([line for line in lines if not line.startswith(KEY + "=")] + [KEY + "=1"]) + "\n")
            temporary.replace(env_output)
        finally:
            temporary.unlink(missing_ok=True)
        result.update(adopted=True, reason="all twelve UUIDs: all bytes, 1200 calls and health; every routing passes post-r32 pooled timing rule")
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


def run(prepare, output, routing_path, run_id, deadline, image, reference_rows, errors):
    begin = time.monotonic()
    import deep_gemm

    historical, saved = saved_routings(routing_path)
    torch, dist, rank, device, _, _, generator, _, raw_weights, buf, forward = prepare(1, tokens=7988)
    other = deep_gemm.get_symm_buffer_for_mega_moe(buf.group, 128, 8192, 8, 4096, 1536, use_fp8_dispatch=True, activation="swiglu")
    buffers = (buf, other)
    import importlib.util

    helper = Path(importlib.util.find_spec("deep_gemm").origin).parent / "include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"
    result = {"schema": SCHEMA, "run_id": run_id, "rank": rank, "worker": output.name, "uuid": str(torch.cuda.get_device_properties(rank).uuid), "device": torch.cuda.get_device_name(rank), "image_sha256": image, "header_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(), "routing_sha256": hashlib.sha256(routing_path.read_bytes()).hexdigest(), "exact": {}, "reference": {}, "timing": {}, "healthy": False}

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
        sampled = torch.linspace(0, case[0].shape[0] - 1, SAMPLED_TOKENS, device=device).long()
        fp32 = reference_rows(torch, *(v[sampled] for v in case), *raw_weights, 0).to(torch.bfloat16)
        ref = {}
        call("control", case, out)
        l2, scaled = errors(torch, out[sampled], fp32)
        ref["control"] = {"samples": SAMPLED_TOKENS, "finite": bool(torch.isfinite(out[sampled]).all().item()), "relative_l2": l2, "max_scaled_error": scaled}
        expected = snapshot(torch, buf, case[1], out)
        call("candidate", case, out)
        l2, scaled = errors(torch, out[sampled], fp32)
        ref["candidate"] = {"samples": SAMPLED_TOKENS, "finite": bool(torch.isfinite(out[sampled]).all().item()), "relative_l2": l2, "max_scaled_error": scaled}
        check_reference(ref)
        result["reference"][name] = ref
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
    kinds = ENDURANCE_KINDS
    # All 24 routing/input/buffer cells repeat fifty times; compare at 600 and 1200.
    changed = {name: (-v[0], v[1].roll(1, 0), v[2].roll(1, 0)) for name, v in cached.items()}
    reference, observed = {}, {}
    for name in kinds:
        for version in range(2):
            out = torch.empty_like(cached[name][0])
            call("control", changed[name] if version else cached[name], out)
            reference[name, version] = out.clone()
            for buffer in range(2):
                observed[name, version, buffer] = torch.empty_like(out)
    torch.cuda.synchronize()
    checks = {}
    for i in range(CALLS):
        name = kinds[i % len(kinds)]
        version, buffer = (i // len(kinds)) % 2, (i // (2 * len(kinds))) % 2
        call("candidate", changed[name] if version else cached[name], observed[name, version, buffer], buffer=buffers[buffer])
        if i + 1 in (600, 1200):
            torch.cuda.synchronize()
            checks[str(i + 1)] = {f"{n}/{v}/{b}": bool(torch.equal(reference[n, v].view(torch.uint8), observed[n, v, b].view(torch.uint8))) for n in kinds for v in range(2) for b in range(2)}
            assert all(checks[str(i + 1)].values()), "endurance checkpoint mismatch"
    endurance = {"calls": CALLS, "buffers": 2, "input_versions": 2, "checks": checks, "restored": []}
    case = cached["historical"]
    out = torch.empty_like(case[0])
    for buffer in buffers:
        call("candidate", case, out, buffer=buffer)
        endurance["restored"].append(equal(torch, restoration, snapshot(torch, buffer, case[1], out)))
    result["endurance"] = endurance
    finish(torch, dist, buffers, result, output, rank, begin)


def finish(torch, dist, buffers, result, output, rank, begin):
    """Release all symmetric views and the EP1 group before WORLD; publish only after cleanup."""
    torch.cuda.synchronize()
    dist.barrier()
    group = buffers[0].group
    for buffer in buffers:
        # sgl-deep-gemm 0.1.5 destroy() only clears x/x_sf: the remaining
        # DLPack views otherwise remain live across process-group shutdown.
        for name in ("topk_idx", "topk_weights", "l1_acts", "l1_acts_sf", "l2_acts", "l2_acts_sf"):
            setattr(buffer, name, None)
        buffer.destroy()
    dist.destroy_process_group(group)
    # A fast rank must not finalize WORLD while its peers still release EP1.
    dist.barrier()
    dist.destroy_process_group()
    result.update(healthy=True, elapsed_seconds=time.monotonic() - begin, finished_at=time.time())
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    # Temporary exit diagnostic; remove once the rank/agent exit delay is located.
    # dump_traceback_later includes all Python threads and defaults to stderr.
    faulthandler.dump_traceback_later(10, repeat=True)
    print(json.dumps({k: v for k, v in result.items() if k not in ("exact", "timing")}), flush=True)
