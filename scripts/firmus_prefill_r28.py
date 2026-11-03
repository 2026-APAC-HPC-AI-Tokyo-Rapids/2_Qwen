"""Byte-exact small Prefill candidates and their common in-job measurements."""
# ruff: noqa: B023 - each generator is screened before advancing to its next shape
import hashlib
import importlib
import json
import math
import os
import runpy
import statistics
import time
from pathlib import Path

SCHEMA = "firmus-prefill-small-r28/v1"
SECONDS, ROUNDS, CALLS = 90, 10, 1200
PAIR_SHAPES = ((3994, 3994), (3986, 4027), (3986, 3986), (4027, 4027), (4096, 4096), (3900, 4096), (4096, 3901))
SPECS = {
    "gemm-tail": {"key": "SGLANG_PREFILL_GEMM_TAIL_SPLIT", "shapes": ("QKV", "O"), "reps": 24, "margin_us": 2.0},
    "fa3-pair-order": {"key": "SGLANG_PREFILL_FA3_PAIR_GLOBAL_ORDER", "shapes": tuple("+".join(map(str, s)) for s in PAIR_SHAPES), "reps": 12, "margin_us": 1.0},
    "norm-store": {"key": "SGLANG_PREFILL_NORM_STORE_PAIR", "shapes": ("7988", "7989", "3994", "2049"), "reps": 8, "margin_us": 0.5},
}


def read(path):
    def bad(value):
        raise ValueError("nonfinite JSON: " + value)

    def number(value):
        result = float(value)
        assert math.isfinite(result), "nonfinite JSON number"
        return result

    return json.loads(path.read_text(), parse_constant=bad, parse_float=number)


def validate(record, name):
    spec = SPECS[name]
    assert record["healthy"] is True
    assert set(record["shapes"]) == set(spec["shapes"]), "missing shape"
    for shape in record["shapes"].values():
        assert shape["bytes_equal"] is True and shape["input_versions"] == 2 and shape["restored"] is True, "byte or restoration mismatch"
        e = shape["endurance"]
        assert e == {"calls": CALLS, "buffers": 2, "input_versions": 2, "bytes_equal": True, "restored": True}, "incomplete endurance"
        for order in ("forward", "reverse"):
            row = shape["samples_us"][order]
            assert set(row) == {"control", "candidate"}
            for values in row.values():
                assert len(values) == ROUNDS and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values), "invalid timing"
            c, t = (statistics.median(row[m]) for m in ("control", "candidate"))
            assert c - t >= max(spec["margin_us"], .005 * c), "below both-order timing margin"


def exact(torch, left, right):
    return len(left) == len(right) and all(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)) for a, b in zip(left, right, strict=True))


def screen(torch, steps, inputs, set_version, reps, deadline):
    """Closures own fixed output buffers. Reset/copy operations are outside timing."""
    expected, byte_ok = [], True
    for version in (0, 1, 0):
        set_version(version)
        control = tuple(t.clone() for t in steps["control"][0]())
        for step in steps["candidate"]:
            byte_ok &= exact(torch, step(), control)
        if len(expected) < 2:
            expected.append(control)
        else:
            byte_ok &= exact(torch, control, expected[0])
    assert byte_ok, "candidate differs from control bytes"
    graphs = {}
    for mode in ("control", "candidate"):
        step = steps[mode][0]
        step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(reps):
                step()
        graphs[mode] = graph
        graph.replay()
    samples = {order: {mode: [] for mode in graphs} for order in ("forward", "reverse")}
    for _ in range(ROUNDS):
        assert time.time() < deadline, "r28 timing deadline"
        for order, modes in (("forward", ("control", "candidate")), ("reverse", ("candidate", "control"))):
            for mode in modes:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); graphs[mode].replay(); end.record(); end.synchronize()
                samples[order][mode].append(start.elapsed_time(end) * 1000 / reps)
    for version in (1, 0):
        set_version(version)
        for mode, graph in graphs.items():
            graph.replay()
            torch.cuda.synchronize()
            assert exact(torch, steps[mode][0].output, expected[version]), "changed-input graph replay"
    del graphs
    # Ordinary launches with two buffers. Check every version transition, then
    # restore version zero against its CONTROL reference, not candidate output.
    for version in (0, 1, 0, 1):
        assert time.time() < deadline, "r28 endurance deadline"
        set_version(version)
        for i in range(CALLS // 4):
            steps["candidate"][i % 2]()
        torch.cuda.synchronize()
        assert all(exact(torch, step.output, expected[version]) for step in steps["candidate"]), "endurance bytes"
    set_version(0)
    restored = all(exact(torch, step(), expected[0]) for step in steps["candidate"])
    assert restored and inputs is not None
    return {"bytes_equal": True, "input_versions": 2, "restored": restored, "samples_us": samples, "endurance": {"calls": CALLS, "buffers": 2, "input_versions": 2, "bytes_equal": True, "restored": restored}}


def gemm_cases(torch, checks):
    from sglang.srt.layers.attention.prefill_fp8 import dense_gemm

    for projection, n, k in (("QKV", 9216, 4096), ("O", 4096, 8192)):
        lhs, rhs, out = checks["dense_inputs"](7988, n, k, 2828 + k)
        a, sa = lhs
        versions = (a.clone(), (-a.float()).to(a.dtype))
        outputs = [out, torch.empty_like(out), torch.empty_like(out)]
        steps = {}
        for mode, indices in (("control", (0,)), ("candidate", (1, 2))):
            steps[mode] = []
            for index in indices:
                def call(index=index, mode=mode):
                    dense_gemm(a, sa, *rhs, outputs[index], tail_split=mode == "candidate")
                    return (outputs[index],)
                call.output = (outputs[index],)
                steps[mode].append(call)
        yield projection, steps, (lhs, rhs, versions, outputs), lambda v: a.copy_(versions[v])


def norm_cases(torch, checks):
    nq = importlib.import_module("sglang.kernels.ops.layernorm.add_rmsnorm_group_quant")
    for rows in map(int, SPECS["norm-store"]["shapes"]):
        x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        w = (torch.randn(4096, device="cuda") * .1 + 1).bfloat16()
        versions = (x.clone(), -x)
        outputs = [(torch.empty_like(x), torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((32, (rows + 3) // 4 * 4), device="cuda").T[:rows]) for _ in range(3)]
        steps = {}
        for mode, indices in (("control", (0,)), ("candidate", (1, 2))):
            steps[mode] = []
            for index in indices:
                def call(index=index, mode=mode):
                    r, q, s = outputs[index]
                    r.copy_(residual)
                    nq.norm_quant[(rows,)](x, r, w, q, s, s.stride(1), 1e-6, STORE_PAIR=mode == "candidate", num_warps=8, enable_fp_fusion=False)
                    return r, q, s
                call.output = outputs[index]
                steps[mode].append(call)
        yield str(rows), steps, (x, residual, w, versions, outputs), lambda v: x.copy_(versions[v])


def fa3_cases(torch, checks, pin):
    from sglang.srt.layers.attention.decode_fa3_verify import load
    from sglang.srt.layers.attention.prefill_fp8 import (
        pair_candidate,
        prefill_candidate,
    )
    load()
    for lengths in PAIR_SHAPES:
        common = checks["prefill_selection_inputs"](lengths, padding=0)
        common["fp32_producer"] = True
        os.environ["SGLANG_PREFILL_FA3_VERIFY"] = pin
        os.environ[SPECS["fa3-pair-order"]["key"]] = "0"
        prefill_candidate.cache_clear()
        control = pair_candidate(lengths)
        assert control["block_n"] == 96, "current serving selection is not a pair"
        candidate = prefill_candidate(control["name"] + "-global-order")
        assert candidate["variant"] == control["variant"] | 1048576
        from sglang.kernels.ops.attention.rope import apply_qknorm_rope_fp8_prefill_fp32
        apply_qknorm_rope_fp8_prefill_fp32(*common["rope_args"])
        image = None
        if control.get("native_v_n"):
            image = torch.empty((2, 4, (max(lengths) + 95) // 96, 96 * 128), device="cuda", dtype=torch.uint8)
        empty = torch.empty(0, device="cuda", dtype=torch.uint8)
        q = common["q"]
        versions = (q.clone(), (-q.float()).to(q.dtype))
        rows = common["rows"]
        outputs = [(torch.empty_like(q), torch.empty((64, (rows + 3) // 4 * 4), device="cuda").T[:rows], torch.empty((64, rows), device="cuda"), torch.empty(17, device="cuda", dtype=torch.int32)) for _ in range(3)]
        steps = {}
        for mode, indices in (("control", (0,)), ("candidate", (1, 2))):
            spec = control if mode == "control" else candidate
            steps[mode] = []
            for index in indices:
                def call(index=index, spec=spec):
                    o, s, lse, metadata = outputs[index]
                    torch.ops.apac_fa3_verify.prefill_variant_fwd_out(q, common["k"], common["v"], image if spec.get("native_v_n") else empty, common["cu"], o, s, lse, metadata, max(lengths), 128**-.5, spec["id"])
                    return o, s
                call.output = outputs[index][:2]
                steps[mode].append(call)
        def set_version(v):
            q.copy_(versions[v])
            if image is not None:
                torch.ops.apac_fa3_verify.prefill_prepack(common["v"], image, common["cu"])
        yield "+".join(map(str, lengths)), steps, (common, image, outputs, versions), set_version


def run(name, output, run_id, deadline, image, pin):
    begin = time.monotonic()
    import torch

    torch.set_num_threads(1)
    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 4
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.manual_seed(2828 + rank)
    result = {"schema": SCHEMA, "candidate": name, "run_id": run_id, "rank": rank, "worker": output.name, "uuid": str(torch.cuda.get_device_properties(rank).uuid), "device": torch.cuda.get_device_name(rank), "image_sha256": image, "fa3_pin": pin, "shapes": {}, "healthy": False}
    try:
        from sglang.srt.layers.attention import prefill_fp8
        nq = importlib.import_module("sglang.kernels.ops.layernorm.add_rmsnorm_group_quant")
        manifest = read(Path("/opt/apac/fa3-verify/manifest.json"))
        result["source_sha256"] = hashlib.sha256(Path(prefill_fp8.__file__).read_bytes() + Path(nq.__file__).read_bytes() + manifest["binary_sha256"].encode()).hexdigest()
        checks = runpy.run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
        cases = {"gemm-tail": gemm_cases, "norm-store": norm_cases, "fa3-pair-order": lambda t, c: fa3_cases(t, c, pin)}
        for label, steps, inputs, set_version in cases[name](torch, checks):
            assert time.time() < deadline, "r28 worker deadline"
            result["shapes"][label] = screen(torch, steps, inputs, set_version, SPECS[name]["reps"], deadline)
        assert time.time() <= deadline, "r28 worker deadline after final shape"
        result["healthy"] = True
    except Exception as error:  # noqa: BLE001 - independent candidate refusal, keep evidence
        import traceback
        traceback.print_exc()
        result["reason"] = f"{type(error).__name__}: {error}"
    result.update(elapsed_seconds=time.monotonic() - begin, finished_at=time.time())
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "shapes"}), flush=True)
