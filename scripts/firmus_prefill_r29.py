"""Byte-exact QKV-only tail split and its in-job measurements."""
# ruff: noqa: B023 - each generator is screened before advancing to its next shape
import hashlib
import json
import math
import os
import runpy
import statistics
import time
from pathlib import Path

SCHEMA = "firmus-prefill-qkv-tail-r29/v1"
SECONDS, ROUNDS, CALLS = 90, 10, 1200
SPECS = {
    "qkv-tail": {"key": "SGLANG_PREFILL_QKV_TAIL_SPLIT", "shapes": ("QKV",), "reps": 24, "margin_us": 2.0},
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
        assert time.time() < deadline, "r29 timing deadline"
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
        assert time.time() < deadline, "r29 endurance deadline"
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

    for projection, n, k in (("QKV", 9216, 4096),):
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
        manifest = read(Path("/opt/apac/fa3-verify/manifest.json"))
        result["source_sha256"] = hashlib.sha256(Path(prefill_fp8.__file__).read_bytes() + manifest["binary_sha256"].encode()).hexdigest()
        checks = runpy.run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
        assert name == "qkv-tail"
        for label, steps, inputs, set_version in gemm_cases(torch, checks):
            assert time.time() < deadline, "r29 worker deadline"
            result["shapes"][label] = screen(torch, steps, inputs, set_version, SPECS[name]["reps"], deadline)
        assert time.time() <= deadline, "r29 worker deadline after final shape"
        result["healthy"] = True
    except Exception as error:  # noqa: BLE001 - independent candidate refusal, keep evidence
        import traceback
        traceback.print_exc()
        result["reason"] = f"{type(error).__name__}: {error}"
    result.update(elapsed_seconds=time.monotonic() - begin, finished_at=time.time())
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "shapes"}), flush=True)
