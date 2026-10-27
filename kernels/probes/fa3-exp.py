"""Bounded fixed-Prefill exponent experiment; no serving promotion by this probe."""

import argparse
import ctypes
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

# Declare the admission contract before measurements. Model quality requires a
# separate same-checkpoint evaluation; these gates admit only a kernel candidate.
CONTRACT = {
    "delta_relative_l2": .015,
    "delta_max_scaled": .08,
    "reference_relative_l2": .05,
    "reference_max_scaled": .25,
    "reference_l2_growth": 1.05,
    "reference_l2_allowance": .0005,
    "reference_max_growth": 1.10,
    "reference_max_allowance": .005,
    "minimum_speedup_fraction": .03,
    "minimum_saving_us": 5.0,
    "maximum_control_drift": .05,
}
TIMED_ROWS = (3986, 4027, 4096)


def accept_errors(delta, control, candidate):
    values = [v for record in (delta, control, candidate) for v in record.values()]
    return all(math.isfinite(v) for v in values) and (
        delta["relative_l2"] <= CONTRACT["delta_relative_l2"]
        and delta["max_scaled"] <= CONTRACT["delta_max_scaled"]
        and control["relative_l2"] <= CONTRACT["reference_relative_l2"]
        and candidate["relative_l2"] <= CONTRACT["reference_relative_l2"]
        and control["max_scaled"] <= CONTRACT["reference_max_scaled"]
        and candidate["max_scaled"] <= CONTRACT["reference_max_scaled"]
        and candidate["relative_l2"] <= control["relative_l2"] * CONTRACT["reference_l2_growth"] + CONTRACT["reference_l2_allowance"]
        and candidate["max_scaled"] <= control["max_scaled"] * CONTRACT["reference_max_growth"] + CONTRACT["reference_max_allowance"]
    )


def eligible(timing):
    return len(timing) == len(TIMED_ROWS) and {r["rows"] for r in timing} == set(TIMED_ROWS) and all(
        all(math.isfinite(r[k]) for k in ("ratio", "saving_us", "control_drift", "paired_ratio_p90"))
        and r["ratio"] <= 1 - CONTRACT["minimum_speedup_fraction"]
        and r["saving_us"] >= CONTRACT["minimum_saving_us"]
        and abs(r["control_drift"]) <= CONTRACT["maximum_control_drift"]
        and r["paired_ratio_p90"] < 1.0
        for r in timing
    )


def run(output):
    import deep_gemm
    import torch

    began = time.monotonic()
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261027)
    directory = Path(__file__).parent
    libs = {}
    for name in ("control", "poly"):
        lib = ctypes.CDLL(str(directory / f"fa3-exp-{name}.so"), mode=ctypes.RTLD_LOCAL)
        lib.prefill_run.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 9 + [ctypes.c_int, ctypes.c_void_p]
        lib.prefill_run.restype = ctypes.c_int
        assert lib.prefill_init() == 0, name
        libs[name] = lib
    torch.ops.load_library("/opt/apac/fa3-verify/libapac_fa3_verify.so")
    results = {"candidate_name": "poly", "candidate_eligible": False, "numerical_passed": False,
               "scope": "one GPU fixed Prefill kernel; model output quality not yet evaluated",
               "contract": CONTRACT, "numeric": [], "o_projection": [], "timing": []}
    output.mkdir(parents=True, exist_ok=True)

    def save():
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    def inputs(rows):
        return {"rows": rows, "q": torch.empty((rows, 64, 128), device="cuda", dtype=torch.float8_e4m3fn),
                "k": torch.empty((rows, 4, 128), device="cuda", dtype=torch.float8_e4m3fn),
                "v": torch.empty((rows, 4, 128), device="cuda", dtype=torch.float8_e4m3fn),
                "cu": torch.tensor([0, rows], device="cuda", dtype=torch.int32)}

    def fill(data, profile):
        amplitude = 3.0 if profile == "sharp" else 1.0
        for name in ("q", "k", "v"):
            target = data[name]
            target.copy_((torch.randn(target.shape, device="cuda", dtype=torch.float32) * (amplitude if name != "v" else 1.)).to(target.dtype))
        if profile == "flat":
            data["q"].zero_()
        if profile == "zero_v":
            data["v"].zero_()

    def buffers(rows):
        padded = (rows + 7) // 4 * 4
        return {"out": torch.full((rows + 4, 64, 128), 123., device="cuda", dtype=torch.bfloat16),
                "lse": torch.full((64, rows), float("nan"), device="cuda"),
                "metadata": torch.empty(17, device="cuda", dtype=torch.int32),
                "fp8": torch.full((padded, 64, 128), 3., device="cuda", dtype=torch.float8_e4m3fn),
                "scale": torch.full((64, padded), -123., device="cuda").T}

    def call(data, buf, name):
        rows = data["rows"]
        if name == "image":
            torch.ops.apac_fa3_verify.prefill_fixed_fwd_out(data["q"], data["k"], data["v"], data["cu"], buf["fp8"], buf["scale"], buf["lse"], buf["metadata"], rows, 128**-.5)
            return
        rc = libs[name].prefill_run(0, rows, 1, rows, *(data[k].data_ptr() for k in ("q", "k", "v", "cu")), *(buf[k].data_ptr() for k in ("out", "lse", "metadata", "fp8", "scale")), buf["scale"].stride(1), torch.cuda.current_stream().cuda_stream)
        assert rc == 0, (name, rc)

    def dequant(buf, rows):
        assert torch.isfinite(buf["scale"][:rows]).all() and (buf["scale"][:rows] > 0).all()
        return buf["fp8"][:rows].float() * buf["scale"][:rows, :, None]

    def canaries(buf, rows):
        assert (buf["out"] == 123.).all(), "fused epilogue wrote BF16 output"
        assert torch.isnan(buf["lse"]).all(), "fixed Prefill unexpectedly wrote LSE"
        assert (buf["fp8"][rows:].float() == 3.).all(), "FP8 tail overwritten"
        assert (buf["scale"][rows:] == -123.).all(), "scale tail overwritten"

    def errors(a, b):
        assert a.shape == b.shape and a.isfinite().all() and b.isfinite().all(), "nonfinite/shape"
        delta = a.float() - b.float()
        return {"relative_l2": (delta.norm() / b.float().norm().clamp_min(1e-12)).item(),
                "max_scaled": (delta.abs().amax() / b.float().abs().amax().clamp_min(1e-12)).item()}

    def reference(data):
        rows = data["rows"]
        selected = sorted({0, 1, 7, 8, 15, 16, 127, 128, 223, 224, 225, rows - 2, rows - 1,
                           *(round(i * (rows - 1) / 31) for i in range(32))})
        index = torch.tensor(selected, device="cuda")
        q = data["q"].float()[index].reshape(-1, 4, 16, 128).permute(1, 2, 0, 3)
        k = data["k"].float().permute(1, 2, 0)
        v = data["v"].float().permute(1, 0, 2)
        scores = torch.einsum("hgsd,hdk->hgsk", q, k) * (128**-.5)
        scores.masked_fill_(torch.arange(rows, device="cuda")[None, :] > index[:, None], -float("inf"))
        return index, torch.einsum("hgsk,hkd->shgd", scores.softmax(-1), v).reshape(-1, 64, 128)

    def capture(fn):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        return graph

    try:
        save()
        weight = torch.randn((4096, 8192), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        weight_scale = 10. ** (torch.rand((32, 64), device="cuda") * 3 - 3)
        weight_fp32 = weight.float() * weight_scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        for rows in (3584, *TIMED_ROWS):
            data = inputs(rows)
            fill(data, "normal")
            control, poly, image = [buffers(rows) for _ in range(3)]
            graph = capture(lambda data=data, poly=poly: call(data, poly, "poly"))
            for profile in ("normal", "flat", "sharp", "zero_v"):
                fill(data, profile)
                call(data, control, "control")
                call(data, image, "image")
                assert torch.equal(control["fp8"].view(torch.uint8), image["fp8"].view(torch.uint8)), (rows, profile, "control != pinned image FP8")
                assert torch.equal(control["scale"], image["scale"]), (rows, profile, "control != pinned image scales")
                call(data, poly, "poly")
                fresh = poly["fp8"].clone(), poly["scale"].clone()
                graph.replay()
                assert torch.equal(poly["fp8"].view(torch.uint8), fresh[0].view(torch.uint8)) and torch.equal(poly["scale"], fresh[1]), "changed-input graph != fresh candidate"
                for buf in (control, poly, image):
                    canaries(buf, rows)
                a, b = dequant(poly, rows), dequant(control, rows)
                index, ref = reference(data)
                delta, ce, pe = errors(a, b), errors(b[index], ref), errors(a[index], ref)
                record = {"rows": rows, "profile": profile, "reference_rows": len(index), "delta": delta,
                          "control_reference": ce, "candidate_reference": pe,
                          "differing_fp8_codes": int((poly["fp8"][:rows].view(torch.uint8) != control["fp8"][:rows].view(torch.uint8)).sum()),
                          "pinned_control_exact": True, "changed_input_graph_exact": True, "canaries_passed": True}
                results["numeric"].append(record)
                save()
                assert accept_errors(delta, ce, pe), record
                if profile == "zero_v":
                    assert torch.count_nonzero(a) == 0 and torch.count_nonzero(b) == 0, "zero V"
                if profile == "normal" and rows in TIMED_ROWS:
                    dense = []
                    for buf in (control, poly):
                        result = torch.empty((rows, 4096), device="cuda", dtype=torch.bfloat16)
                        deep_gemm.fp8_gemm_nt((buf["fp8"][:rows].view(rows, 8192), buf["scale"][:rows]), (weight, weight_scale), result)
                        dense.append(result)
                    expected = ref.reshape(len(index), 8192) @ weight_fp32.T
                    dd, dc, dp = errors(dense[1], dense[0]), errors(dense[0][index], expected), errors(dense[1][index], expected)
                    projection = {"rows": rows, "delta": dd, "control_reference": dc, "candidate_reference": dp}
                    results["o_projection"].append(projection)
                    save()
                    assert accept_errors(dd, dc, dp), projection
            del graph, data, control, poly, image, a, b, fresh
        assert any(r["delta"]["relative_l2"] > 0 for r in results["numeric"]), "candidate has no observable effect"
        results["numerical_passed"] = True
        save()
        props = torch.cuda.get_device_properties(0)
        results["device"] = {"name": props.name, "uuid": str(props.uuid), "sm_count": props.multi_processor_count}
        clocks = []
        for rows in TIMED_ROWS:
            sets = [inputs(rows) for _ in range(4)]
            for data in sets:
                fill(data, "normal")
            outputs = {name: [buffers(rows) for _ in sets] for name in libs}
            graphs = {}
            for name in libs:
                def batch(name=name, sets=sets, outputs=outputs):
                    for i in range(8):
                        call(sets[i % 4], outputs[name][i % 4], name)
                graphs[name] = capture(batch)
            for i in range(8):
                graphs["control" if i % 2 == 0 else "poly"].replay()
            torch.cuda.synchronize()
            samples = {name: [] for name in libs}
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            for pair in range(24):
                for name in (("control", "poly") if pair % 2 == 0 else ("poly", "control")):
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) * 1000 / 8)
            clocks.append(subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader,nounits"], text=True).strip())
            medians = {name: statistics.median(values) for name, values in samples.items()}
            ratios = [p / c for p, c in zip(samples["poly"], samples["control"], strict=True)]
            row = {"rows": rows, "samples_us": samples, "paired_ratios": ratios, "median_us": medians,
                   "ratio": statistics.median(ratios), "saving_us": medians["control"] - medians["poly"],
                   "paired_ratio_p90": sorted(ratios)[math.ceil(.9 * len(ratios)) - 1],
                   "control_drift": statistics.median(samples["control"][-5:]) / statistics.median(samples["control"][:5]) - 1,
                   "rotating_sets": 4, "calls_per_replay": 8, "alternating_pairs": 24}
            results["timing"].append(row)
            save()
            del graphs, sets, outputs
        results["candidate_eligible"] = eligible(results["timing"])
        results["clock_power_temperature_samples"] = clocks
        results["elapsed_s"] = time.monotonic() - began
        save()
        print("RESULT_JSON " + json.dumps(results), flush=True)
        print(f"PASS kernel=fa3_exp numerical_passed=true candidate_eligible={results['candidate_eligible']}", flush=True)
    except BaseException as exc:
        results["failure"] = repr(exc)
        results["elapsed_s"] = time.monotonic() - began
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
