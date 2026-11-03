"""NumPy model of 16 values/lane at H4096; not a Triton/GPU rounding proof."""
import argparse
import json
from pathlib import Path

import numpy as np


def bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def codes(x):
    # E4M3 finite positive codebook; search neighboring values with RNE ties.
    code = np.arange(127)
    table = ((code & 7) / 8 + (code >> 3 != 0)) * np.exp2(np.where(code >> 3, (code >> 3) - 7, -6))
    absolute = np.abs(x).clip(0, 448)
    upper = np.searchsorted(table, absolute).clip(0, 126)
    lower = (upper - 1).clip(0, 126)
    du, dl = table[upper] - absolute, absolute - table[lower]
    pick = np.where((dl < du) | ((dl == du) & (lower % 2 == 0)), lower, upper)
    return (pick + np.signbit(x) * 128).astype(np.uint8), table


def reductions(h):
    values = h.reshape(len(h), 16, 256)
    control = values[:, 0] * values[:, 0]
    candidate = control.copy()
    for i in range(1, 16):
        v = values[:, i]
        control = np.add(control, np.multiply(v, v, dtype=np.float32), dtype=np.float32)
        # Exact FP32 product in FP64 then one FP32 rounding (finite inputs in
        # these cases); NumPy has no portable float32 fused multiply-add.
        candidate = (v.astype(np.float64) * v.astype(np.float64) + candidate.astype(np.float64)).astype(np.float32)
    def tree(a):
        while a.shape[-1] > 1:
            n = a.shape[-1] // 2
            a = np.add(a[:, :n], a[:, n:], dtype=np.float32)
        return a
    return tree(control), tree(candidate)


def evaluate(x, r, w):
    h = np.add(x, r, dtype=np.float32)
    exact_h = x.astype(np.float64) + r.astype(np.float64)
    y64 = exact_h / np.sqrt(np.mean(exact_h ** 2, axis=1, keepdims=True) + 1e-6) * w.astype(np.float64)
    ref = bf16(y64).astype(np.float64).reshape(len(h), 32, 128)
    refmax = np.maximum(np.abs(ref).max(-1, keepdims=True), 1e-10)
    reference_codes, table = codes(ref * (448.0 / refmax))
    report = {}
    for mode, total in zip(("control", "candidate"), reductions(h), strict=True):
        denominator = np.add(total / np.float32(4096), np.float32(1e-6), dtype=np.float32)
        rstd = np.divide(np.float32(1), np.sqrt(denominator), dtype=np.float32)
        y32 = np.multiply(np.multiply(h, rstd, dtype=np.float32), w, dtype=np.float32)
        groups = bf16(y32).reshape(len(h), 32, 128)
        amax = np.maximum(np.abs(groups).max(-1, keepdims=True), np.float32(1e-10))
        gain = np.divide(np.float32(448), amax, dtype=np.float32)
        q, _ = codes(np.multiply(groups, gain, dtype=np.float32))
        scale = amax * np.float32(1 / 448)
        dequant = table[q & 127] * np.where(q >= 128, -1, 1) * scale
        report[mode] = {"max_abs_normalization_error": float(np.abs(y32.astype(np.float64) - y64).max()), "max_abs_error": float(np.abs(dequant.reshape(y64.shape) - y64).max()), "fp8_code_mismatches": int(np.count_nonzero(q != reference_codes))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20261103)
    w = bf16(1 + rng.standard_normal(4096).astype(np.float32) * .1)
    report = {"shape": [7988, 4096], "gpu_executed": False, "triton_executed": False, "limits": "NumPy IEEE sqrt, assumed 16 strided values/lane and 256-lane binary reduction, not GPU approximate rsqrt or a compiled Triton layout proof", "cases": {}}
    real = {m: {"max_abs_normalization_error": 0.0, "max_abs_error": 0.0, "fp8_code_mismatches": 0} for m in ("control", "candidate")}
    for start in range(0, 7988, 128):
        n = min(128, 7988 - start)
        x = bf16(rng.standard_normal((n, 4096)).astype(np.float32))
        r = bf16(rng.standard_normal((n, 4096)).astype(np.float32) * 4)
        observed = evaluate(x, r, w)
        for mode, metrics in real.items():
            for metric, value in metrics.items():
                metrics[metric] = value + observed[mode][metric] if metric.endswith("mismatches") else max(value, observed[mode][metric])
    report["cases"]["real"] = real
    x = bf16(rng.standard_normal((256, 4096)).astype(np.float32)); r = bf16(x * .25)
    x[::4] *= np.float32(2 ** 50); r[::4] *= np.float32(2 ** 50)
    x[1::4] *= np.float32(2 ** -60); r[1::4] *= np.float32(2 ** -60)
    x[2::4] = -r[2::4]
    report["cases"]["extreme"] = evaluate(x, r, w)
    report["cases"]["zero"] = evaluate(np.zeros_like(x), np.zeros_like(r), w)
    _, table = codes(np.zeros(1))
    mid = (table[1:] + table[:-1]) * .5 / 448
    v = np.concatenate((mid, np.nextafter(mid.astype(np.float32), np.float32(np.inf)), np.nextafter(mid.astype(np.float32), np.float32(-np.inf))))
    x = bf16(np.resize(v, (256, 4096))); x[:, 0] = 1; x[1::2] *= -1
    report["cases"]["fp8-boundary"] = evaluate(x, np.zeros_like(x), np.ones_like(w))
    report["noninferior_in_model"] = all(v["candidate"][k] <= v["control"][k] for v in report["cases"].values() for k in ("max_abs_error", "fp8_code_mismatches"))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
