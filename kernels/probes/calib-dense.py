"""Calibrate Decode M128 QKV/O using the serving selector's exact graph method."""

import os
import time
from pathlib import Path
from runpy import run_path

# Avoid unrelated exhaustive SGLang shape warmup in the bounded allocation.
os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
STARTED = time.monotonic()

import torch
from sglang.srt.layers.quantization.decode_dense_gemm import run_decode_dense_gemm


def main():
    root = Path(__file__).parent
    common = run_path(str(root / "calib-common.py"))
    common["device_info"](torch)
    selector = run_path(str(root / "select-firmus-decode-dense.py"))
    checks = run_path(str(root / "check-firmus-decode-kernels.py"))
    checks["require_dense_host"]()
    for key in selector["ENV_KEYS"]:
        os.environ.pop(key, None)
    original = checks["dense_call"]

    def call(candidate, inputs):
        if candidate["backend"] == "deepgemm":
            return original(candidate, inputs)
        lhs, rhs, out = inputs
        op = "QKV" if rhs[0].shape == (9216, 4096) else "O"
        return lambda: run_decode_dense_gemm(op, candidate["name"], lhs, rhs, out)

    checks["dense_call"] = call
    checks["dense_validate_candidate"].__globals__["dense_call"] = call
    for op, n, k in checks["DENSE_PROJECTIONS"]:
        # O uses its own Firmus-screened geometry, not QKV's N9216 tuning.
        configs = ((128, 80, 7, 1, 0), (64, 144, 5, 1, 0), (128, 144, 5, 1, 0), (128, 80, 7, 2, 1)) if op == "qkv" else ((128, 32, 10, 1, 0), (64, 64, 9, 1, 0), (128, 64, 8, 1, 0), (64, 64, 13, 2, 1))
        choices = checks["dense_candidates"](op)
        candidates = [choices[0], *[{"name": "dg-" + "-".join(map(str, c)), "backend": "deepgemm", "config": c} for c in configs], next(c for c in choices if c["backend"] == "triton")]
        with checks["dense_fp32_reference"]():
            prepared = selector["prepare_projection"](torch, checks, op, n, k, candidates)
        for name, result in prepared["results"].items():
            assert result["numeric_passed"], (op, name, result)
        names = [c["name"] for c in candidates]
        for block in range(selector["BLOCKS"]):
            for name in selector["round_robin"](names, block):
                prepared["results"][name]["blocks"][block].extend(selector["measure_graph"](torch, prepared["graphs"][name]))
        stock_blocks = prepared["results"]["stock"]["blocks"]
        for name in names:
            result = prepared["results"][name]
            common["report"](f"{op}-M128-N{n}-K{k}", name, result["blocks"], stock_blocks, result["numeric"])
        del prepared
    print(f"PASS calibration=dense m=128 blocks=2 weight_sets=4 calls_per_graph=20 replays=25 elapsed_seconds={time.monotonic() - STARTED:.3f}", flush=True)


if __name__ == "__main__":
    main()
