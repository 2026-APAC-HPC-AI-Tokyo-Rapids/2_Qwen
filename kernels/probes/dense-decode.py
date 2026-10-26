"""One H200, existing numeric gate and rotating-weight ABBA graph timing."""
import json
import os
import runpy
from pathlib import Path

import dense_dg1
import torch

assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
root = Path(__file__).resolve().parent
assert Path(dense_dg1.__file__).resolve().parent == root / "dense-dg1", "probe must load the freshly compiled companion"
checks = runpy.run_path(str(root / "check-firmus-decode-kernels.py"))
selector = runpy.run_path(str(root / "select-firmus-decode-dense.py"))
# Use exactly the serving overlay entry point for the new companion.
from sglang.srt.layers.quantization.decode_dense_gemm import run_decode_dense_gemm

original = checks["dense_call"]


def serving_call(candidate, inputs):
    if candidate["backend"] == "deepgemm":
        return original(candidate, inputs)
    lhs, rhs, out = inputs
    op = "QKV" if rhs[0].shape[0] == 9216 else "O"
    return lambda: run_decode_dense_gemm(op, candidate["name"], lhs, rhs, out)


checks["dense_call"] = serving_call
checks["dense_validate_candidate"].__globals__["dense_call"] = serving_call
checks["require_dense_host"]()
results = []
with checks["dense_fp32_reference"]():
    for op, n, k in checks["DENSE_PROJECTIONS"]:
        choices = checks["dense_candidates"](op)
        # Include every currently selectable config, as well as stock.
        for rows in (128, 160, 176):
            prepared = selector["prepare_projection"](torch, checks, op, n, k, choices, rows)
            active = [c["name"] for c in choices if prepared["results"][c["name"]]["numeric_passed"]]
            assert len(active) == len(choices), prepared["results"]
            for block in range(selector["BLOCKS"]):
                for name in selector["round_robin"](active, block):
                    samples = selector["measure_graph"](torch, prepared["graphs"][name])
                    prepared["results"][name]["blocks"][block].extend(samples)
            for name in active:
                record = {"op": op, "m": rows, "candidate": name, **prepared["results"][name]}
                results.append(record)
                print(json.dumps({**record, "distribution": selector["distribution"]([x for b in record["blocks"] for x in b])}), flush=True)
            del prepared
            torch.cuda.empty_cache()
output = Path(os.environ["RUNTIME_DIR"]) / "results" / "dense-decode.json"
output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
print("PASS dense DG1 QKV/O M=128/160/176 FP32 block-scale gate and changed-input graphs", flush=True)
