"""Calibrate B32 TARGET_VERIFY with B4 shared metadata and 94-layer graphs."""

import os
import time
from pathlib import Path
from runpy import run_path

os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
STARTED = time.monotonic()

import torch
from sglang.srt.layers.attention import decode_fa3_verify


def main():
    root = Path(__file__).parent
    common = run_path(str(root / "calib-common.py"))
    common["device_info"](torch)
    selector = run_path(str(root / "select-firmus-decode-fa3.py"))
    checks = run_path(str(root / "check-firmus-decode-kernels.py"))
    decode_fa3_verify.load()
    names = ("control96", "p128s2", "t64s2", "t64s3")
    assert all(n in decode_fa3_verify.NAMES for n in names)
    commons = [checks["verify_inputs"](32, 93101 + 32 + i)[0] for i in range(4)]
    for c in commons[1:]:
        c["cache_seqlens"] = commons[0]["cache_seqlens"]
        c["cu_seqlens_q"] = commons[0]["cu_seqlens_q"]
    # Validate stock too; candidates retain the existing gate's FP32,
    # boundary-length and changed-input checks without relaxing tolerances.
    stock_numeric = checks["verify_validate_candidate"](32, "stock")
    stock_step = checks["verify_step"](commons, "stock")
    stock_graph = checks["verify_capture"](stock_step)
    pooled_stock = [[], []]
    for name in names:
        result, step, graph = selector["prepare_candidate"](torch, checks, 32, name, commons)
        assert result["numeric_passed"] and "error" not in result, (name, result)
        stocks, candidates = [], []
        for block in range(selector["BLOCKS"]):
            stock_times, candidate_times = [], []
            stock_graph.replay()
            graph.replay()
            for pair in range(selector["PAIRS"]):
                order = ((stock_graph, stock_times), (graph, candidate_times))
                if (pair + block) % 2:
                    order = tuple(reversed(order))
                for current, times in order:
                    times.append(selector["measure"](torch, current))
            stocks.append(stock_times)
            candidates.append(candidate_times)
            pooled_stock[block].extend(stock_times)
        common["report"]("fa3-B32-Q4-L94", name, candidates, stocks, result)
        del graph, step
    common["report"]("fa3-B32-Q4-L94", "stock", pooled_stock, pooled_stock, stock_numeric)
    print(f"PASS calibration=fa3 batch=32 layers=94 kv_sets=4 blocks=2 pairs_per_block=20 elapsed_seconds={time.monotonic() - STARTED:.3f}", flush=True)


if __name__ == "__main__":
    main()
