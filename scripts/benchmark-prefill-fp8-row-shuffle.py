"""Finite bitwise/copy timing probe; no quantization or precision change."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cpu_contract(q_rows, scale_rows, dst2src):
    """Native contract oracle: valid EP1 destination-to-source entries only."""
    assert len(q_rows) == len(scale_rows)
    assert all(0 <= i < len(q_rows) for i in dst2src)
    return [q_rows[i] for i in dst2src], [scale_rows[i] for i in dst2src]


def main():
    import torch
    from sglang.kernels.ops.moe.prefill_fp8_row_shuffle import (
        _shuffle_fp8_scale_bits,
        shuffle_fp8_rows_with_scales,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(20261004)
    result = {"purpose": "finite bitwise fused gather probe, no serving adoption", "device": torch.cuda.get_device_name(), "torch": torch.__version__, "num_warps": 8, "rows_per_program": 1, "cases": []}
    def make_runs(q, scales, mapping, stock_q, stock_s, fused_q, fused_s):
        rows = mapping.numel()
        def stock():
            if rows:
                torch.ops.sgl_kernel.shuffle_rows.default(q, mapping, stock_q)
                torch.ops.sgl_kernel.shuffle_rows.default(scales, mapping, stock_s)

        def fused():
            if rows:
                _shuffle_fp8_scale_bits[(rows,)](q.view(torch.uint8), scales.view(torch.int32), mapping, fused_q.view(torch.uint8), fused_s.view(torch.int32), K=4096, G=32, num_warps=8)

        return stock, fused

    for tokens in (8192, 4096, 32, 0):
        rows = tokens * 8
        # Valid source indices, permutations/repeated sources and row endpoints.
        mapping = (torch.randperm(rows, device="cuda") // 8).to(torch.int32)
        q = (torch.arange(tokens * 4096, device="cuda", dtype=torch.int64) % 256).to(torch.uint8).reshape(tokens, 4096).view(torch.float8_e4m3fn)
        scale_bits = (torch.arange(tokens * 32, device="cuda", dtype=torch.int64) * 104729).to(torch.int32).reshape(tokens, 32)
        if tokens:
            scale_bits[0, :4] = torch.tensor([0, -2147483648, 2143289345, -4194303], dtype=torch.int32, device="cuda")
        scales = scale_bits.view(torch.float32)
        stock_q = torch.empty((rows, 4096), device="cuda", dtype=q.dtype)
        stock_s = torch.empty((rows, 32), device="cuda", dtype=scales.dtype)
        fused_q, fused_s = torch.empty_like(stock_q), torch.empty_like(stock_s)

        stock, fused = make_runs(q, scales, mapping, stock_q, stock_s, fused_q, fused_s)

        stock()
        fused()
        torch.cuda.synchronize()
        assert torch.equal(stock_q.view(torch.uint8), fused_q.view(torch.uint8))
        assert torch.equal(stock_s.view(torch.int32), fused_s.view(torch.int32))
        public_q, public_s = shuffle_fp8_rows_with_scales(q, scales, mapping)
        assert torch.equal(stock_q.view(torch.uint8), public_q.view(torch.uint8))
        assert torch.equal(stock_s.view(torch.int32), public_s.view(torch.int32))
        # Quantize once, then use the exact same quant output for both gather paths.
        if tokens:
            q, scales = sglang_per_token_group_quant_fp8(torch.randn((tokens, 4096), device="cuda", dtype=torch.bfloat16), 128)
            stock, fused = make_runs(q, scales, mapping, stock_q, stock_s, fused_q, fused_s)
            stock()
            fused()
            torch.cuda.synchronize()
            assert torch.equal(stock_q.view(torch.uint8), fused_q.view(torch.uint8))
            assert torch.equal(stock_s.view(torch.int32), fused_s.view(torch.int32))
        entry = {"tokens": tokens, "rows": rows, "bitwise_raw_patterns": "pass", "bitwise_quant_once": "pass" if tokens else "empty", "preallocated_outputs": True, "timings": {}}
        if rows:
            for name, run in (("stock", stock), ("fused", fused)):
                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, capture_error_mode="global"):
                    run()
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(stock_q.view(torch.uint8), fused_q.view(torch.uint8))
                assert torch.equal(stock_s.view(torch.int32), fused_s.view(torch.int32))
                measurements = []
                for _ in range(5):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    measurements.append(start.elapsed_time(end) / 20)
                entry["timings"][name] = {"graph_ms": measurements, "replays_per_measurement": 20}
                del graph
        result["cases"].append(entry)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
