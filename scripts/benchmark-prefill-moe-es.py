"""Finite Qwen3 TP4 MoE comparison; synthetic routes are not measured route counts."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def cases():
    return [(8192, "balanced"), (8192, "skewed"), (4096, "skewed"), (32, "few_experts")]


def main():
    import torch
    from sglang.srt.layers.moe.cutlass_moe import cutlass_fused_experts_fp8

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(20261004)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"
    experts, hidden, intermediate, topk = 128, 4096, 384, 8
    w1 = (torch.randn((experts, intermediate * 2, hidden), device=device, dtype=torch.bfloat16) * 8).to(torch.float8_e4m3fn)
    w2 = (torch.randn((experts, hidden, intermediate), device=device, dtype=torch.bfloat16) * 8).to(torch.float8_e4m3fn)
    s1 = torch.full((experts, intermediate * 2 // 128, hidden // 128), 0.0025, device=device)
    s2 = torch.full((experts, hidden // 128, intermediate // 128), 0.0025, device=device)
    strides = [torch.full((experts,), value, device=device, dtype=torch.int64) for value in (hidden, intermediate * 2, intermediate, hidden)]
    workspace = torch.empty(90000, device=device, dtype=torch.uint8)
    pointers = [torch.empty(experts, device=device, dtype=torch.int64) for _ in range(5)]
    offsets = torch.empty(experts + 1, device=device, dtype=torch.int32)
    problems = [torch.empty((experts, 3), device=device, dtype=torch.int32) for _ in range(2)]
    def relative_l2(actual, reference):
        return float(torch.linalg.vector_norm(actual.float() - reference.float()) / torch.linalg.vector_norm(reference.float()))

    def dense_reference(x, ids, weights, dtype):
        # Independent dense math with the same dequantized checkpoint-style weights.
        out = torch.zeros_like(x, dtype=torch.float32)
        for expert in torch.unique(ids).tolist():
            rows, slots = torch.where(ids == expert)
            dw1 = w1[expert].float() * s1[expert].repeat_interleave(128, 0).repeat_interleave(128, 1)
            dw2 = w2[expert].float() * s2[expert].repeat_interleave(128, 0).repeat_interleave(128, 1)
            up = x[rows].to(dtype) @ dw1.to(dtype).transpose(0, 1)
            activation = torch.nn.functional.silu(up[:, :intermediate]) * up[:, intermediate:]
            down = activation @ dw2.to(dtype).transpose(0, 1)
            weighted = down.float() * weights[rows, slots].to(torch.bfloat16).float().unsqueeze(1)
            out.index_add_(0, rows, weighted)
        return out

    result = {"purpose": "finite synthetic local comparison; fixed serving mode off", "model_shape": {"experts": experts, "hidden": hidden, "intermediate_per_tp_rank": intermediate, "topk": topk}, "workspace_bytes": 90000, "device": torch.cuda.get_device_name(), "torch": torch.__version__, "route_counts_measured_from_model": False, "quality_policy": "No elementwise equality gate. Small shape BF16/FP32 reference relative-L2 candidate envelope is baseline*1.25+0.005; large shape stock relative-L2<0.1 detects gross systematic errors. Independent reference is not downstream accuracy proof.", "cases": []}
    for tokens, distribution in cases():
        x = torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16)
        row = torch.arange(tokens, device=device).unsqueeze(1)
        lane = torch.arange(topk, device=device).unsqueeze(0)
        if distribution == "balanced":
            ids = (row * topk + lane) % experts
        elif distribution == "skewed":
            ids = torch.where(row % 4 == 0, (row * 17 + lane * 13) % experts, (row + lane) % 16)
        else:
            ids = (row + lane) % 8
        ids = ids.to(torch.int32).contiguous()
        weights = torch.softmax(torch.randn((tokens, topk), device=device), dim=-1)
        counts = torch.bincount(ids.flatten().long(), minlength=experts).cpu().tolist()
        record = {"tokens": tokens, "distribution": distribution, "expert_counts": counts, "variants": {}}

        def run(flags, x=x, weights=weights, ids=ids):
            return cutlass_fused_experts_fp8(x, w1.transpose(1, 2), w2.transpose(1, 2), s1.transpose(1, 2), s2.transpose(1, 2), weights, ids, *strides, workspace, *pointers, offsets, *problems, enable_es=flags)

        reference = run((False, False)).clone()
        torch.cuda.synchronize()
        dense_refs = {name: dense_reference(x, ids, weights, dtype) for name, dtype in [("bf16", torch.bfloat16), ("fp32", torch.float32)]} if tokens == 32 else {}
        baseline_errors = {name: relative_l2(reference, dense) for name, dense in dense_refs.items()}
        record["baseline_reference_relative_l2"] = baseline_errors
        for name, flags in [("off", (False, False)), ("up", (True, False)), ("down", (False, True)), ("both", (True, True))]:
            actual = run(flags)
            torch.cuda.synchronize()
            a, b = actual.float(), reference.float()
            diff = float(1 - 2 * (a * b).sum() / (a.square().sum() + b.square().sum()))
            stock_error = relative_l2(actual, reference)
            dense_errors = {ref_name: relative_l2(actual, dense) for ref_name, dense in dense_refs.items()}
            assert torch.isfinite(actual).all() and stock_error < 0.1, (tokens, distribution, name, stock_error)
            for ref_name, error in dense_errors.items():
                assert error <= baseline_errors[ref_name] * 1.25 + 0.005, (name, ref_name, baseline_errors[ref_name], error)
            for _ in range(3):
                run(flags)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            for _ in range(20):
                run(flags)
            end.record()
            end.synchronize()
            eager_ms = start.elapsed_time(end) / 20
            wall_ms = (time.perf_counter() - wall_start) * 1000 / 20
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="global"):
                captured = run(flags)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            graph_diff = float(1 - 2 * (captured.float() * reference.float()).sum() / (captured.float().square().sum() + reference.float().square().sum()))
            assert torch.isfinite(captured).all() and relative_l2(captured, reference) < 0.1, (tokens, distribution, name, graph_diff)
            for ref_name, dense in dense_refs.items():
                assert relative_l2(captured, dense) <= baseline_errors[ref_name] * 1.25 + 0.005, (name, ref_name, "graph")
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            end.synchronize()
            graph_ms = start.elapsed_time(end) / 20
            record["variants"][name] = {"flags": flags, "reference_normalized_diff": diff, "stock_relative_l2": stock_error, "dense_reference_relative_l2": dense_errors, "graph_normalized_diff": graph_diff, "eager_cuda_event_ms": eager_ms, "eager_host_wall_ms": wall_ms, "graph_cuda_event_ms": graph_ms, "graph_capture": "success", "repeats": 20}
            del graph, captured, actual
        result["cases"].append(record)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    print("All four fixed local variants passed quality and CUDA graph checks; serving mode remains off.", flush=True)


if __name__ == "__main__":
    main()
