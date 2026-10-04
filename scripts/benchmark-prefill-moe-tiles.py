"""Finite Qwen3 TP4 MoE comparison; synthetic routes are not measured route counts."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
import time
from pathlib import Path


def cases():
    return [(8192, "balanced"), (8192, "skewed"), (4096, "skewed"), (32, "few_experts")]


def main():
    import sglang.srt.layers.moe.cutlass_moe as moe
    import torch
    from sglang.srt.layers.moe.cutlass_moe import cutlass_fused_experts_fp8

    apac_prefill_moe_tiles = importlib.import_module("apac_prefill_moe_tiles")  # torch libraries loaded

    # Force the same geometry on small oracle inputs only inside this finite external probe.
    def finite_mm(mode, *inputs):
        return moe.fp8_blockwise_scaled_grouped_mm(*inputs) if mode < 0 else apac_prefill_moe_tiles.grouped_mm(*inputs, mode)

    moe._prefill_tile_mm = finite_mm

    native_manifest = json.loads(Path("/opt/apac/prefill-moe-tiles/manifest.json").read_text())
    assert native_manifest["architecture"] == "sm_90a" and torch.cuda.get_device_capability() == (9, 0)
    assert hashlib.sha256(Path(apac_prefill_moe_tiles.__file__).read_bytes()).hexdigest() == native_manifest["module_sha256"]

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
    workspace = torch.empty(1048576, device=device, dtype=torch.uint8)
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

    result = {"purpose": "finite geometry probe; no serving selection, small native oracle forced only in probe", "model_shape": {"experts": experts, "hidden": hidden, "intermediate_per_tp_rank": intermediate, "topk": topk}, "workspace_bytes": 1048576, "device": torch.cuda.get_device_name(), "torch": torch.__version__, "native_module": apac_prefill_moe_tiles.__file__, "native_manifest": native_manifest, "route_counts_measured_from_model": False, "quality_policy": "No elementwise equality gate. Small shape BF16/FP32 reference relative-L2 candidate envelope is baseline*1.25+0.005; large shape stock relative-L2<0.1 detects gross systematic errors. Independent reference is not downstream accuracy proof. Candidate relative-L2 above1e-3 requires investigation; the0.1 threshold is only gross-error screening, not quality approval.", "cases": []}
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

        def run(modes, x=x, weights=weights, ids=ids):
            return cutlass_fused_experts_fp8(x, w1.transpose(1, 2), w2.transpose(1, 2), s1.transpose(1, 2), s2.transpose(1, 2), weights, ids, *strides, workspace, *pointers, offsets, *problems, enable_es=(False, False), prefill_tile_modes=modes)

        reference = run((-1, -1)).clone()
        torch.cuda.synchronize()
        dense_refs = {name: dense_reference(x, ids, weights, dtype) for name, dtype in [("bf16", torch.bfloat16), ("fp32", torch.float32)]} if tokens == 32 else {}
        baseline_errors = {name: relative_l2(reference, dense) for name, dense in dense_refs.items()}
        record["baseline_reference_relative_l2"] = baseline_errors
        graphs = {}
        for name, modes in [("wheel", (-1, -1)), ("clone", (0, 0)), ("down-c1", (-1, 1)), ("down-n256", (-1, 2)), ("down-pingpong", (-1, 3)), ("up-pingpong", (4, -1))]:
            record["in_progress"] = name
            args.output.write_text(json.dumps(result | {"current_case": record}, indent=2) + "\n")
            try:
                actual = run(modes)
            except RuntimeError as error:
                record["variants"][name] = {"status": "failed", "error": str(error)}
                args.output.write_text(json.dumps(result | {"current_case": record}, indent=2) + "\n")
                if name != "clone" and "Failed to implement GEMM" in str(error):
                    continue
                raise
            torch.cuda.synchronize()
            a, b = actual.float(), reference.float()
            diff = float(1 - 2 * (a * b).sum() / (a.square().sum() + b.square().sum()))
            stock_error = relative_l2(actual, reference)
            abs_error = (a - b).abs()
            max_abs = float(abs_error.max())
            max_rel = float((abs_error / b.abs().clamp_min(1e-8)).max())
            max_abs_relative_to_peak = max_abs / max(float(b.abs().max()), 1e-8)
            numerical_attention = name != "wheel" and stock_error > 1e-3
            dense_errors = {ref_name: relative_l2(actual, dense) for ref_name, dense in dense_refs.items()}
            assert torch.isfinite(actual).all() and stock_error < 0.1, (tokens, distribution, name, stock_error)
            bitwise_equal = bool(torch.equal(actual, reference))
            if name == "clone" and tokens > 32:
                assert stock_error <= 1e-3, ("stock clone equivalence failed", tokens, distribution, stock_error)
            for ref_name, error in dense_errors.items():
                assert error <= baseline_errors[ref_name] * 1.25 + 0.005, (name, ref_name, baseline_errors[ref_name], error)
            for _ in range(3):
                run(modes)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            for _ in range(20):
                run(modes)
            end.record()
            end.synchronize()
            eager_ms = start.elapsed_time(end) / 20
            wall_ms = (time.perf_counter() - wall_start) * 1000 / 20
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="global"):
                captured = run(modes)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            graph_error = relative_l2(captured, reference)
            graph_abs_error = (captured.float() - reference.float()).abs()
            graph_max_abs = float(graph_abs_error.max())
            graph_max_rel = float((graph_abs_error / reference.float().abs().clamp_min(1e-8)).max())
            graph_diff = float(1 - 2 * (captured.float() * reference.float()).sum() / (captured.float().square().sum() + reference.float().square().sum()))
            assert torch.isfinite(captured).all() and relative_l2(captured, reference) < 0.1, (tokens, distribution, name, graph_diff)
            if name == "clone" and tokens > 32:
                assert relative_l2(captured, reference) <= 1e-3, ("stock clone graph equivalence failed", tokens, distribution)
            for ref_name, dense in dense_refs.items():
                assert relative_l2(captured, dense) <= baseline_errors[ref_name] * 1.25 + 0.005, (name, ref_name, "graph")
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            end.synchronize()
            graph_ms = start.elapsed_time(end) / 20
            if name == "clone" and tokens > 32:
                record["clone_to_wheel_timing_ratio"] = graph_ms / record["variants"]["wheel"]["graph_cuda_event_ms"]
            record["variants"][name] = {"tile_modes": modes, "bitwise_equal_to_wheel": bitwise_equal, "reference_normalized_diff": diff, "stock_relative_l2": stock_error, "max_absolute_error": max_abs, "max_relative_error_denominator_floor_1e_minus8": max_rel, "max_abs_relative_to_reference_peak": max_abs_relative_to_peak, "numerical_attention_not_quality_approved": numerical_attention, "all_finite": True, "dense_reference_relative_l2": dense_errors, "graph_normalized_diff": graph_diff, "graph_relative_l2": graph_error, "graph_max_absolute_error": graph_max_abs, "graph_max_relative_error_denominator_floor_1e_minus8": graph_max_rel, "graph_numerical_attention_not_quality_approved": name != "wheel" and graph_error > 1e-3, "eager_cuda_event_ms": eager_ms, "eager_host_wall_ms": wall_ms, "graph_cuda_event_ms": graph_ms, "graph_capture": "success", "repeats": 20}
            graphs[name] = (graph, captured)
            del actual
            record.pop("in_progress", None)
        # Bound five rounds, rotating/reversing order to distinguish geometry from measurement order.
        samples = {name: [] for name in graphs}
        names = list(graphs)
        for round_index in range(5):
            order = names[round_index % len(names):] + names[:round_index % len(names)]
            if round_index % 2:
                order.reverse()
            for name in order:
                graph, _ = graphs[name]
                graph.replay()
                torch.cuda.synchronize()
                start.record()
                for _ in range(20):
                    graph.replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / 20)
        for name, values in samples.items():
            record["variants"][name]["graph_cuda_event_ms"] = statistics.median(values)
            record["variants"][name]["graph_cuda_event_rounds_ms"] = values
        ratio = record["variants"]["clone"]["graph_cuda_event_ms"] / record["variants"]["wheel"]["graph_cuda_event_ms"]
        record["clone_to_wheel_timing_ratio"] = ratio
        record["large_shape_clone_timing_consistent_with_wheel"] = 0.9 <= ratio <= 1.1 if tokens > 32 else None
        record["timing_limit"] = "five bounded 20-replay rounds; median/range, synthetic routing; no E2E claim"
        del graphs, graph, captured
        result["cases"].append(record)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    print("Finite probe completed; numerical/oracle diagnostics must be reviewed before any candidate E2E; serving remains off.", flush=True)


if __name__ == "__main__":
    main()
