#!/usr/bin/env python3
"""Bounded pre-model ASPIRE diagnostic; CUDA event times are not E2E gains."""
from __future__ import annotations

import json
import statistics
import time
import traceback
from pathlib import Path

OUTPUT = Path('/results/silu-quant-diagnostic.json')


def shape_specs() -> list[dict[str, int]]:
    return [{'tokens': t, 'rows': 8 * t, 'input_width': 768, 'output_width': 384, 'group_size': 128} for t in (1024, 4096, 8192)]


def graph_timing(torch, operation) -> tuple[dict, object]:
    # Import/JIT and graph capture are deliberately outside CUDA-event timing.
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = operation()
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(32):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return {'replays_timed': len(samples), 'median_us': statistics.median(samples), 'min_us': min(samples), 'max_us': max(samples), 'samples_us': samples}, output


def error_metrics(torch, actual, reference) -> dict[str, float]:
    difference = actual - reference
    return {'max_absolute': difference.abs().max().item(), 'rms': difference.square().mean().sqrt().item(), 'relative_l2': (difference.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-12)).item()}


def run(result: dict) -> None:
    import torch
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    assert torch.cuda.device_count() == 1, 'diagnostic must expose only allocated GPU0'
    torch.cuda.set_device(0)
    torch.manual_seed(2026)
    result['device'] = {'name': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)), 'torch': torch.__version__}
    assert torch.cuda.get_device_capability(0)[0] == 9, 'bounded diagnostic requires Hopper'
    for shape in shape_specs():
        setup_start = time.monotonic()
        rows = shape['rows']
        x = torch.randn((rows, 768), device='cuda', dtype=torch.bfloat16)
        intermediate = torch.empty((rows, 384), device='cuda', dtype=torch.bfloat16)

        def baseline(x=x, intermediate=intermediate):
            silu_and_mul(x, intermediate)
            return sglang_per_token_group_quant_fp8(intermediate, 128)

        def fused(x=x):
            return sglang_per_token_group_quant_fp8(x, 128, fuse_silu_and_mul=True)

        baseline_time, (baseline_q, baseline_s) = graph_timing(torch, baseline)
        fused_time, (fused_q, fused_s) = graph_timing(torch, fused)
        # Compare dequantized results with a FP32-input arithmetic reference.
        # This is synthetic numerical characterization, not model accuracy.
        reference = torch.nn.functional.silu(x[:, :384].float()) * x[:, 384:].float()
        actuals = {}
        for name, q, scale in [('baseline', baseline_q, baseline_s), ('fused', fused_q, fused_s)]:
            assert tuple(q.shape) == (rows, 384) and tuple(scale.shape) == (rows, 3)
            assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
            finite = bool(torch.isfinite(q.float()).all().item() and torch.isfinite(scale).all().item())
            assert finite, f'{name} produced nonfinite output/scales'
            dequant = (q.float().view(rows, 3, 128) * scale[:, :, None]).view(rows, 384)
            actuals[name] = {'q_shape': list(q.shape), 'scale_shape': list(scale.shape), 'q_dtype': str(q.dtype), 'scale_dtype': str(scale.dtype), 'scale_stride': list(scale.stride()), 'finite': finite, 'scale_min': scale.min().item(), 'scale_max': scale.max().item(), 'dequant_error_vs_fp32_reference': error_metrics(torch, dequant, reference)}
            if name == 'baseline':
                baseline_dequant = dequant
            else:
                actuals['fused_vs_baseline'] = error_metrics(torch, dequant, baseline_dequant)
        result['shapes'].append({'shape': shape, 'baseline': baseline_time, 'fused': fused_time, 'median_kernel_region_speedup': baseline_time['median_us'] / fused_time['median_us'], 'numerical': actuals, 'shape_elapsed_wall_seconds_including_setup_JIT_and_validation': time.monotonic() - setup_start})
        OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
        del baseline, fused, x, intermediate, baseline_q, baseline_s, fused_q, fused_s, reference, dequant, baseline_dequant
    torch.cuda.synchronize()


def main() -> None:
    started = time.monotonic()
    result = {'schema_version': 'qwen-silu-quant-diagnostic/v1', 'status': 'running', 'scope': 'Synthetic same-input isolated kernel microbenchmark; not official benchmark/E2E speedup or model accuracy', 'phase': 'ASPIRE node0 GPU0 before model launch', 'seed': 2026, 'input_dtype': 'bfloat16', 'shapes': []}
    try:
        run(result)
        result['status'] = 'completed'
    except (AssertionError, RuntimeError, ImportError, OSError, ValueError, AttributeError, TypeError) as error:
        result['status'] = 'error'
        result['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc()
    result['elapsed_wall_seconds'] = time.monotonic() - started
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
