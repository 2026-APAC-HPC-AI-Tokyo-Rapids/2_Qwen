"""Standalone DeepGEMM grouped FP8 GEMM ceiling for the Prefill MegaMoE workload.

Run once on one GPU (cuda:0) inside the serving image. It times DeepGEMM's
contiguous-layout m-grouped FP8 GEMM (m_grouped_fp8_gemm_nt_contiguous, SM90
1d2d: per-token 1x128 FP32 activation scales, 128x128 FP32 weight scales) on the
per-rank expert shapes of the Qwen3 EP4 MegaMoE, for comparison with the fused
kernel's GEMM region (about 1.5 ms for 1.24 TFLOP per rank at 4096 tokens/rank):
  fc1: [rows, 4096] x [32, 3072, 4096]^T (gate+up)
  fc2: [rows, 1536] x [32, 4096, 1536]^T (down)
Each local expert's rows start on a 128-row boundary and the padding rows carry
index -1, as DeepGEMM's contiguous layout requires. Row layouts:
  even-256:    256 rows per expert, the 8192 rows of 4096 tokens x top-8 / 4 (a
               single 4096-token batch spread over the four ranks).
  even-1024:   1024 rows per expert, 32768 rows: 4096 tokens on each of the 4
               ranks x top-8 / 4 ranks, the rank load of the MegaMoE bench.
  routed-1024: expert counts of rank 0 sampled from the bench's balanced routing
               (top-8 of 128 uniform randoms, 16384 tokens), padded per expert.
TFLOPS counts the executed (padded) rows; tflops_useful the routed rows only.
The activation scales are laid out MN-major and TMA-aligned up front so the
timed call is the GEMM alone. Each case first runs about HEAT_SECONDS back to
back so the SM clock and power are those of sustained load, then times a CUDA
graph of REPS calls; the SM clock and power are read from NVML right before and
after the replays. One JSON line per case, {case, median_us, p10_us, p90_us,
tflops, ...}, plus a fc1+fc2 row per layout with the summed time.
"""
import argparse
import json
import os
import statistics
import time

os.environ["DG_PRINT_CONFIGS"] = "1"
import deep_gemm
import torch

EXPERTS, RANKS, TOPK, GLOBAL_EXPERTS = 32, 4, 8, 128
GEMMS = (("fc1", 3072, 4096), ("fc2", 4096, 1536))
LAYOUTS = ("even-256", "even-1024", "routed-1024")
ALIGN = 128
HEAT_SECONDS, REPS, REPLAYS = 3.0, 5, 20


def expert_counts(layout):
    if layout.startswith("even"):
        return [int(layout.split("-")[1])] * EXPERTS
    generator = torch.Generator(device="cuda").manual_seed(5)
    tokens = 1024 * RANKS * RANKS
    ids = torch.rand(tokens, GLOBAL_EXPERTS, device="cuda", generator=generator).topk(TOPK, dim=1).indices
    return torch.bincount(ids.flatten(), minlength=GLOBAL_EXPERTS)[:EXPERTS].tolist()


def grouped_layout(counts):
    indices = []
    for expert, count in enumerate(counts):
        indices += [expert] * count + [-1] * (-count % ALIGN)
    return torch.tensor(indices, device="cuda", dtype=torch.int32)


def gpu_state():
    return torch.cuda.clock_rate(), torch.cuda.power_draw() / 1000


def time_call(call):
    call()
    torch.cuda.synchronize()
    deadline = time.perf_counter() + HEAT_SECONDS
    while time.perf_counter() < deadline:
        for _ in range(20):
            call()
        torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(REPS):
            call()
    graph.replay()
    torch.cuda.synchronize()
    clock_before, power_before = gpu_state()
    times = []
    for _ in range(REPLAYS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / REPS)
    clock_after, power_after = gpu_state()
    times.sort()
    return {"median_us": statistics.median(times), "p10_us": times[len(times) // 10], "p90_us": times[(len(times) * 9) // 10], "sm_clock_mhz_before": clock_before, "sm_clock_mhz_after": clock_after, "power_w_before": power_before, "power_w_after": power_after}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.cuda.set_device(0)
    names = sorted(name for name in dir(deep_gemm) if "grouped" in name or "align" in name)
    print(json.dumps({"device": torch.cuda.get_device_name(), "deep_gemm": deep_gemm.__file__, "grouped_and_alignment_api": names}), flush=True)
    assert hasattr(deep_gemm, "m_grouped_fp8_gemm_nt_contiguous"), names
    counts = {layout: expert_counts(layout) for layout in LAYOUTS}
    indices = {layout: grouped_layout(counts[layout]) for layout in LAYOUTS}
    rows = {}
    out = open(args.output, "w")  # noqa: SIM115 - closed at exit

    def emit(row):
        print(json.dumps(row), flush=True)
        out.write(json.dumps(row) + "\n")
        out.flush()

    for gemm, n, k in GEMMS:
        generator = torch.Generator(device="cuda").manual_seed(713)
        b = (torch.randn(EXPERTS, n, k, device="cuda", generator=generator) * 4).to(torch.float8_e4m3fn)
        sfb = torch.rand(EXPERTS, n // 128, k // 128, device="cuda", generator=generator) * 0.005 + 0.005
        for layout in LAYOUTS:
            padded = indices[layout].numel()
            useful = sum(counts[layout])
            a = torch.randn(padded, k, device="cuda", generator=generator).to(torch.float8_e4m3fn)
            sfa = (torch.rand(k // 128, padded, device="cuda", generator=generator) * 0.005 + 0.005).T
            d = torch.empty(padded, n, device="cuda", dtype=torch.bfloat16)

            def call(a=a, sfa=sfa, b=b, sfb=sfb, d=d, indices=indices[layout]):
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous((a, sfa), (b, sfb), d, indices)

            timing = time_call(call)
            assert torch.isfinite(d[indices[layout] >= 0]).all(), (gemm, layout)
            flops = 2 * padded * n * k
            row = {"case": f"dgemm-{gemm}-{layout}", "variant": "image", **timing, "rows_padded": padded, "rows_useful": useful, "n": n, "k": k, "flops": flops, "tflops": flops / timing["median_us"] / 1e6, "tflops_useful": flops * useful / padded / timing["median_us"] / 1e6}
            rows.setdefault(layout, []).append(row)
            emit(row)
    for layout, parts in rows.items():
        total_us, flops, useful = sum(r["median_us"] for r in parts), sum(r["flops"] for r in parts), sum(r["flops"] * r["rows_useful"] / r["rows_padded"] for r in parts)
        emit({"case": f"dgemm-fc1+fc2-{layout}", "variant": "image", "median_us": total_us, "p10_us": sum(r["p10_us"] for r in parts), "p90_us": sum(r["p90_us"] for r in parts), "sm_clock_mhz_before": parts[0]["sm_clock_mhz_before"], "sm_clock_mhz_after": parts[-1]["sm_clock_mhz_after"], "power_w_before": parts[0]["power_w_before"], "power_w_after": parts[-1]["power_w_after"], "rows_padded": parts[0]["rows_padded"], "rows_useful": parts[0]["rows_useful"], "flops": flops, "tflops": flops / total_us / 1e6, "tflops_useful": useful / total_us / 1e6})


if __name__ == "__main__":
    main()
