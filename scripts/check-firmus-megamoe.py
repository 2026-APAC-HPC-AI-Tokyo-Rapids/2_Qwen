"""Finite numeric gate inside the benchmark's existing 4-GPU Slurm step.

No scheduler submission. Compare the native kernel with independent FP32 GEMMs,
the same FP8 activation groups, and late top-k weights with FP32 accumulation.
This local numeric screen is not the organisers' benchmark accuracy check;
Q1/Q4 require at least 99% of their reference score before official acceptance.
"""
import json
import os
from datetime import timedelta

import deep_gemm
import torch
import torch.distributed as dist
from firmus_megamoe_replicas import check_replicas


def quantized(x, group):
    blocks = x.float().reshape(x.shape[0], -1, group)
    scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape)


def dequantized(w, scale):
    return w.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    from pathlib import Path
    from runpy import run_path

    run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")), run_name="__main__")
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=240), device_id=torch.device("cuda", rank))
    assert dist.get_world_size() == 4
    torch.manual_seed(713 + rank)
    device = torch.device("cuda", rank)
    # Exactly Qwen's local EP4 shapes, not a smaller substitute kernel.
    w13 = (torch.randn(32, 3072, 4096, dtype=torch.bfloat16, device=device) * 4).to(torch.float8_e4m3fn)
    w2 = (torch.randn(32, 4096, 1536, dtype=torch.bfloat16, device=device) * 4).to(torch.float8_e4m3fn)
    s13 = torch.rand(32, 24, 32, device=device) * 0.005 + 0.005
    s2 = torch.rand(32, 32, 12, device=device) * 0.005 + 0.005
    l1, l2 = deep_gemm.transform_weights_for_mega_moe_sm90((w13, s13), (w2, s2))
    buf = deep_gemm.get_symm_buffer_for_mega_moe(dist.group.WORLD, 128, 4096, 8, 4096, 1536, use_fp8_dispatch=True, activation="swiglu")
    cases = [("cross-rank-empty-tail", [7, 0, 65, 129]), ("capacity-hot-experts", [4096] * 4), ("decode-graph", [128, 120, 4, 4]), ("decode-quant-zero", [9, 0, 0, 0])]
    # Exercise all SwapAB widths, plus a full tile and tail.
    # Other ranks still participate in the collective with zero live tokens.
    cases.extend((f"decode-tile-{count}", [count, 0, 0, 0]) for count in (1, 9, 17, 25, 33, 41, 49, 57, 65))
    # megamoe-idle-prefill-geometry.patch: an idle Prefill rank passes 256 masked
    # output rows so DeepGEMM picks the Prefill geometry for the other ranks' tokens.
    idle_prefill_rows = os.environ.get("SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY", "0") == "1"
    if idle_prefill_rows:
        cases.append(("idle-prefill-rows", [4096, 4096, 4096, 0]))
    for case, counts in cases:
        patterns = torch.randn(4, 4096, dtype=torch.bfloat16, device=device) * 0.25
        if case == "decode-quant-zero":
            patterns.zero_()
        ids = torch.stack([torch.randperm(128, device=device)[:8] for _ in range(4)]).to(torch.int32)
        if case in ("capacity-hot-experts", "decode-quant-zero") or case.startswith("decode-tile-"):
            ids[:] = torch.arange(8, dtype=torch.int32, device=device)
        weights = torch.rand(4, 8, device=device)
        ids[0, -1] = -1  # Masked/padded top-k slot must not read an expert.
        weights /= weights.sum(-1, keepdim=True)
        gathered = []
        for value in (patterns, ids, weights):
            all_ranks = [torch.empty_like(value) for _ in range(4)]
            dist.all_gather(all_ranks, value)
            gathered.append(torch.cat(all_ranks))
        all_x, all_ids, all_weights = gathered
        x = quantized(all_x, 128)
        # FC2 activation scale granularity of the kernel geometry under test: the
        # Prefill geometry (host picks it from 256 rows per rank; idle Prefill ranks
        # pass 256 masked rows) quantizes the SwiGLU output per 128 K, Decode per 64.
        # The error bounds below are unchanged.
        prefill_geometry = max(counts) >= 256
        l2_group = 128 if prefill_geometry else 64
        reference = torch.zeros(16, 4096, device=device)
        for local_expert in range(32):
            rows, slots = torch.where(all_ids == rank * 32 + local_expert)
            if rows.numel() == 0:
                continue
            gate, up = (x[rows] @ dequantized(w13[local_expert], s13[local_expert]).T).chunk(2, dim=-1)
            activation = torch.nn.functional.silu(gate) * up
            out = quantized(activation, l2_group) @ dequantized(w2[local_expert], s2[local_expert]).T
            # Native FC2 stores BF16 terms, then multiplies and sums in FP32.
            reference.index_add_(0, rows, out.to(torch.bfloat16).float() * all_weights[rows, slots, None])
        dist.all_reduce(reference)
        count = counts[rank]
        indices = torch.arange(count, device=device) % 4
        hidden, top_ids, top_weights = patterns[indices].contiguous(), ids[indices].contiguous(), weights[indices].contiguous()
        rows = 256 if case == "idle-prefill-rows" and count == 0 else max(count, 1)
        actual = torch.empty(rows, 4096, dtype=torch.bfloat16, device=device)

        def forward(hidden, top_ids, top_weights, count, actual):
            deep_gemm.mega_moe_pre_dispatch_sm90(hidden, top_ids, top_weights, buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights, num_tokens=count, group_size=128, routed_scaling_factor=1.0)
            deep_gemm.fp8_mega_moe(actual, l1, l2, buf, recipe=(128, 128, 128), activation="swiglu", fast_math=True)

        forward(hidden, top_ids, top_weights, count, actual)
        torch.cuda.synchronize()
        graph = None
        if case == "decode-graph":
            # An idle EAGLE3 rank replays the minimum one-request graph (four
            # token rows), with every route masked, rather than a zero-row graph.
            # Replays change GPU input data without changing its addresses.
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(graph):
                forward(hidden, top_ids, top_weights, count, actual)
        for replay in range(4 if graph is not None else 1):
            if graph is not None:
                indices = (torch.arange(count, device=device) + replay) % 4
                hidden.copy_(patterns[indices])
                top_ids.copy_(ids[indices])
                top_weights.copy_(weights[indices])
                # CUDA TopK masks IDs but leaves weights nonzero. Rank 2 also
                # alternates idle/live to expose stale routes across replays.
                live = (0 if replay % 2 == 0 else count) if rank == 2 else max(0, count - replay * 7)
                top_ids[live:] = -1
                graph.replay()
            else:
                live = count
            torch.cuda.synchronize()
            observed = actual[:count].float()
            expected = reference[rank * 4 + indices].to(torch.bfloat16).float()
            expected[live:] = 0
            assert torch.isfinite(observed).all(), (case, rank, "non-finite output")
            relative_l2 = ((observed - expected).norm() / expected.norm().clamp_min(1e-10)).item()
            max_scaled_error = ((observed - expected).abs().max() / expected.abs().max().clamp_min(1e-10)).item() if count else 0.0
            print(json.dumps({"case": case, "replay": replay, "rank": rank, "tokens": count, "live_tokens": live, "relative_l2": relative_l2, "max_scaled_error": max_scaled_error, "workspace_bytes": buf.buffer.numel()}), flush=True)
            # Same tolerances as the validated prefill kernel; no missing or
            # double routing weights, stale graph inputs, or misplaced scales.
            assert relative_l2 < 0.025 and max_scaled_error < 0.08, (case, rank, relative_l2, max_scaled_error)
            if case == "idle-prefill-rows" and count == 0:
                # Every masked row dispatches nothing and combines to zero.
                assert torch.count_nonzero(actual).item() == 0, (case, rank, "masked idle rows must combine to zero")
            dist.barrier()
    if rank == 0:
        print("PASS: SM90 MegaMoE EP4 numerical gate; original FP8 scales, late FP32 combine, changing masked Decode graph inputs", flush=True)
    # Synthetic representative shapes, not a replay of measured expert routes.
    # Record all-rank latency separately from the official serving score.
    for label, count in (("decode-balanced", 128), ("prefill-balanced", 4096)):
        hidden = torch.randn(count, 4096, dtype=torch.bfloat16, device=device) * 0.25
        top_weights, top_ids = torch.rand(count, 128, device=device).topk(8, dim=1)
        top_ids = top_ids.to(torch.int32)
        top_weights /= top_weights.sum(-1, keepdim=True)
        actual = torch.empty_like(hidden)
        for _ in range(5):
            forward(hidden, top_ids, top_weights, count, actual)
        torch.cuda.synchronize()
        graph = None
        if count == 128:
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(graph):
                forward(hidden, top_ids, top_weights, count, actual)
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(20)]
        dist.barrier()
        for start, end in events:
            start.record()
            if graph is not None:
                graph.replay()
            else:
                forward(hidden, top_ids, top_weights, count, actual)
            end.record()
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all(), (label, rank)
        durations = torch.tensor([start.elapsed_time(end) for start, end in events], device=device)
        dist.all_reduce(durations, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(json.dumps({"benchmark": label, "tokens_per_rank": count, "cuda_graph": graph is not None, "max_rank_ms": durations.tolist()}), flush=True)
    if idle_prefill_rows:
        # Same collective with rank 3 idle: 1 output row (Decode SwapAB geometry
        # on the idle rank, current serving) against 256 masked rows (patch).
        busy = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device) * 0.25
        busy_weights, busy_ids = torch.rand(4096, 128, device=device).topk(8, dim=1)
        busy_ids = busy_ids.to(torch.int32)
        busy_weights /= busy_weights.sum(-1, keepdim=True)
        for label, idle_rows in (("prefill-one-idle-1row", 1), ("prefill-one-idle-256rows", 256)):
            count = 0 if rank == 3 else 4096
            hidden, top_ids, top_weights = busy[:count].contiguous(), busy_ids[:count].contiguous(), busy_weights[:count].contiguous()
            actual = torch.empty(idle_rows if count == 0 else count, 4096, dtype=torch.bfloat16, device=device)
            for _ in range(5):
                forward(hidden, top_ids, top_weights, count, actual)
            torch.cuda.synchronize()
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(20)]
            dist.barrier()
            for start, end in events:
                start.record()
                forward(hidden, top_ids, top_weights, count, actual)
                end.record()
            torch.cuda.synchronize()
            durations = torch.tensor([start.elapsed_time(end) for start, end in events], device=device)
            dist.all_reduce(durations, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(json.dumps({"benchmark": label, "tokens_per_rank": [4096, 4096, 4096, 0], "idle_output_rows": idle_rows, "max_rank_ms": durations.tolist()}), flush=True)
    check_replicas(quantized, dequantized)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
