"""Time the Qwen3 attention-side SGLang kernels at Firmus serving shapes.

--select-decode-gate runs only a four-GPU, pre-serving numeric check and an
ABAB gate+router comparison, writing one JSON selection artifact. It does not
run the standalone attention/dense/MegaMoE benchmark suite.

Run by the 1-node kernel bench for the image's SGLang tree and for the patched
tree (patches/sglang-bench/*.patch) with the same case set; each JSON line is
one timing row {case, variant, median_us, p10_us, p90_us, output_sha256}
(scripts/summarize-megamoe-kernel-bench.py pairs rows by case). A case times
the production operation sequence of its tree: the patched tree's fused kernel
where it exists, otherwise the unfused sequence it replaces. GPU time per call
comes from a CUDA graph of REPS back-to-back calls over rotating input sets
(Prefill-sized inputs do not stay in L2); median/p10/p90 are over replays.
The hash covers the outputs of one fresh call. Floating-point reordering is
allowed, so a hash difference is information; the numeric checks are in
check-firmus-decode-kernels.py.

Shapes: Prefill 3986/4027/4096 tokens per DP rank, Decode verify 4..144 rows
(EAGLE3 4 tokens per request, up to 36 requests). The FA3 TARGET_VERIFY
runtime knobs, request partitions and existing FlashInfer/Triton alternatives
use the same helper on both trees. Their rows carry full-output differences
from the default call. Timing is per call; layers_per_call=94 marks the paired
step probes. FlashInfer planning and page/token-index construction are excluded.

Decode dense selection runs before serving via select-firmus-decode-dense.py.
"""
import argparse
import hashlib
import inspect
import io
import json
import math
import os
import runpy
import statistics
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from runpy import run_path

import sglang.kernels.ops.attention.rope as rope_ops
import torch
from sgl_kernel import fused_add_rmsnorm
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.moe.topk import _mask_topk_ids_padded_region, fused_topk

try:
    from sglang.kernels.ops.layernorm.add_rmsnorm_group_quant import (
        add_rmsnorm_group_quant_fp8,
    )
except ImportError:
    add_rmsnorm_group_quant_fp8 = None
try:
    from sglang.kernels.ops.moe.trtllm_lora_temp.topk_softmax_pack import (
        topk_softmax_masked,
    )
except ImportError:
    topk_softmax_masked = None
HAS_FUSED_KV = hasattr(rope_ops, "apply_qknorm_rope_fp8_kv")

REPS = 20
REPLAYS = 25
GATE_SELECTION_PAIRS = 20
ROWS = (4027, 3986, 4096, 144, 140, 136, 132, 129, 128, 126, 16, 4)
TOPK_ROWS = (144, 140, 136, 132, 129, 128, 126, 16, 4)
SLOTS = 8192 + 64
EPS = 1e-6


def sha256(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def time_graph(calls):
    """GPU microseconds per call of a graph cycling through `calls`."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for call in calls:
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    repetitions = max(REPS, len(calls))
    with torch.cuda.graph(graph):
        for rep in range(repetitions):
            calls[rep % len(calls)]()
    for _ in range(3):
        graph.replay()
    times = []
    for _ in range(REPLAYS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / repetitions)
    times.sort()
    return {"median_us": statistics.median(times), "p10_us": times[len(times) // 10], "p90_us": times[(len(times) * 9) // 10]}


def input_sets(rows):
    return 4 if rows >= 1024 else 2


class Writer:
    def __init__(self, path, variant):
        self.file = open(path, "w")  # noqa: SIM115 - closed at exit
        self.variant = variant

    def __call__(self, case, sha, timing, **extra):
        line = json.dumps({"case": case, "variant": self.variant, **timing, "output_sha256": sha, **extra})
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()


def cos_sin_cache():
    angles = torch.arange(8192, device="cuda").float()[:, None] * (1000000.0 ** (-torch.arange(0, 128, 2, device="cuda").float() / 128))
    return torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()


def bench_rope(write):
    cache = cos_sin_cache()
    gen = torch.Generator(device="cuda").manual_seed(4401)
    qw = (torch.randn(128, device="cuda", generator=gen) * 0.25 + 1).bfloat16()
    kw = (torch.randn(128, device="cuda", generator=gen) * 0.25 + 1).bfloat16()

    def inputs(rows, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        qkv = torch.randn(rows, 72 * 128, device="cuda", dtype=torch.bfloat16, generator=g) * 4
        positions = torch.randint(0, 8192, (rows,), device="cuda", dtype=torch.int64, generator=g)
        loc = torch.randperm(SLOTS - 64, device="cuda", generator=g)[:rows] + 64
        return qkv, positions, loc

    def split(qkv):
        q, k, v = qkv.split([8192, 512, 512], dim=-1)
        return q.view(-1, 64, 128), k.view(-1, 4, 128), v.view(-1, 4, 128)

    def rope_only(qkv, positions, loc, q_fp8, k_cache, v_cache):
        q, k, _ = split(qkv)
        rope_ops.apply_qknorm_rope_fp8_q(q, k, qw, kw, cache, positions, EPS, q_fp8)

    def prologue(qkv, positions, loc, q_fp8, k_cache, v_cache):
        # QK-norm/RoPE/FP8 Q plus the FP8 KV-cache write that set_kv_buffer does.
        q, k, v = split(qkv)
        if HAS_FUSED_KV:
            rope_ops.apply_qknorm_rope_fp8_kv(q, k, qw, kw, cache, positions, EPS, q_fp8, v, k_cache.view(torch.float8_e4m3fn), v_cache.view(torch.float8_e4m3fn), loc)
            return
        rope_ops.apply_qknorm_rope_fp8_q(q, k, qw, kw, cache, positions, EPS, q_fp8)
        cache_k, cache_v = k.to(torch.float8_e4m3fn).view(torch.uint8), v.to(torch.float8_e4m3fn).view(torch.uint8)
        store_cache(cache_k.reshape(-1, 512), cache_v.reshape(-1, 512), k_cache.view(-1, 512), v_cache.view(-1, 512), loc, row_bytes=512, v_row_bytes=512, size_limit=SLOTS)

    for name, op in (("qknorm-rope-fp8", rope_only), ("attn-prologue-kv", prologue)):
        for rows in ROWS:
            q_fp8 = torch.empty(rows, 64, 128, device="cuda", dtype=torch.float8_e4m3fn)
            k_cache = torch.zeros(SLOTS, 4, 128, device="cuda", dtype=torch.uint8)
            v_cache = torch.zeros_like(k_cache)
            qkv, positions, loc = inputs(rows, 7)
            op(qkv, positions, loc, q_fp8, k_cache, v_cache)
            sha = sha256(q_fp8, qkv, k_cache, v_cache)
            sets = [inputs(rows, 100 + i) for i in range(input_sets(rows))]
            timing = time_graph([lambda s=s, op=op, out=(q_fp8, k_cache, v_cache): op(*s, *out) for s in sets])
            write(f"{name}-{rows}", sha, timing, calls_per_decode_step=94 if name == "attn-prologue-kv" and rows <= 144 else None)


def bench_norm_quant(write):
    gen = torch.Generator(device="cuda").manual_seed(9031)
    weight = (torch.randn(4096, device="cuda", generator=gen) * 0.1 + 1).bfloat16()

    def inputs(rows, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16, generator=g) * 2, torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16, generator=g) * 8

    def op(x, residual):
        # input_layernorm (fused add) + the QKV GEMM's group-128 FP8 quant.
        if add_rmsnorm_group_quant_fp8 is not None:
            return add_rmsnorm_group_quant_fp8(x, residual, weight, EPS)
        fused_add_rmsnorm(x, residual, weight, EPS)
        return sglang_per_token_group_quant_fp8(x, 128, column_major_scales=True, scale_tma_aligned=True)

    for rows in ROWS:
        x, residual = inputs(rows, 11)
        q, scale = op(x, residual)
        sha = sha256(q, scale, residual)
        sets = [inputs(rows, 200 + i) for i in range(input_sets(rows))]
        write(f"input-norm-quant-{rows}", sha, time_graph([lambda s=s: op(*s) for s in sets]))


def bench_topk(write):
    for rows in TOPK_ROWS:
        gen = torch.Generator(device="cuda").manual_seed(rows)
        hidden = torch.empty(rows, 4096, device="cuda", dtype=torch.bfloat16)
        logits = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16, generator=gen) * 4
        count = torch.tensor(rows - 2, device="cuda", dtype=torch.int32)

        def op(hidden=hidden, logits=logits, count=count):
            # Decode router: top-8 softmax plus id=-1 on CUDA-graph padding rows.
            if topk_softmax_masked is not None:
                return fused_topk(hidden, logits, 8, True, num_token_non_padded=count)
            weights, ids = fused_topk(hidden, logits, 8, True)
            _mask_topk_ids_padded_region(ids, count)
            return weights, ids

        write(f"topk-mask-{rows}", sha256(*op()), time_graph([op]))


def bench_decode_router(write):
    """Production JIT router + zero bias + padding mask, before/after fusion.

    One call per layer (94 per target verify); all timed calls retain the
    full top-8 arithmetic. Candidate fusion is also compared with its own
    unfused path so the other three serving overlays cannot confound it.
    """
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    fused = "num_token_non_padded" in inspect.signature(moe_fused_gate).parameters
    for rows in TOPK_ROWS:
        gen = torch.Generator(device="cuda").manual_seed(61000 + rows)
        logits = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16, generator=gen) * 4
        count = torch.tensor(max(0, rows - 3), device="cuda", dtype=torch.int32)

        def unfused(logits=logits, count=count):
            # Match fused_topk's allocation inside every layer invocation.
            bias = torch.zeros(128, device="cuda", dtype=torch.float32)
            weights, ids = moe_fused_gate(logits, bias, 8, scoring_func="softmax", renormalize=True)
            _mask_topk_ids_padded_region(ids, count)
            return weights, ids

        def production(logits=logits, count=count, unfused=unfused):
            if fused:
                return moe_fused_gate(logits, None, 8, scoring_func="softmax", renormalize=True, num_token_non_padded=count)
            return unfused()

        for name, op in (("decode-router-unfused", unfused), ("decode-router-production", production)):
            write(f"{name}-{rows}", sha256(*op()), time_graph([op]), calls_per_decode_step=94, fused_padding=fused and name.endswith("production"))


def bench_decode_gate(write):
    """Isolate GEMM and GEMM+router, including a 94-layer weight working set."""
    import importlib.util

    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    if importlib.util.find_spec("sglang.kernels.ops.moe.decode_gate") is not None:
        from sglang.kernels.ops.moe.decode_gate import decode_gate
    else:
        decode_gate = torch.nn.functional.linear
    fused_padding = "num_token_non_padded" in inspect.signature(moe_fused_gate).parameters
    for rows in (64, 96, 128, 132, 144):
        for layers in (1, 94):
            gen = torch.Generator(device="cuda").manual_seed(61023 + rows + layers)
            inputs = [(torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16, generator=gen), torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16, generator=gen) / 64) for _ in range(layers)]
            count = torch.tensor(rows - 3, device="cuda", dtype=torch.int32)

            def route(logits, count=count):
                if fused_padding:
                    return moe_fused_gate(logits, None, 8, scoring_func="softmax", renormalize=True, num_token_non_padded=count)
                bias = torch.zeros(128, device="cuda", dtype=torch.float32)
                weights, ids = moe_fused_gate(logits, bias, 8, scoring_func="softmax", renormalize=True)
                _mask_topk_ids_padded_region(ids, count)
                return weights, ids

            for kind, gate in (("cublas", torch.nn.functional.linear), ("production", decode_gate)):
                for with_router in (False, True):
                    def op(hidden, weight, gate=gate, with_router=with_router, route=route):
                        logits = gate(hidden, weight)
                        return route(logits) if with_router else (logits,)

                    calls = [lambda hidden=h, weight=w: op(hidden, weight) for h, w in inputs]
                    suffix = "router" if with_router else "gemm"
                    write(f"decode-gate-{suffix}-{kind}-{rows}-layers{layers}", sha256(*calls[0]()), time_graph(calls), calls_per_decode_step=94, weight_working_set_bytes=layers * 128 * 4096 * 2, layers_in_graph=layers, full_k=4096, accumulation="fp32", logits_dtype="bf16")


def bench_fa3_verify(write):
    """Compare every verify row on identical FP8 bytes, including Q casts.

    FlashInfer planning and token/page index construction are excluded from
    per-layer timings; those are step-level costs, not a serving speed claim.
    """
    helper = Path(__file__).resolve().parents[1] / "overlay/sglang/python/sglang/kernels/ops/attention/verify_attention_candidates.py"
    ops = runpy.run_path(str(helper))
    for batch in ops["BATCHES"]:
        common = ops["make_inputs"](batch)
        cases = ops["fa3_cases"](common)
        cases.update(ops["alternative_cases"](common))
        cases.update(ops["step_cases"](common))
        reference = cases["fa3-verify-default"]().clone()
        for name, call in cases.items():
            output = call()
            error = output.float() - reference.float()
            assert output.dtype == torch.bfloat16 and output.isfinite().all(), name
            write(f"{name}-b{batch}", sha256(output), time_graph([call]), max_abs_diff_vs_default=error.abs().max().item(), relative_l2_vs_default=(error.norm() / reference.float().norm().clamp_min(1e-12)).item(), kv_dtype=str(common["k_cache"].dtype), planning_included=False, layers_per_call=94 if "step94" in name else 1)


def gate_timing_summary(samples):
    """CUDA-event microseconds for a complete 94-layer gate+router replay."""
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("gate timing samples must be finite and positive")
    ordered = sorted(samples)
    return {"median_us": statistics.median(ordered), "p10_us": ordered[len(ordered) // 10], "p90_us": ordered[len(ordered) * 9 // 10]}


def select_decode_gate(ranks):
    """Select one common worker setting only when every GPU clears the rule."""
    result = {"selected": "cublas", "env": {}, "rule": {"min_median_savings_us_per_94_layers": 200.0, "both_blocks_faster": True, "gpus": 4, "blocks": 2, "pairs_per_block": GATE_SELECTION_PAIRS}, "reason": "incomplete_rank_results", "rank_savings_us": []}
    if len(ranks) != 4 or {rank.get("rank") for rank in ranks} != set(range(4)):
        return result
    if any(rank.get("numeric", {}).get("passed") is not True for rank in ranks):
        result["reason"] = "numeric_check_failed_or_skipped"
        return result
    for rank in sorted(ranks, key=lambda rank: rank["rank"]):
        blocks = rank.get("blocks", [])
        if rank.get("error") or len(blocks) != 2:
            result["reason"] = "timing_failed_or_incomplete"
            return result
        pooled = {name: [] for name in ("cublas", "triton")}
        block_savings = []
        try:
            for block in blocks:
                medians = {}
                for name, pooled_values in pooled.items():
                    values = block[name]["samples_us"]
                    if len(values) != GATE_SELECTION_PAIRS:
                        raise ValueError("incomplete block")
                    medians[name] = gate_timing_summary(values)["median_us"]
                    pooled_values.extend(values)
                block_savings.append(medians["cublas"] - medians["triton"])
            savings = gate_timing_summary(pooled["cublas"])["median_us"] - gate_timing_summary(pooled["triton"])["median_us"]
        except (KeyError, TypeError, ValueError):
            result["reason"] = "invalid_timing_samples"
            return result
        result["rank_savings_us"].append({"rank": rank["rank"], "pooled_median_savings_us": savings, "block_median_savings_us": block_savings})
    if any(any(value <= 0 for value in rank["block_median_savings_us"]) for rank in result["rank_savings_us"]):
        result["reason"] = "not_faster_in_both_blocks_on_every_gpu"
    elif any(rank["pooled_median_savings_us"] < 200.0 for rank in result["rank_savings_us"]):
        result["reason"] = "less_than_200us_on_at_least_one_gpu"
    else:
        result.update(selected="triton", env={"SGLANG_DECODE_GATE_GEMM": "triton"}, reason="all_gpus_passed_numeric_and_timing_rules")
    return result


def check_gate_for_selection():
    """Run exactly the gate cases, treating a SKIP as a failed selection gate."""
    log = io.StringIO()
    try:
        with redirect_stdout(log):
            checks = run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
            passed = checks["check_decode_gate"]() is True
        return {"passed": passed, "log": log.getvalue()}
    except Exception:  # noqa: BLE001 - any numeric failure must disable the candidate and retain its traceback
        return {"passed": False, "log": log.getvalue(), "error": traceback.format_exc()}


def measure_gate_selection_blocks(graphs):
    """ABAB on each GPU in the same process, without collectives in the loop."""
    for _ in range(3):
        for name in ("cublas", "triton"):
            graphs[name].replay()
    torch.cuda.synchronize()
    blocks = []
    for block_id in range(2):
        samples = {name: [] for name in graphs}
        for _ in range(GATE_SELECTION_PAIRS):
            for name in ("cublas", "triton"):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000)
        blocks.append({"block": block_id, "order": "AB" * GATE_SELECTION_PAIRS, **{name: {"samples_us": values, **gate_timing_summary(values)} for name, values in samples.items()}})
    return blocks


def measure_decode_gate_selection():
    """M128, identical 94 distinct BF16 layer weights for cuBLAS and Triton."""
    from sglang.kernels.ops.moe.decode_gate import decode_gate
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    gen = torch.Generator(device="cuda").manual_seed(61023)
    inputs = [(torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16, generator=gen), torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16, generator=gen) / 64) for _ in range(94)]
    count = torch.tensor(128, device="cuda", dtype=torch.int32)
    graphs = {}
    # Full unpadded M128 is the production Decode target-verify bucket.
    # Both implementations use the exact same B6 router, including count load.
    for name, gate in (("cublas", torch.nn.functional.linear), ("triton", decode_gate)):
        def step(gate=gate):
            for hidden, weight in inputs:
                logits = gate(hidden, weight)
                moe_fused_gate(logits, None, 8, scoring_func="softmax", renormalize=True, num_token_non_padded=count)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        graphs[name] = graph
    blocks = measure_gate_selection_blocks(graphs)
    pooled = {name: gate_timing_summary([value for block in blocks for value in block[name]["samples_us"]]) for name in graphs}
    fluctuation = {}
    for name in graphs:
        first, second = [block[name]["median_us"] for block in blocks]
        fluctuation[name] = {"block1_minus_block0_median_us": second - first, "relative_median_change": second / first - 1}
    return {"blocks": blocks, "pooled": pooled, "block_fluctuation": fluctuation}


def run_decode_gate_selection(output):
    """torchrun on the allocated Decode four GPUs; rank0 writes one artifact."""
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise ValueError("--select-decode-gate requires torchrun --nproc-per-node=4")
    started = time.monotonic()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        local = {"rank": rank, "local_rank": int(os.environ["LOCAL_RANK"]), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": torch.cuda.get_device_name(), "numeric": check_gate_for_selection()}
        ranks = [None] * 4
        dist.all_gather_object(ranks, local)
        # Every rank makes the same decision before entering any timing barriers.
        if all(item["numeric"]["passed"] for item in ranks):
            dist.barrier()
            try:
                local.update(measure_decode_gate_selection())
            except Exception:  # noqa: BLE001 - a failed comparison cannot enable a worker optimization
                local["error"] = traceback.format_exc()
            dist.all_gather_object(ranks, local)
        if rank == 0:
            result = {"schema": "decode-gate-selection/v1", **select_decode_gate(ranks), "shape": {"m": 128, "n": 128, "k": 4096, "layers": 94, "weight_bytes": 94 * 128 * 4096 * 2, "dtype": "bf16", "accumulation": "fp32"}, "timing_unit": "us_per_94_layers_gate_plus_router", "ranks": ranks, "elapsed_seconds": time.monotonic() - started}
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"selected": result["selected"], "env": result["env"], "reason": result["reason"], "output": str(path)}), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=("image", "candidate"))
    parser.add_argument("--select-decode-gate", action="store_true", help="in-job four-GPU numeric gate and paired selection; write one JSON artifact")
    args = parser.parse_args()
    if args.select_decode_gate:
        if args.variant is not None:
            parser.error("--variant is only for the standalone kernel bench")
        run_decode_gate_selection(args.output)
        return
    if args.variant is None:
        parser.error("--variant is required for the standalone kernel bench")
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.cuda.set_device(0)
    print(json.dumps({"device": torch.cuda.get_device_name(), "fused_kv": HAS_FUSED_KV, "fused_norm_quant": add_rmsnorm_group_quant_fp8 is not None, "masked_topk": topk_softmax_masked is not None}), flush=True)
    write = Writer(args.output, args.variant)
    bench_rope(write)
    bench_norm_quant(write)
    bench_topk(write)
    bench_decode_router(write)
    bench_decode_gate(write)
    bench_fa3_verify(write)


if __name__ == "__main__":
    main()
