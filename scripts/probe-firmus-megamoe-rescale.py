"""Diagnostic upper bound on the scale path; never a serving selection gate.

Production uses deep_gemm.fp8_mega_moe unchanged. Only this script calls the
separate host diagnostic symbol. Imports of CUDA packages are worker-only.
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

SCHEMA = "megamoe-rescale-exposure/v1"
FLAG = "DG_R41_DIAGNOSTIC_NO_RESCALE"
SYMBOL = "sm90_fp8_mega_moe_rescale_exposure_diagnostic_impl"
PREFIX = f"#define {FLAG} 1\n#define sm90_fp8_mega_moe_impl {SYMBOL}\n"
TEMPLATE = "8448,4096,1536,128,8,128,128,256,128,92160,1474560,4,64,64,256,132,1,cute::numeric_limits<float>::infinity(),true,0,true,true,false,false,false,true,false,true"
REQUIRED = {"DG_SM90_PREFILL_LOCAL_EP1": "1", "DG_SM90_PREFILL_2WG": "1", "DG_SM90_PREFILL_4STAGE": "1", "DG_SM90_PREFILL_LEAN": "1", "DG_SM90_PREFILL_SPLIT_ACCUM": "1", "DG_SM90_PREFILL_FC1_FIRST_K": "1"}
FIXTURE_SHA = "1907184191e49a6dcc1431cdadafe9bfef2eb90bf4ad3f58407f6ed32a136b52"
ROUNDS, CALLS = 12, 8
STEADY_STEPS = 292358


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def assert_serving_environment(environment):
    assert FLAG not in environment, "Diagnostic flag must never be in the serving environment"
    assert all(environment.get(k) == v for k, v in REQUIRED.items()), "Production Prefill flags required"
    assert environment.get("DG_SM90_PREFILL_TAIL_PAIR", "0") == "0"
    # These compiler overrides would invalidate the same-image/default-flags claim.
    assert not environment.get("DG_JIT_NVCC_COMPILER")
    assert environment.get("DG_JIT_USE_NVRTC", "0") == "0"


def assert_jit_sources(production, diagnostic):
    """Validate actual host-generated cache sources, not a reconstructed TU."""
    # Both are built by the same Compiler object with identical flags. The cache
    # source includes the recursive header hash and the actual host template args.
    def body(text):
        lines = text.splitlines(keepends=True)
        assert lines[0].startswith("// Includes' hash value: ")
        return lines[0], "".join(lines[1:])
    p_hash, prod = body(production)
    d_hash, diag = body(diagnostic)
    assert p_hash == d_hash
    assert FLAG not in prod and SYMBOL not in prod
    assert "#define DG_SM90_PREFILL_FC1_FIRST_K 1\n" in prod
    assert diag == PREFIX + prod, "Diagnostic must differ only by its explicit compile-time flag/symbol"
    match = re.search(r"&sm90_fp8_mega_moe_impl<(.+?)>\);", prod, re.DOTALL)
    assert match and re.sub(r"\s+", "", match[1]) == TEMPLATE, "Unexpected production JIT geometry"


def cache_identity(cache, cuobjdump):
    variants = {}
    sources = {}
    for path in cache.glob("cache/kernel.sm90_fp8_mega_moe.*/kernel.cu"):
        source = path.read_text()
        variant = "diagnostic" if PREFIX in source else "production"
        assert variant not in variants, "More than one MegaMoE instantiation per variant"
        symbols = subprocess.check_output([str(cuobjdump), "-symbols", str(path.with_suffix(".cubin"))], text=True)
        names = [line.split()[-1] for line in symbols.splitlines() if line.startswith("STT_FUNC") and "STO_ENTRY" in line and "__instantiate_kernel" not in line]
        assert len(names) == 1
        assert (SYMBOL in names[0]) == (variant == "diagnostic")
        variants[variant] = {"jit_cache_key": path.parent.name, "kernel_name": names[0], "generated_source_sha256": hashlib.sha256(source.encode()).hexdigest()}
        sources[variant] = source
    assert set(variants) == {"production", "diagnostic"}
    assert_jit_sources(sources["production"], sources["diagnostic"])
    return variants


def summarize(samples):
    assert len(samples) == 2 * ROUNDS
    orders = {}
    for order in ("forward", "reverse"):
        rows = [r for r in samples if r["order"] == order]
        assert sorted(r["round"] for r in rows) == list(range(ROUNDS))
        for row in rows:
            assert all(math.isfinite(row[k]) and row[k] > 0 for k in ("production_us", "diagnostic_us"))
        orders[order] = statistics.median(r["production_us"] - r["diagnostic_us"] for r in rows)
    return {"order_median_delta_us": orders, "median_delta_us": statistics.median(r["production_us"] - r["diagnostic_us"] for r in samples)}


def aggregate(directory, run_id, image, codes):
    base = {"schema_version": SCHEMA, "run_id": run_id, "image_sha256": image, "status": "incomplete", "worker_exit_codes": codes, "interpretation": "upper bound on the whole scale path, including loads, reciprocal, multiply and dependency"}
    rows = []
    try:
        assert codes == [0, 0, 0], "Worker failure or timeout"
        for worker in range(3):
            for rank in range(4):
                row = json.loads((directory / f"worker-{worker}" / f"rank-{rank}.json").read_text())
                assert row["schema_version"] == SCHEMA and row["status"] == "complete"
                assert row["run_id"] == run_id and row["image_sha256"] == image
                assert row["worker"] == worker and row["rank"] == rank
                assert row["routing_sha256"] == FIXTURE_SHA
                assert_serving_environment(row["serving_environment"])
                assert set(row["jit"]) == {"production", "diagnostic"}
                for variant in row["jit"].values():
                    assert variant["jit_cache_key"] and variant["kernel_name"]
                assert SYMBOL not in row["jit"]["production"]["kernel_name"]
                assert SYMBOL in row["jit"]["diagnostic"]["kernel_name"]
                clocks = row["sm_clock_samples_mhz"]
                assert clocks and all(math.isfinite(c) and c > 0 for c in clocks)
                row_summary = summarize(row["samples"])
                row_summary.update(worker=worker, rank=rank, gpu_uuid=row["gpu_uuid"], sm_clock_mhz_mean=statistics.mean(clocks))
                row_summary["clocks_per_steady_k128_cta_step"] = row_summary["median_delta_us"] * 132 * statistics.mean(clocks) / STEADY_STEPS
                rows.append(row_summary)
        assert len({r["gpu_uuid"] for r in rows}) == 12, "Expected twelve distinct GPUs"
        delta = statistics.median(r["median_delta_us"] for r in rows)
        base.update(status="complete", gpus=rows, median_12gpu_delta_us=delta, decision="closed" if delta < 50 else "conditionally_open", threshold_us_per_layer=50, remaining_uncertainty="loads and reciprocal share is not separated; this is not an achievable redesign speedup")
    except (AssertionError, KeyError, ValueError, OSError, TypeError) as error:
        base["reason"] = str(error)
    return base


def run_worker(args, record):
    # Must execute before importing deep_gemm or creating any CUDA object.
    assert_serving_environment(os.environ)
    rank = int(os.environ["LOCAL_RANK"])
    cache = args.output / f"jit-rank-{rank}"
    assert not cache.exists(), "Probe requires a fresh, private JIT cache"
    os.environ["DG_JIT_CACHE_DIR"] = str(cache)
    from datetime import timedelta

    import deep_gemm
    import numpy as np
    import torch
    import torch.distributed as dist

    assert_serving_environment(os.environ)
    header = Path(deep_gemm.__file__).parent / "include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"
    header_source = header.read_text()
    assert header_source.count(f"if constexpr (!{FLAG}) final_accum[") == 8, "Runtime diagnostic overlay is missing"
    assert f"#ifndef {FLAG}\n#define {FLAG} 0\n#endif" in header_source
    record["runtime_header_sha256"] = hashlib.sha256(header.read_bytes()).hexdigest()
    assert torch.cuda.device_count() == 4
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    assert torch.cuda.get_device_properties(rank).multi_processor_count == 132
    dist.init_process_group("nccl", timeout=timedelta(seconds=70), device_id=torch.device("cuda", rank))
    assert dist.get_world_size() == 4
    singles = [dist.new_group([r]) for r in range(4)]
    group = singles[rank]
    generator = torch.Generator(device="cuda").manual_seed(1041)
    # Production weight shapes; BF16 temporary generation stays off the timed path.
    w13 = (torch.randn(128, 3072, 4096, device="cuda", dtype=torch.bfloat16, generator=generator) * 4).to(torch.float8_e4m3fn)
    w2 = (torch.randn(128, 4096, 1536, device="cuda", dtype=torch.bfloat16, generator=generator) * 4).to(torch.float8_e4m3fn)
    s13 = torch.rand(128, 24, 32, device="cuda", generator=generator) * 0.005 + 0.005
    s2 = torch.rand(128, 32, 12, device="cuda", generator=generator) * 0.005 + 0.005
    l1, l2 = deep_gemm.transform_weights_for_mega_moe_sm90((w13, s13), (w2, s2))
    buf = deep_gemm.get_symm_buffer_for_mega_moe(group, 128, 8192, 8, 4096, 1536, use_fp8_dispatch=True, activation="swiglu")
    assert buf.num_max_tokens_per_rank == 8448
    fixture = args.routing.read_bytes()
    assert hashlib.sha256(fixture).hexdigest() == FIXTURE_SHA
    with np.load(args.routing) as routing:
        ids = torch.as_tensor(routing["logical_ids"][:, 0, 0, :3994, :].reshape(7988, 8).copy(), device="cuda", dtype=torch.int32)
    hidden = torch.randn(7988, 4096, dtype=torch.bfloat16, device="cuda", generator=generator) * 0.25
    weights = torch.full((7988, 8), 1 / 8, dtype=torch.float32, device="cuda")
    out = torch.empty_like(hidden)
    deep_gemm.mega_moe_pre_dispatch_sm90(hidden, ids, weights, buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights, num_tokens=7988, group_size=128, routed_scaling_factor=1.0)
    production = deep_gemm.fp8_mega_moe
    # Normal serving wrapper must resolve to the unchanged normal host entry.
    assert production.__globals__["_C"] is deep_gemm._C
    assert "fp8_mega_moe" in production.__code__.co_names
    assert "fp8_mega_moe_rescale_exposure_diagnostic" not in production.__code__.co_names

    def call(variant):
        if variant == "production":
            production(out, l1, l2, buf, recipe=(128, 128, 128), activation="swiglu", fast_math=True)
        else:
            deep_gemm._C.fp8_mega_moe_rescale_exposure_diagnostic(out, *l1, *l2, None, buf.buffer, buf.handle.buffer_ptrs, buf.group.rank(), buf.num_max_tokens_per_rank, buf.num_experts, buf.num_topk, (128, 128, 128), "swiglu", None, True)

    # JIT, all initializations, warmup and graph capture precede event timing.
    graphs = {}
    for variant in ("production", "diagnostic"):
        for _ in range(3):
            call(variant)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(CALLS):
                call(variant)
        graphs[variant] = graph
        graph.replay()
        torch.cuda.synchronize()
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    jit = cache_identity(cache, cuda_home / "bin/cuobjdump")
    uuid = "GPU-" + str(torch.cuda.get_device_properties(rank).uuid).removeprefix("GPU-")
    clock_file = args.output / f"clock-rank-{rank}.csv"
    samples = []
    with clock_file.open("w") as clock_output:
        clock = subprocess.Popen(["nvidia-smi", f"--id={uuid}", "--query-gpu=timestamp,uuid,clocks.sm", "--format=csv,noheader,nounits", "--loop-ms=100"], stdout=clock_output, stderr=subprocess.STDOUT)
        try:
            # Alternate paired forward/reverse orders to limit temporal drift.
            for round_id in range(ROUNDS):
                for order, variants in (("forward", ("production", "diagnostic")), ("reverse", ("diagnostic", "production"))):
                    row = {"round": round_id, "order": order}
                    for variant in variants:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        graphs[variant].replay()
                        end.record()
                        end.synchronize()
                        row[variant + "_us"] = start.elapsed_time(end) * 1000 / CALLS
                    samples.append(row)
        finally:
            clock.terminate()
            clock.wait(timeout=2)
    clocks = []
    for line in clock_file.read_text().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 3 and fields[1] == uuid:
            clocks.append(float(fields[2]))
    assert clocks and all(c > 0 and math.isfinite(c) for c in clocks), "Missing SM clock samples"
    record.update(status="complete", gpu_uuid=uuid, samples=samples, sm_clock_samples_mhz=clocks, jit=jit, routing_sha256=FIXTURE_SHA, serving_environment={k: v for k, v in os.environ.items() if k.startswith(("DG_", "SGLANG_")) and k != "DG_JIT_CACHE_DIR"}, calls_per_graph=CALLS, rounds_per_order=ROUNDS, template_arguments=TEMPLATE, **summarize(samples))
    # Preserve the observed r33/r34 EP1 teardown order, including all DLPack
    # views. Graphs must release their references before the buffer/group.
    torch.cuda.synchronize()
    for graph in graphs.values():
        graph.reset()
    dist.barrier()
    for name in ("topk_idx", "topk_weights", "l1_acts", "l1_acts_sf", "l2_acts", "l2_acts_sf"):
        setattr(buf, name, None)
    buf.destroy()
    dist.destroy_process_group(group)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--routing", type=Path)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--worker-exit-codes", type=int, nargs=3)
    args = parser.parse_args()
    if args.aggregate:
        result = aggregate(args.aggregate, args.run_id, args.image_sha256, args.worker_exit_codes)
        write(args.output, result)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["LOCAL_RANK"])
    record = {"schema_version": SCHEMA, "status": "incomplete", "worker": args.worker, "rank": rank, "run_id": args.run_id, "image_sha256": args.image_sha256, "started_unix": time.time()}
    path = args.output / f"rank-{rank}.json"
    write(path, record)
    try:
        run_worker(args, record)
    except Exception as error:
        record.update(status="incomplete", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(path, record)


if __name__ == "__main__":
    main()
