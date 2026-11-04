"""Bounded dense N64 gate and isolated promotion-path timing upper bound."""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

SCHEMA = "prefill-dense-r44/v3"
FLAG = "DG_SM90_PREFILL_DENSE_N64"
DIAGNOSTIC = "DG_R44_DIAGNOSTIC_NO_PROMOTION"
SYMBOL = "sm90_fp8_gemm_promotion_diagnostic_impl"
PREFIX = f"#define {DIAGNOSTIC} 1\n#define sm90_fp8_gemm_1d2d_impl {SYMBOL}\n"
SHAPES = {"qkv": (7988, 9216, 4096), "o": (7988, 4096, 8192)}
PAIR_M, SINGLE_M = 7988, 3994
NUMERIC_MS = (7988, 8003, 3994)
ROUNDS, CALLS, ENDURANCE = 4, 4, 500
WORKER_SECONDS, AGGREGATE_SECONDS = 110, 120
THRESHOLD_US = 10


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def budget_allows_start(seconds):
    return seconds + AGGREGATE_SECONDS + 630 <= 1020


def assert_serving_environment(environment):
    assert DIAGNOSTIC not in environment, "Diagnostic flag must never enter the environment"
    assert environment.get("DG_SM90_PREFILL_DENSE_FIRST_K") == "1", "Production first-K required"
    assert environment.get(FLAG, "0") == "0", "Gate starts from production"
    assert not environment.get("DG_JIT_NVCC_COMPILER")
    assert environment.get("DG_JIT_USE_NVRTC", "0") == "0"


def assert_jit_sources(production, candidate, diagnostic, shape):
    """Actual host-generated source identity, including the recursive include hash."""
    first, prod = production.split("\n", 1)
    assert first.startswith("// Includes' hash value: ")
    assert production.split("\n", 1)[0] == candidate.split("\n", 1)[0] == diagnostic.split("\n", 1)[0]
    assert FLAG not in prod and DIAGNOSTIC not in prod and SYMBOL not in prod
    assert "#define DG_SM90_PREFILL_DENSE_FIRST_K 1\n" in prod
    assert candidate.split("\n", 1)[1] == f"#define {FLAG} 1\n" + prod
    assert diagnostic.split("\n", 1)[1] == PREFIX + prod
    _, n, k = SHAPES[shape]
    match = re.search(r"&sm90_fp8_gemm_1d2d_impl<(.+?)>\);", prod, re.DOTALL)
    expected = f"cute::UMMA::Major::K,0,{n},{k},1,256,128,128,128,128,128,3,128,256,2,true,132,GemmType::Normal,cutlass::bfloat16_t,epilogue::transform::EpilogueIdentity"
    assert match and re.sub(r"\s+", "", match[1]) == expected, "Unexpected production template"


def cache_identity(cache, cuobjdump):
    identities, sources = {s: {} for s in SHAPES}, {s: {} for s in SHAPES}
    for path in cache.glob("cache/kernel.sm90_fp8_gemm_1d2d.*/kernel.cu"):
        source = path.read_text()
        variant = "diagnostic" if PREFIX in source else "candidate" if f"#define {FLAG} 1\n" in source else "production"
        for shape, (_, n, k) in SHAPES.items():
            compact = re.sub(r"\s+", "", source)
            if f"cute::UMMA::Major::K,0,{n},{k}," not in compact:
                continue
            assert variant not in identities[shape], "Duplicate JIT instantiation"
            symbols = subprocess.check_output([str(cuobjdump), "-symbols", str(path.with_suffix(".cubin"))], text=True)
            names = [line.split()[-1] for line in symbols.splitlines() if line.startswith("STT_FUNC") and "STO_ENTRY" in line and "__instantiate_kernel" not in line]
            assert len(names) == 1 and (SYMBOL in names[0]) == (variant == "diagnostic")
            identities[shape][variant] = {"jit_cache_key": path.parent.name, "kernel_name": names[0], "generated_source_sha256": hashlib.sha256(source.encode()).hexdigest()}
            sources[shape][variant] = source
    for shape in SHAPES:
        assert set(identities[shape]) == {"production", "candidate", "diagnostic"}
        assert_jit_sources(*(sources[shape][v] for v in ("production", "candidate", "diagnostic")), shape)
    return identities


def summarize(samples, alternative="candidate"):
    assert len(samples) == 2 * ROUNDS
    deltas, orders = [], {}
    for order in ("forward", "reverse"):
        rows = [row for row in samples if row["order"] == order]
        assert sorted(row["round"] for row in rows) == list(range(ROUNDS))
        for row in rows:
            for shape in SHAPES:
                assert set(row[shape]) == {"production_us", alternative + "_us"}
                assert all(math.isfinite(v) and v > 0 for v in row[shape].values())
        savings = [sum(row[s]["production_us"] - row[s][alternative + "_us"] for s in SHAPES) for row in rows]
        deltas.extend(savings)
        orders[order] = statistics.median(savings)
    return {"median_pair_saving_us": statistics.median(deltas), "order_median_pair_saving_us": orders}


def aggregate(directory, run_id, image, codes, started, now=None):
    now = time.time() if now is None else now
    base = {"schema_version": SCHEMA, "run_id": run_id, "image_sha256": image, "status": "incomplete", "adopted": False, "worker_exit_codes": codes, "threshold_us_per_layer": THRESHOLD_US, "timing_m": PAIR_M, "additional_timing_ms": [], "numeric_ms": list(NUMERIC_MS), "interpretation": "production minus no-promotion diagnostic is an upper bound on the whole promotion path exposure, not achievable saving", "gpus": []}
    # Retain every published rank, including failed numerics and partial records.
    for worker in range(3):
        for rank in range(4):
            try:
                row = json.loads((directory / f"worker-{worker}/rank-{rank}.json").read_text())
            except (ValueError, OSError) as error:
                row = {"worker": worker, "rank": rank, "status": "incomplete", "reason": str(error)}
            base["gpus"].append(row)
    try:
        assert now <= started + AGGREGATE_SECONDS, "Aggregate deadline expired"
        assert codes == [0, 0, 0], "Workers must complete their normal exit(0) path"
        for index, row in enumerate(base["gpus"]):
            assert row["schema_version"] == SCHEMA and row["status"] == "complete"
            assert row["run_id"] == run_id and row["image_sha256"] == image
            assert (row["worker"], row["rank"]) == divmod(index, 4)
            assert started <= row["started_unix"] <= row["published_unix"] <= started + WORKER_SECONDS, "Worker deadline expired"
            assert row["exit_mode"] == "os._exit(0)"
            assert row["teardown"] == "cuda_synchronized_graphs_reset_process_group_destroyed"
            assert row["sequence_lengths"] == [3994, 3994]
            assert row["timing_m"] == PAIR_M and row["numeric_ms"] == list(NUMERIC_MS)
            assert_serving_environment(row["serving_environment"])
            assert row["weight_sha256"] and row["runtime_header_sha256"]
            for shape in SHAPES:
                assert set(row["jit"][shape]) == {"production", "candidate", "diagnostic"}
                for variant, identity in row["jit"][shape].items():
                    assert identity["jit_cache_key"] and identity["generated_source_sha256"]
                    assert (SYMBOL in identity["kernel_name"]) == (variant == "diagnostic")
            clocks = row["sm_clock_samples_mhz"]
            assert clocks and all(math.isfinite(c) and c > 0 for c in clocks)
            row.update(summarize(row["samples"]))
            row["median_diagnostic_bound_us"] = summarize(row["diagnostic_samples"], "diagnostic")["median_pair_saving_us"]
        assert len({r["gpu_uuid"] for r in base["gpus"]}) == 12, "Expected twelve distinct GPUs"
        assert len({r["weight_sha256"] for r in base["gpus"]}) == 1, "Checkpoint mismatch"
        assert len({r["runtime_header_sha256"] for r in base["gpus"]}) == 1, "Overlay mismatch"
        for shape in SHAPES:
            for variant in ("production", "candidate", "diagnostic"):
                assert len({r["jit"][shape][variant]["generated_source_sha256"] for r in base["gpus"]}) == 1, "Generated source mismatch"
        # Preserve the decisive bound even if the candidate fails its checks.
        base["median_12gpu_diagnostic_bound_us"] = statistics.median(r["median_diagnostic_bound_us"] for r in base["gpus"])
        for row in base["gpus"]:
            assert set(row["numerics"]) == {str(m) for m in NUMERIC_MS}
            for rows in NUMERIC_MS:
                for shape in SHAPES:
                    numeric = row["numerics"][str(rows)][shape]
                    assert numeric["passed"], "Candidate numerical check failed"
                    assert numeric["comparison_mode"] == "bitwise"
                    assert numeric["comparison_mode"] != "bitwise" or numeric["bitwise_equal_production"], "Candidate is not bitwise equal"
                    assert numeric["production_reference_passed"] and numeric["production_comparison_passed"]
                    for key in ("production_max_abs", "production_max_rel", "reference_max_abs", "reference_max_rel"):
                        assert math.isfinite(numeric[key]) and numeric[key] >= 0
                    assert numeric["reference_passed"] and numeric["reference_kind"] == "independent_fp32_k128_no_tf32"
            for shape in SHAPES:
                endurance = row["endurance"][shape]
                assert endurance["calls"] >= ENDURANCE and endurance["passed"] and endurance["compared_every_call"]
                assert endurance["reference"] == "own_first_output" and endurance["rows"] == PAIR_M
        median = statistics.median(r["median_pair_saving_us"] for r in base["gpus"])
        base.update(status="complete", adopted=median >= THRESHOLD_US, median_12gpu_pair_saving_us=median)
    except (AssertionError, KeyError, ValueError, OSError, TypeError) as error:
        base["reason"] = str(error)
        base["traceback"] = traceback.format_exc()
    return base


def dense_snapshot(tensors):
    """Production SM90 parameter allocations and BF16-to-FP32 scale loader copy.

    fc275674 fp8.py allocates FP32 scale parameters; parameter.py copies the
    checkpoint into them. Q/K/V are concatenated on dim0 in that order.
    """
    import torch

    def loaded(projection, suffix):
        raw = tensors[f"self_attn.{projection}.{suffix}"]
        dtype = torch.float32 if suffix == "weight_scale_inv" else torch.float8_e4m3fn
        return torch.empty(raw.shape, dtype=dtype, device=raw.device).copy_(raw)

    qkv = tuple(torch.cat([loaded(p, s) for p in ("q_proj", "k_proj", "v_proj")], dim=0) for s in ("weight", "weight_scale_inv"))
    o = tuple(loaded("o_proj", s) for s in ("weight", "weight_scale_inv"))
    return [qkv, o]


def snapshot_weights(model):
    import torch
    from safetensors import safe_open

    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.layers.0."
    names = [prefix + "self_attn." + p + "." + s for p in ("q_proj", "k_proj", "v_proj", "o_proj") for s in ("weight", "weight_scale_inv")]
    tensors, provenance = {}, {}
    digest = hashlib.sha256()
    for file in sorted({index[name] for name in names}):
        with safe_open(model / file, framework="pt", device="cpu") as handle:
            for name in sorted(n for n in names if index[n] == file):
                tensor = handle.get_tensor(name)
                raw = tensor.view(torch.uint8).numpy().tobytes()
                provenance[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": hashlib.sha256(raw).hexdigest()}
                digest.update(name.encode()); digest.update(raw)
                tensors[name.removeprefix(prefix)] = tensor.cuda()
    return dense_snapshot(tensors), digest.hexdigest(), provenance


def numerical_check(lhs, weight, production, candidate):
    """Full output reference using ordinary FP32 GEMMs, never DeepGEMM.

    FP8 values and K128 scale granularity are held fixed. Dot-product order
    can differ inside K128. gamma(2*(128 + 2 + blocks)) times the absolute
    dot sum bounds both FP32 evaluation orders; BF16 adds half an ulp. We require
    bit equality with production as well as this independent reference.
    """
    import torch

    a, sa = lhs
    b, sb = weight
    # The per-element promotion order is preserved: any byte mismatch rejects
    # this candidate, even when it remains inside the reference error bound.
    bitwise = bool(torch.equal(candidate.view(torch.uint8), production.view(torch.uint8)))
    m, k = a.shape
    n = b.shape[0]
    ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    absolute = torch.zeros_like(ref)
    for kb in range(k // 128):
        aa = a[:, kb * 128:(kb + 1) * 128].float()
        bb = b[:, kb * 128:(kb + 1) * 128].float()
        scale = sa[:m, kb, None] * sb[:, kb].repeat_interleave(128)[None, :]
        product = aa @ bb.T
        ref.addcmul_(product, scale)
        absolute.addcmul_(aa.abs() @ bb.abs().T, scale.abs())
    eps = 2**-24
    operations = 2 * (128 + 2 + k // 128)
    allowance = (operations * eps / (1 - operations * eps)) * absolute + ref.abs() * 2**-8 + 2**-126
    cf, pf = candidate.float(), production.float()
    prod_error, ref_error = (cf - pf).abs(), (cf - ref).abs()
    production_reference_passed = bool(torch.all(torch.isfinite(pf) & torch.isfinite(ref) & ((pf - ref).abs() <= allowance)).item())
    production_comparison_passed = bitwise or bool(torch.all(torch.isfinite(cf) & torch.isfinite(pf) & (prod_error <= 2 * allowance)).item())
    reference_passed = bool(torch.all(torch.isfinite(cf) & torch.isfinite(ref) & (ref_error <= allowance)).item())
    def metric(value):
        number = value.max().item()
        return number if math.isfinite(number) else None

    return {"passed": bitwise and production_comparison_passed and production_reference_passed and reference_passed, "comparison_mode": "bitwise" if bitwise else "fp32_reordering", "production_comparison_passed": production_comparison_passed, "production_reference_passed": production_reference_passed, "bitwise_equal_production": bitwise, "reference_passed": reference_passed, "reference_kind": "independent_fp32_k128_no_tf32", "production_max_abs": metric(prod_error), "production_max_rel": metric(prod_error / pf.abs().clamp_min(2**-24)), "reference_max_abs": metric(ref_error), "reference_max_rel": metric(ref_error / ref.abs().clamp_min(2**-24)), "reference_tolerance": "gamma(2*(128+2+K/128))*sum_abs_products + abs(reference)/256 + 2^-126; production bit equality is mandatory; twice this error bound is reported for FP32/BF16 comparison only"}


def run_worker(args, record):
    assert_serving_environment(os.environ)  # before importing any GPU package
    rank = int(os.environ["LOCAL_RANK"])
    cache = args.cache
    # One fresh node-local cache per container; four ranks use existing atomic publish.
    os.environ["DG_JIT_CACHE_DIR"] = str(cache)
    import deep_gemm
    import torch
    import torch.distributed as dist
    from sglang.srt.layers.prefill_r37 import Dense

    torch.cuda.set_device(rank)
    assert torch.cuda.device_count() == 4 and torch.cuda.get_device_capability() == (9, 0)
    properties = torch.cuda.get_device_properties(rank)
    assert properties.multi_processor_count == 132
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    header = Path(deep_gemm.__file__).parent / "include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh"
    text = header.read_text()
    assert DIAGNOSTIC in text and "FP8MMASelector<64>" in text
    record["runtime_header_sha256"] = hashlib.sha256(header.read_bytes()).hexdigest()
    dist.init_process_group("gloo", timeout=timedelta(seconds=65))
    assert dist.get_world_size() == 4
    weights, record["weight_sha256"], record["checkpoint_tensors"] = snapshot_weights(args.model)
    case = Dense(PAIR_M, weights)
    # Scheduler page rounding affects accounting, not the dense input rows.
    record["sequence_lengths"] = [3994, 3994]
    record["timing_m"] = PAIR_M
    record["numeric_ms"] = list(NUMERIC_MS)
    record["activation"] = "prefill_r37.Dense deterministic BF16 N(0,0.25), production K128 quantizer"
    diagnostic_out = [torch.empty_like(x) for x in case.out["control"]]

    def call(variant, projection):
        if variant == "diagnostic":
            deep_gemm._C.fp8_gemm_nt_promotion_diagnostic(*case.lhs[projection], *weights[projection], diagnostic_out[projection])
        else:
            os.environ[FLAG] = "1" if variant == "candidate" else "0"
            case.step("candidate" if variant == "candidate" else "control", projection)

    def compile_variants(variants):
        # Different shapes compile concurrently, exactly once per worker.
        # Other ranks finish checkpoint loading while owners compile. The
        # barrier precedes every non-owner call into DeepGEMM's disk cache.
        if rank < len(SHAPES):
            for variant in variants:
                call(variant, rank)
            torch.cuda.synchronize()
        dist.barrier()

    def capture(variants):
        graphs = {s: {} for s in SHAPES}
        for i, shape in enumerate(SHAPES):
            for variant in variants:
                for _ in range(3):
                    call(variant, i)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(CALLS):
                        call(variant, i)
                graphs[shape][variant] = graph
                graph.replay()
                torch.cuda.synchronize()
        return graphs

    phases = {}
    def checkpoint(name):
        phases[name] = time.time() - args.torchrun_started
        print(f"dense_r44 rank={rank} phase={name} torchrun_seconds={phases[name]:.3f}", flush=True)

    compile_variants(("production", "candidate"))
    checkpoint("candidate_jit_ready")
    # Correctness before endurance and timing; no diagnostic JIT on this path.
    pair_case = case
    record["numerics"] = {}
    for rows in NUMERIC_MS:
        case = pair_case if rows == PAIR_M else Dense(rows, weights)
        record["numerics"][str(rows)] = {}
        for i, shape in enumerate(SHAPES):
            call("production", i)
            call("candidate", i)
            torch.cuda.synchronize()
            record["numerics"][str(rows)][shape] = numerical_check(case.lhs[i], weights[i], case.out["control"][i], case.out["candidate"][i])
    case = pair_case
    checkpoint("correctness_done")
    record["endurance"] = {}
    for i, shape in enumerate(SHAPES):
        # One call per replay; compare every consecutive output with its first.
        first = case.out["candidate"][i].clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call("candidate", i)
        ok = torch.ones((), device="cuda", dtype=torch.bool)
        for _ in range(ENDURANCE):
            graph.replay()
            ok.logical_and_(torch.all(case.out["candidate"][i].view(torch.uint8) == first.view(torch.uint8)))
        torch.cuda.synchronize()
        record["endurance"][shape] = {"calls": ENDURANCE, "passed": bool(ok.item()), "compared_every_call": True, "reference": "own_first_output", "rows": PAIR_M}
        graph.reset()
    checkpoint("endurance_done")
    graphs = capture(("production", "candidate"))
    uuid = "GPU-" + str(properties.uuid).removeprefix("GPU-")
    clock_file = args.output / f"clock-rank-{rank}.csv"
    samples, diagnostic_samples = [], []
    with clock_file.open("w") as output:
        clock = subprocess.Popen(["nvidia-smi", f"--id={uuid}", "--query-gpu=timestamp,uuid,clocks.sm", "--format=csv,noheader,nounits", "--loop-ms=100"], stdout=output, stderr=subprocess.STDOUT)
        try:
            def timing(timing_graphs, alternative, target):
                for round_id in range(ROUNDS):
                    for order, variants in (("forward", ("production", alternative)), ("reverse", (alternative, "production"))):
                        row = {"round": round_id, "order": order}
                        for shape in SHAPES:
                            row[shape] = {}
                            for variant in variants:
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                start.record(); timing_graphs[shape][variant].replay(); end.record()
                                end.synchronize()
                                row[shape][variant + "_us"] = start.elapsed_time(end) * 1000 / CALLS
                        target.append(row)

            timing(graphs, "candidate", samples)
            checkpoint("candidate_timing_done")
            # Lower priority bound uses its own paired production measurement.
            compile_variants(("diagnostic",))
            diagnostic_graphs = capture(("diagnostic",))
            for shape in SHAPES:
                graphs[shape].update(diagnostic_graphs[shape])
            timing(graphs, "diagnostic", diagnostic_samples)
            checkpoint("diagnostic_done")
        finally:
            clock.terminate(); clock.wait(timeout=2)
    clocks = []
    for line in clock_file.read_text().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and fields[1] == uuid:
            clocks.append(float(fields[2]))
    assert clocks and all(math.isfinite(c) and c > 0 for c in clocks)
    os.environ[FLAG] = "0"
    record["jit"] = cache_identity(cache, Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/cuobjdump")
    checkpoint("teardown_started")
    teardown(graphs, cache, args.output, rank)
    checkpoint("teardown_done")
    record.update(status="complete", gpu_uuid=uuid, samples=samples, diagnostic_samples=diagnostic_samples, sm_clock_samples_mhz=clocks, calls_per_graph=CALLS, rounds_per_order=ROUNDS, phase_seconds_from_torchrun=phases, exit_mode="os._exit(0)", teardown="cuda_synchronized_graphs_reset_process_group_destroyed", serving_environment={k: v for k, v in os.environ.items() if k.startswith(("DG_", "SGLANG_")) and k != "DG_JIT_CACHE_DIR"})
    record.update(summarize(samples))


def teardown(graphs, cache, output, rank):
    import torch
    import torch.distributed as dist

    # CUDA work and all captured references precede group destruction. Dense
    # uses no symmetric memory or EP1 group; only this CPU coordination group.
    torch.cuda.synchronize()
    for shape_graphs in graphs.values():
        for graph in shape_graphs.values():
            graph.reset()
    dist.barrier()
    if rank == 0:
        shutil.copytree(cache / "cache", output / "jit-cache/cache")
    dist.barrier()
    dist.destroy_process_group()


def publish_and_exit(args, record):
    # Designed success path: close the atomic JSON writer and flush Python
    # output before skipping interpreter/CUDA static destructors. No fallback.
    record["published_unix"] = time.time()
    assert record["published_unix"] <= args.started + WORKER_SECONDS, "Worker publication deadline expired"
    write(args.output / f"rank-{record['rank']}.json", record)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--started", type=float, required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--torchrun-started", type=float)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--worker-exit-codes", type=int, nargs=3)
    args = parser.parse_args()
    if args.aggregate:
        result = aggregate(args.aggregate, args.run_id, args.image_sha256, args.worker_exit_codes, args.started)
        write(args.output, result)
        # Selection is deliberately performed by sbatch after the 120s group
        # exits successfully, so a killed supervisor cannot enable a candidate.
        return
    rank = int(os.environ["LOCAL_RANK"])
    record = {"schema_version": SCHEMA, "status": "incomplete", "worker": args.worker, "rank": rank, "run_id": args.run_id, "image_sha256": args.image_sha256, "started_unix": time.time()}
    write(args.output / f"rank-{rank}.json", record)
    try:
        run_worker(args, record)
        publish_and_exit(args, record)
    except Exception as error:
        record.update(status="incomplete", reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write(args.output / f"rank-{rank}.json", record)
        raise


if __name__ == "__main__":
    main()
