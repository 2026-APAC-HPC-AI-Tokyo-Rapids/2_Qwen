"""Screen the BM256 Prefill tail-wave skip before PP4 model load.

One torchrun per worker, four local ranks; rank zero warms the four JITs
(QKV/O x stock/skip) in a shared node-local cache before the other ranks.
The host reads the switch on every call; compiled_dims stays serving's "nk".
Only stdlib imports run during CPU aggregate tests.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

KEY = "DG_SM90_1D2D_PREFILL_TAIL_SKIP"
SCHEMA = "firmus-prefill-dense-selection/v1"
ROUNDS = 15
REPS = 20
ENDURANCE = 200
PROJECTIONS = (("qkv", 9216, 4096), ("o", 4096, 8192))


def aggregate(directory, env_output, run_id, deadline, workers_ok=True):
    """Never mutate the pinned environment unless every fresh result passes."""
    try:
        assert workers_ok, "selector process failed or timed out"
        assert time.time() <= deadline, "selection deadline expired"
        for worker in range(3):
            for rank in range(4):
                record = json.loads((directory / f"prefill-{worker}" / f"rank-{rank}.json").read_text())
                assert record["schema"] == SCHEMA and record["run_id"] == run_id, "stale result"
                assert record["worker"] == f"prefill-{worker}" and record["rank"] == rank, "wrong GPU result"
                assert record["correctness"] is True, "correctness failed"
                assert record["endurance"] is True and record["endurance_calls"] == ENDURANCE, "endurance failed"
                assert record["distinct_kernels"] is True, "host did not select distinct kernels"
                assert record["finished_at"] <= deadline, "late result"
                assert record["rounds"] >= ROUNDS and record["calls_per_sample"] >= REPS, "incomplete timing"
                medians = {}
                for mode in ("stock", "skip"):
                    samples = record["pair_us"][mode]
                    assert len(samples) == record["rounds"], "missing timing samples"
                    assert all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in samples), "invalid timing"
                    medians[mode] = statistics.median(samples)
                assert medians["stock"] - medians["skip"] >= 2.0, f"prefill-{worker} GPU {rank}: saving below 2 us"
        original = env_output.read_text()
        assert not any(line.split("=", 1)[0] == KEY for line in original.splitlines()), "tail switch already present"
        assert time.time() <= deadline, "selection deadline expired"
        # Preserve every pinned line. Exactly one new assignment, only for Prefill.
        temporary = env_output.with_name(env_output.name + ".tail-skip.tmp")
        try:
            temporary.write_text(original + ("\n" if original and not original.endswith("\n") else "") + KEY + "=1\n")
            temporary.replace(env_output)
        finally:
            temporary.unlink(missing_ok=True)
        return {"adopted": True, "reason": "all twelve GPUs passed and saved at least 2 us"}
    except Exception as error:  # noqa: BLE001 - every gate failure must keep the pin
        return {"adopted": False, "reason": f"{type(error).__name__}: {error}"}


@contextmanager
def mode(enabled):
    previous = os.environ.get(KEY)
    os.environ[KEY] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(KEY, None)
        else:
            os.environ[KEY] = previous


def require_host():
    """An old image with just the new header cannot implement this host switch."""
    record = json.loads(Path("/opt/apac/deep-gemm-host-build.json").read_text())
    patch = Path(__file__).resolve().parents[1] / "source/deep-gemm-host/sm90-1d2d-prefill-tail-skip.patch"
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    assert record["source_commit"] == "1c8138a59b3df5dac50b76b730834ed57982fdfb", "wrong DeepGEMM source"
    assert record["patches"][patch.name] == digest, "tail skip requires the rebuilt host image"
    return digest


def inputs(torch, rows, n, k, seed):
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16, generator=generator) * 4
    x[0].zero_()
    a, sa = sglang_per_token_group_quant_fp8(x, 128, column_major_scales=True, scale_tma_aligned=True)
    b = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=generator) * 4).to(torch.float8_e4m3fn)
    sb = 10.0 ** (torch.rand(n // 128, k // 128, device="cuda", generator=generator) * 3 - 3)
    out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    assert sa.dtype == sb.dtype == torch.float32 and sa.stride(0) == 1
    assert sa.stride(1) % 4 == 0 and sb.is_contiguous()
    return (a, sa), (b, sb), out


def reference_check(torch, values):
    """Decode selector's FP32 reference and the same two error tolerances."""
    (a, sa), (b, sb), out = values
    x = a.float() * sa.repeat_interleave(128, 1)
    weight = b.float() * sb.repeat_interleave(128, 0).repeat_interleave(128, 1)
    reference = x @ weight.T
    error = out.float() - reference
    metrics = {"finite": bool(out.isfinite().all() and reference.isfinite().all()), "relative_l2": (error.norm() / reference.norm().clamp_min(1e-12)).item(), "max_scaled_error": (error.abs().amax() / reference.abs().amax().clamp_min(1e-12)).item()}
    assert metrics["finite"] and 0 <= metrics["relative_l2"] <= 0.005 and 0 <= metrics["max_scaled_error"] <= 0.01, metrics
    return metrics


def call(values):
    import deep_gemm

    lhs, rhs, out = values
    deep_gemm.fp8_gemm_nt(lhs, rhs, out, compiled_dims="nk")


def confirm_host_call(values, log):
    """Capture native cout, not Python stdout; an ignored switch must fail closed."""
    saved = os.dup(1)
    try:
        with log.open("w") as stream:
            os.dup2(stream.fileno(), 1)
            os.environ["DG_PRINT_CONFIGS"] = "1"
            with mode(True):
                call(values)
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.environ.pop("DG_PRINT_CONFIGS", None)
    assert KEY + "=1:" in log.read_text(), "missing enabled host print line"


def prepare(torch, rank, output):
    prepared, numeric = {}, {}
    for op, n, k in PROJECTIONS:
        for rows in (7988, 3994):
            values = inputs(torch, rows, n, k, 91000 + rank * 100 + n + rows)
            with mode(False):
                call(values)
            baseline = values[2].clone()
            numeric[f"{op}-{rows}"] = reference_check(torch, values)
            values[2].fill_(float("nan"))
            if rows == 7988:
                confirm_host_call(values, output / f"rank-{rank}-{op}-host.log")
            else:
                with mode(True):
                    call(values)
            assert torch.equal(values[2].view(torch.int16), baseline.view(torch.int16)), f"{op} M{rows}: stock/skip bytes differ"
            if rows == 7988:
                prepared[op] = (values, baseline)
    return prepared, numeric


def measure(torch, prepared):
    samples = {"stock": [], "skip": []}
    for turn in range(ROUNDS):
        pair = {"stock": 0.0, "skip": 0.0}
        for op, _, _ in PROJECTIONS:
            for name in (("stock", "skip") if turn % 2 == 0 else ("skip", "stock")):
                with mode(name == "skip"):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(REPS):
                        call(prepared[op][0])
                    end.record()
                    end.synchronize()
                pair[name] += start.elapsed_time(end) * 1000 / REPS
        for name, values in samples.items():
            values.append(pair[name])
    return samples


def run(output, run_id, deadline):
    begin = time.monotonic()
    import torch
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 4
    worker = output.name
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema": SCHEMA, "worker": worker, "rank": rank, "run_id": run_id, "correctness": False, "endurance": False, "distinct_kernels": False}
    try:
        torch.cuda.set_device(rank)
        assert torch.cuda.get_device_capability() == (9, 0)
        assert torch.cuda.get_device_properties(rank).multi_processor_count == 132
        result["host_patch_sha256"] = require_host()
        # A shared node-local cache, as in Decode. Only rank zero cold-compiles.
        dist.init_process_group("gloo")
        torch.backends.cuda.matmul.allow_tf32 = False
        for turn in (0, 1):
            if (rank == 0) == (turn == 0):
                prepared, numeric = prepare(torch, rank, output)
            dist.barrier()
        result.update(correctness=True, distinct_kernels=True, numeric=numeric)
        # Both variants are already warm. Only these four GPUs run here before model load.
        samples = measure(torch, prepared)
        with mode(True):
            for index in range(ENDURANCE):
                call(prepared[PROJECTIONS[index % 2][0]][0])
        torch.cuda.synchronize()
        for values, baseline in prepared.values():
            assert torch.equal(values[2].view(torch.int16), baseline.view(torch.int16)), "endurance bytes differ"
        assert time.time() <= deadline, "selection deadline expired"
        medians = {name: statistics.median(values) for name, values in samples.items()}
        result.update(endurance=True, endurance_calls=ENDURANCE, rounds=ROUNDS, calls_per_sample=REPS, pair_us=samples, median_pair_us=medians, saving_us=medians["stock"] - medians["skip"])
        dist.destroy_process_group()
    except Exception as error:  # noqa: BLE001 - publish a failed GPU result; aggregate keeps stock
        result["error"] = f"{type(error).__name__}: {error}"
    result.update(elapsed_seconds=time.monotonic() - begin, finished_at=time.time())
    # A killed process cannot publish a partial JSON file.
    temporary = output / f"rank-{rank}.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / f"rank-{rank}.json")
    print(json.dumps(result, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-output", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deadline", required=True, type=float)
    parser.add_argument("--workers-ok", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.aggregate:
        result = aggregate(args.aggregate, args.env_output, args.run_id, args.deadline, bool(args.workers_ok))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    else:
        run(args.output, args.run_id, args.deadline)


if __name__ == "__main__":
    main()
