"""Screen the Decode router/dispatch bundle before serving, on the existing GPUs."""
import argparse
import json
import math
import os
import statistics
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from runpy import run_path

ROWS = (144, 152, 160, 168, 176)
LAYERS = 94
KEYS = ("DG_SM90_DECODE_PREPARE", "SGLANG_DECODE_ROUTER")


def choose_bundle(records):
    """Every shape and both blocks must save 5us on the slowest rank."""
    expected = {(rows, block) for rows in ROWS for block in range(2)}
    if {(r["rows"], r["block"]) for r in records} != expected or len(records) != len(expected):
        return False
    for record in records:
        if not record["numeric_passed"]:
            return False
        for name in ("control", "candidate"):
            values = record[name]
            if len(values) != 15 or not all(math.isfinite(v) and v > 0 for v in values):
                return False
        if statistics.median(record["control"]) - statistics.median(record["candidate"]) < 5:
            return False
    return True


@contextmanager
def bundle_mode(enabled):
    previous = {key: os.environ.get(key) for key in KEYS}
    os.environ.update(dict(zip(KEYS, ("1", "cluster") if enabled else ("0", ""))))
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-output", type=Path, required=True)
    args = parser.parse_args()
    import deep_gemm
    import torch
    import torch.distributed as dist
    from sglang.kernels.ops.moe.decode_router import _decode_router

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    host = json.loads(Path("/opt/apac/deep-gemm-host-build.json").read_text())
    assert "sm90-decode-prepare.patch" in host["patches"], "Decode prepare requires the new host image"
    checks = run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
    # select-firmus-decode-kernels.sh runs the MegaMoE serving gate with the bundle
    # enabled as its own torchrun first: re-initializing NCCL on the same rendezvous
    # after that gate's destroy_process_group read a stale bootstrap id (40660502072).
    with bundle_mode(True):
        checks["check_decode_router_cluster"]()

    dist.init_process_group("nccl", timeout=timedelta(seconds=240), device_id=torch.device("cuda", rank))
    assert dist.get_world_size() == 4
    bench = run_path(str(Path(__file__).with_name("bench-firmus-megamoe.py")))
    torch.manual_seed(713 + rank)
    device = torch.device("cuda", rank)
    w13 = (torch.randn(32, 3072, 4096, device=device) * 4).to(torch.float8_e4m3fn)
    w2 = (torch.randn(32, 4096, 1536, device=device) * 4).to(torch.float8_e4m3fn)
    s13 = torch.rand(32, 24, 32, device=device) * 0.005 + 0.005
    s2 = torch.rand(32, 32, 12, device=device) * 0.005 + 0.005
    l1, l2 = deep_gemm.transform_weights_for_mega_moe_sm90((w13, s13), (w2, s2))
    buf, _ = bench["bench_symm_buffer"](dist.group.WORLD)
    # 94 distinct gate weights exceed H200 L2, as in the serving step.
    weights = torch.randn(LAYERS, 128, 4096, device=device, dtype=torch.bfloat16) * 0.02
    records = []
    for rows in ROWS:
        hidden = torch.randn(rows, 4096, device=device, dtype=torch.bfloat16) * 0.25
        generator = torch.Generator(device=device).manual_seed(713 + rank)
        ids = bench["routing"]("balanced", rows, generator, device)
        top_weights = torch.rand(rows, 8, device=device)
        top_weights /= top_weights.sum(-1, keepdim=True)
        logits = torch.empty(rows, 128, device=device, dtype=torch.bfloat16)
        output = torch.empty_like(hidden)
        graphs = {}
        for enabled in (False, True):
            with bundle_mode(enabled):
                def step(enabled=enabled, hidden=hidden, logits=logits, ids=ids, top_weights=top_weights, rows=rows, output=output):
                    for layer in range(LAYERS):
                        if enabled:
                            _decode_router(hidden, weights[layer], logits)
                        else:
                            torch.mm(hidden, weights[layer].T, out=logits)
                        deep_gemm.mega_moe_pre_dispatch_sm90(hidden, ids, top_weights, buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights, num_tokens=rows, group_size=128, routed_scaling_factor=1.0)
                        deep_gemm.fp8_mega_moe(output, l1, l2, buf, recipe=(128, 128, 128), activation="swiglu", fast_math=True)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    step()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with torch.cuda.graph(graph):
                    step()
                graphs[enabled] = graph
        for block in range(2):
            values = {False: [], True: []}
            order = (False, True) if block == 0 else (True, False)
            for _ in range(15):
                for enabled in order:
                    dist.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[enabled].replay()
                    end.record()
                    end.synchronize()
                    duration = torch.tensor(start.elapsed_time(end) * 1000 / LAYERS, device=device)
                    dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                    values[enabled].append(duration.item())
            records.append({"rows": rows, "block": block, "numeric_passed": True, "control": values[False], "candidate": values[True]})
    selected = choose_bundle(records)
    if rank == 0:
        args.output.write_text(json.dumps({"selected": selected, "records": records, "minimum_saving_us_per_layer": 5, "layers": LAYERS}, indent=2) + "\n")
        args.env_output.write_text("DG_SM90_DECODE_PREPARE=1\nSGLANG_DECODE_ROUTER=cluster\n" if selected else "")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
