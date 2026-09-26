"""Short hardware diagnostics inside a two-node, 16-GPU scheduler allocation.

These are operator diagnostics, not Qwen forward passes or NIXL measurements.
"""

import json
import os
import socket
import statistics
import sys
from datetime import timedelta
from pathlib import Path


def main():
    import torch
    import torch.distributed as dist

    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS allocation required")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 16 or int(os.environ["LOCAL_WORLD_SIZE"]) != 8:
        raise RuntimeError("requires two nodes with eight GPUs each")
    output = Path(sys.argv[1])
    torch.manual_seed(2026)
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    groups = [dist.new_group(list(range(start, start + 8))) for start in (0, 8)]
    pairs = [dist.new_group([i, i + 8]) for i in range(8)]
    rows = []

    def measure(name, operation, **metadata):
        for _ in range(3):
            operation()
        torch.cuda.synchronize()
        timings = []
        for _ in range(5):
            dist.barrier()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(10):
                operation()
            end.record()
            end.synchronize()
            timings.append(begin.elapsed_time(end) / 10)
        rows.append({"operation": name, "latency_ms_samples": timings, "median_latency_ms": statistics.median(timings), "warmup_iterations": 3, "iterations_per_sample": 10, **metadata})

    try:
        src = torch.ones(64 * 1024 * 1024, device="cuda", dtype=torch.float32)
        dst = torch.empty_like(src)
        measure("device_copy", lambda dst=dst, src=src: dst.copy_(src), payload_bytes=src.numel() * src.element_size(), accounted_bytes_per_iteration=2 * src.numel() * src.element_size())
        del src, dst
        a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        c = torch.empty_like(a)
        measure("bf16_gemm", lambda a=a, b=b, c=c: torch.mm(a, b, out=c), m=4096, n=4096, k=4096, flops_per_iteration=2 * 4096**3)
        del a, b, c
        for size in (1024 * 1024, 16 * 1024 * 1024, 256 * 1024 * 1024):
            tensor = torch.zeros(size // 4, device="cuda", dtype=torch.float32)
            measure("nccl_intra_node_all_reduce", lambda tensor=tensor: dist.all_reduce(tensor, group=groups[rank // 8]), payload_bytes=size, participants=8)
            measure("nccl_inter_node_pair_all_reduce", lambda tensor=tensor: dist.all_reduce(tensor, group=pairs[local_rank]), payload_bytes=size, participants=2)
            del tensor
        props = torch.cuda.get_device_properties(local_rank)
        record = {"schema_version": "qwen-hardware-probe/v1", "cluster": "aspire2ap", "job_id": os.environ["PBS_JOBID"], "hostname": socket.gethostname(), "rank": rank, "local_rank": local_rank, "torch": torch.__version__, "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(), "gpu": {"name": props.name, "memory_bytes": props.total_memory, "sm_count": props.multi_processor_count}, "measurements": rows, "limitations": ["Synthetic BF16 GEMM, not FP8 MoE/attention/whole-forward latency", "NCCL collectives, not NIXL KV transfer", "Five temporal samples in one allocation, not independent run replication", "GPU event timings; allocation and process startup excluded"]}
        with (output / f"rank-{rank}.json").open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
