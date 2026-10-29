"""Probe the PP4 copy-engine proxy transport on four GPUs before the Prefill workers start.

Run under torchrun on four GPUs of one node with the serving image: rank r
plays PP stage r. The probe makes the same torch calls as
pp4-proxy-copy-engine.patch: the receiver shares [slots, width, rows, hidden]
BF16 buffers with reduce_tensor, the sender creates interprocess events and
shares their IPC handles (here with one all_gather_object), the sender
rebuilds the buffers and the receiver opens the events on its own device. Per
transfer the sender copies two tensors into the receiver's slot on a side
stream (after a GPU sleep on every other transfer, so the wait must really
block) and records the slot event, then sends the transfer number over gloo;
the receiver's stream waits on the event and compares the slot with the
expected values. Slots are reused, gated by acks. Run 40849459077 segfaulted
in cuStreamWaitEvent on the first such wait (the events were then opened on
the sender's device); a crash or error here only loses this rank's result.
--aggregate enables the copy engine (SGLANG_PP4_PROXY_TRANSPORT=copy) only if
all four ranks finished every transfer without a mismatch; anything missing
keeps in-stream PyNccl.
"""
import argparse
import json
import os
from pathlib import Path

STAGES, SLOTS, WIDTH, HIDDEN = 4, 2, 4, 4096
# PP4 microbatch row counts seen in serving (run 40825417205), and the chunk cap.
ROWS = (7988, 3994, 7989, 8192, 1, 7988)
TRANSFERS = 12
SLEEP_CYCLES = 20_000_000


def expected(torch, stage, transfer, index, rows, device):
    """Exact BF16 values from integer arithmetic, identical on every GPU."""
    seed = (stage * 1000 + transfer) * 10 + index
    values = torch.arange(rows * HIDDEN, device=device, dtype=torch.int64).mul_(2654435761).add_(seed).remainder_(65521)
    return values.to(torch.float32).sub_(32760.0).mul_(1.0 / 256.0).to(torch.bfloat16).view(rows, HIDDEN)


def run_probe(output):
    import torch
    import torch.distributed as dist
    from torch.multiprocessing.reductions import reduce_tensor

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo")
    first, last = rank == 0, rank == STAGES - 1
    forward = torch.cuda.Stream(device=device)
    local = {"rank": rank, "transfers_received": 0, "transfers_sent": 0, "mismatches": None}
    buffers = events = stream = peer_buffers = peer_events = None
    shared = {"buffers": None, "events": None}
    if not first:
        buffers = torch.empty((SLOTS, WIDTH, max(ROWS), HIDDEN), dtype=torch.bfloat16, device=device)
        mismatches = torch.zeros((), dtype=torch.int64, device=device)
        shared["buffers"] = reduce_tensor(buffers)
    if not last:
        stream = torch.cuda.Stream(device=device)
        events = [torch.cuda.Event(interprocess=True) for _ in range(SLOTS)]
        shared["events"] = [event.ipc_handle() for event in events]
    # One collective instead of the serving patch's async point-to-point exchange.
    gathered = [None] * STAGES
    dist.all_gather_object(gathered, shared)
    if not last:
        rebuild, args = gathered[rank + 1]["buffers"]
        peer_buffers = rebuild(*args)
    if not first:
        peer_events = [torch.cuda.Event.from_ipc_handle(device, handle) for handle in gathered[rank - 1]["events"]]
    number = torch.zeros(1, dtype=torch.int64)
    sends = []
    for transfer in range(TRANSFERS):
        slot, rows = transfer % SLOTS, ROWS[transfer % len(ROWS)]
        if not first:
            dist.recv(number, src=rank - 1, tag=1)
            assert int(number) == transfer
            forward.wait_event(peer_events[slot])
            with torch.cuda.stream(forward):
                for index in range(2):
                    mismatches += (buffers[slot, index, :rows] != expected(torch, rank - 1, transfer, index, rows, device)).sum()
            forward.synchronize()
            ack = torch.tensor([transfer], dtype=torch.int64)
            sends.append((ack, dist.isend(ack, dst=rank - 1, tag=2)))
            local["transfers_received"] += 1
        if not last:
            if transfer >= SLOTS:
                dist.recv(number, src=rank + 1, tag=2)
                assert int(number) == transfer - SLOTS
            with torch.cuda.stream(forward):
                tensors = [expected(torch, rank, transfer, index, rows, device) for index in range(2)]
            stream.wait_stream(forward)
            with torch.cuda.stream(stream):
                if transfer % 2 == 0:
                    torch.cuda._sleep(SLEEP_CYCLES)
                for index, tensor in enumerate(tensors):
                    peer_buffers[slot, index, :rows].copy_(tensor, non_blocking=True)
                    tensor.record_stream(stream)
                events[slot].record(stream)
            sent = torch.tensor([transfer], dtype=torch.int64)
            sends.append((sent, dist.isend(sent, dst=rank + 1, tag=1)))
            local["transfers_sent"] += 1
    for _, work in sends:
        work.wait()
    if not last:
        stream.synchronize()
        for _ in range(min(SLOTS, TRANSFERS)):
            dist.recv(number, src=rank + 1, tag=2)
    if not first:
        local["mismatches"] = int(mismatches)
    dist.barrier()
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(local, indent=2) + "\n")
    print(json.dumps(local), flush=True)


def aggregate(directory, output, env_output):
    entries = []
    for rank in range(STAGES):
        path = directory / f"rank-{rank}.json"
        entries.append(json.loads(path.read_text()) if path.is_file() else None)
    reason = None
    for rank, entry in enumerate(entries):
        if entry is None:
            reason = f"missing rank {rank}"
        elif entry["transfers_sent"] != (0 if rank == STAGES - 1 else TRANSFERS) or entry["transfers_received"] != (0 if rank == 0 else TRANSFERS):
            reason = f"rank {rank} incomplete"
        elif rank > 0 and entry["mismatches"] != 0:
            reason = f"rank {rank} mismatches"
        if reason:
            break
    transport = "nccl" if reason else "copy"
    result = {"schema": "firmus-pp4-copy-engine-probe/v1", "rule": {"slots": SLOTS, "width": WIDTH, "rows": ROWS, "transfers": TRANSFERS, "sleep_cycles": SLEEP_CYCLES}, "transport": transport, "reason": reason or "all-ranks-exact", "ranks": entries}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if transport == "copy":
        with env_output.open("a") as env:
            env.write("SGLANG_PP4_PROXY_TRANSPORT=copy\n")
    print(json.dumps({"transport": transport, "reason": result["reason"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="GPU result directory (probe) or result JSON (aggregate)")
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--env-output", type=Path)
    args = parser.parse_args()
    if args.aggregate:
        assert args.env_output is not None
        aggregate(args.aggregate, args.output, args.env_output)
    else:
        run_probe(args.output)
