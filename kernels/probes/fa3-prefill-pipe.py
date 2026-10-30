"""One-GPU paired timing of the serving Prefill FA3 rows (N192 varlen, two packed requests).

Rows come from fa3-prefill-pipe.so (PF_VARIANTS order, index 0 is the pinned
n192-direct-o-varlen). Every candidate is checked against an FP64 reference and
for byte equality with the pin, then timed in alternating CUDA-graph pairs.
A separate invocation with --stamp-library records per-tile consumer phase clocks of the pin.
"""
import argparse
import ctypes
import itertools
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch

LAYERS = 94
PAIRS = 20
LENGTHS = (3986, 4027)
SCALE = 1 / math.sqrt(128)


def work_counts(lengths, block_n=192):
    """M-tiles and KV-block visits of the PackGQA M128 (8 tokens x 16 heads) causal tiling."""
    tiles = visits = 0
    for length in lengths:
        for m in range((length * 16 + 127) // 128):
            tiles += 4
            visits += 4 * ((min(8 * (m + 1), length) + block_n - 1) // block_n)
    return visits, tiles


def inputs(device):
    generator = torch.Generator(device=device).manual_seed(20261030)
    rows = sum(LENGTHS)
    q = (torch.randn(rows, 64, 128, device=device, generator=generator) * 2).to(torch.float8_e4m3fn)
    k = torch.randn(rows, 4, 128, device=device, generator=generator).to(torch.float8_e4m3fn)
    v = torch.randn(rows, 4, 128, device=device, generator=generator).to(torch.float8_e4m3fn)
    cu = torch.tensor([0, LENGTHS[0], rows], dtype=torch.int32, device=device)
    return q, k, v, cu


class Library:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.pf_run.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p]
        self.count = self.lib.pf_count()

    def run(self, index, q, k, v, cu, out, scales, metadata):
        stream = torch.cuda.current_stream().cuda_stream
        status = self.lib.pf_run(index, q.shape[0], cu.numel() - 1, max(LENGTHS), q.data_ptr(), k.data_ptr(), v.data_ptr(), cu.data_ptr(), out.data_ptr(), scales.data_ptr(), scales.stride(0), metadata.data_ptr(), SCALE, stream)
        assert status == 0, status


def buffers(device):
    rows = sum(LENGTHS)
    out = torch.zeros(rows, 64, 128, dtype=torch.float8_e4m3fn, device=device)
    # (rows, 64) view with stride (1, padded rows), as the serving caller allocates it.
    padded = (rows + 3) // 4 * 4
    scales = torch.zeros(64, padded, dtype=torch.float32, device=device)
    metadata = torch.zeros(17, dtype=torch.int32, device=device)
    return out, scales, metadata


def reference(q, k, v, stride=61):
    """FP64 causal attention for sampled tokens of each request, all 64 heads."""
    result = {}
    start = 0
    for length in LENGTHS:
        tokens = sorted(set(range(0, length, stride)) | {length - 1})
        index = torch.tensor(tokens, device=q.device) + start
        qs = q[index].double()
        ks = k[start:start + length].double().repeat_interleave(16, dim=1)
        vs = v[start:start + length].double().repeat_interleave(16, dim=1)
        s = torch.einsum('thd,lhd->thl', qs, ks) * SCALE
        mask = torch.arange(length, device=q.device)[None, :] > torch.tensor(tokens, device=q.device)[:, None]
        s.masked_fill_(mask[:, None, :], float('-inf'))
        result[start] = (index, torch.einsum('thl,lhd->thd', torch.softmax(s, dim=-1), vs))
        start += length
    return result


def errors(out, scales, ref):
    num = den = 0.0
    worst = 0.0
    for index, expected in ref.values():
        got = out[index].float().double() * scales[:, index].t().double()[:, :, None]
        num += float(((got - expected) ** 2).sum())
        den += float((expected ** 2).sum())
        worst = max(worst, float(((got - expected).abs() / expected.abs().amax(dim=-1, keepdim=True)).max()))
    return {"relative_l2": math.sqrt(num / den), "max_scaled_error": worst}


def capture(library, index, args):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            library.run(index, *args)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(LAYERS):
                library.run(index, *args)
    torch.cuda.synchronize()
    return graph


def measure(graph):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / LAYERS


def sm_clock():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def read_stamps(library, index, q, k, v, cu):
    lib = library.lib
    args = (q, k, v, cu, *buffers(q.device))
    for _ in range(3):
        library.run(index, *args)
    torch.cuda.synchronize()
    assert lib.pf_stamp_reset() == 0
    library.run(index, *args)
    torch.cuda.synchronize()
    raw = (ctypes.c_ulonglong * (1 << 18))()
    count = ctypes.c_uint()
    assert lib.pf_stamp_read(raw, ctypes.byref(count)) == 0
    records = []
    for slot in range(min(count.value, 1 << 14)):
        meta, *t = raw[slot * 16: slot * 16 + 9]
        bidb, m_block = (meta >> 40) & 15, meta & 0xffffffff
        blocks = (min(8 * (m_block + 1), LENGTHS[bidb]) + 191) // 192
        records.append({"sm": meta >> 48, "wg": (meta >> 44) & 15, "bidb": bidb, "bidh": (meta >> 32) & 255, "m_block": m_block, "blocks": blocks, "t": t})
    lanes = {}
    for r in records:
        lanes.setdefault((r["sm"], r["wg"]), []).append(r)
    for items in lanes.values():
        items.sort(key=lambda r: r["t"][0])
    return records, lanes


def summarize(phases, per_step, records, steady="steady_loop"):
    totals = {name: sum(values) for name, values in phases.items()}
    whole = sum(totals.values())
    return {
        "records": len(records),
        "median_cycles": {name: statistics.median(values) for name, values in phases.items() if values},
        "share_of_consumer_cycles": {name: value / whole for name, value in totals.items()},
        "steady_cycles_per_step_median": statistics.median(per_step),
        "fixed_cycles_per_tile_mean": (whole - totals[steady]) / len(records),
    }


def pin_stamps(library, index, q, k, v, cu):
    records, lanes = read_stamps(library, index, q, k, v, cu)
    names = ("setup_q_wait", "first_qk", "first_softmax", "steady_loop", "drain", "scheduler", "epilogue")
    phases = {name: [] for name in (*names, "between_tiles")}
    per_step = []
    for r in records:
        t = r["t"]
        for i, name in enumerate(names):
            phases[name].append(t[i + 1] - t[i])
        if r["blocks"] > 1:
            per_step.append((t[4] - t[3]) / (r["blocks"] - 1))
    for items in lanes.values():
        for a, b in itertools.pairwise(items):
            phases["between_tiles"].append(b["t"][0] - a["t"][7])
    return summarize(phases, per_step, records)


def pipe_stamps(library, index, q, k, v, cu):
    """One record per tile boundary: t0 end of the tile's steady loop, then next-tile Q wait,
    ping-pong turn, boundary QK, next softmax, PV wait, finalize and store, hand-over."""
    records, lanes = read_stamps(library, index, q, k, v, cu)
    names = ("q_wait", "turn_wait", "boundary_qk", "next_softmax", "pv_wait", "finalize_store", "handover")
    phases = {name: [] for name in (*names, "steady_loop")}
    per_step = []
    complete = [r for r in records if r["t"][7]]
    for r in complete:
        t = r["t"]
        for i, name in enumerate(names):
            phases[name].append(t[i + 1] - t[i])
    for items in lanes.values():
        for a, b in itertools.pairwise(items):
            if a["t"][7]:
                phases["steady_loop"].append(b["t"][0] - a["t"][7])
                if b["blocks"] > 1:
                    per_step.append((b["t"][0] - a["t"][7]) / (b["blocks"] - 1))
    return summarize(phases, per_step, complete)


def stamps(path, names, q, k, v, cu):
    """Per-tile consumer phase clocks of each row of the stamp build ("pin" uses the mma() layout)."""
    library = Library(path)
    assert library.count == len(names), (library.count, names)
    return {name: (pin_stamps if name == "pin" else pipe_stamps)(library, index, q, k, v, cu) for index, name in enumerate(names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, help="timed rows, in PF_VARIANTS order")
    parser.add_argument("--names", help="comma-separated names in PF_VARIANTS order")
    parser.add_argument("--stamp-library", type=Path, help="instead of timing, record phase clocks of its rows")
    parser.add_argument("--stamp-names", help="comma-separated names of the stamp library rows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    begin = time.monotonic()
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    assert props.multi_processor_count == 132 and (props.major, props.minor) == (9, 0), props
    q, k, v, cu = inputs("cuda")
    result = {"device": props.name, "lengths": LENGTHS, "work": dict(zip(("visits", "tiles"), work_counts(LENGTHS))), "rows": {}}
    if args.stamp_library:
        result = {**result, "stamps": stamps(args.stamp_library, args.stamp_names.split(","), q, k, v, cu), "elapsed_seconds": time.monotonic() - begin}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("STAMPS " + json.dumps(result["stamps"]), flush=True)
        return
    names = args.names.split(",")
    library = Library(args.library)
    assert library.count == len(names), (library.count, names)
    ref = reference(q, k, v)
    outputs = {}
    graphs = []
    for index, name in enumerate(names):
        out, scales, metadata = buffers("cuda")
        library.run(index, q, k, v, cu, out, scales, metadata)
        torch.cuda.synchronize()
        outputs[name] = (out, scales)
        row = {"errors": errors(out, scales, ref)}
        if index:
            row["bytes_equal_pin"] = bool(torch.equal(out.view(torch.uint8), outputs[names[0]][0].view(torch.uint8)) and torch.equal(scales, outputs[names[0]][1]))
        result["rows"][name] = row
        out, scales, metadata = buffers("cuda")
        graphs.append(capture(library, index, (q, k, v, cu, out, scales, metadata)))
        if index:
            control, candidate = [], []
            for pair in range(PAIRS):
                order = ((control, graphs[0]), (candidate, graphs[index]))
                for samples, graph in (order if pair % 2 == 0 else order[::-1]):
                    samples.append(measure(graph))
            row.update(control_us=control, candidate_us=candidate, ratio=statistics.median(candidate) / statistics.median(control))
        print(f"ROW {name} " + json.dumps(row), flush=True)
    pin_error = result["rows"][names[0]]["errors"]["relative_l2"]
    for name in names[1:]:
        row = result["rows"][name]
        row["accurate"] = row["errors"]["relative_l2"] <= pin_error * 1.02 + 1e-6
    result["sm_clock_power"] = sm_clock()
    result["elapsed_seconds"] = time.monotonic() - begin
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = ", ".join(f"{name} {result['rows'][name]['ratio']:.4f} eq={result['rows'][name]['bytes_equal_pin']} l2={result['rows'][name]['errors']['relative_l2']:.5f}" for name in names[1:])
    print(f"PASS fa3-pipe pin l2={pin_error:.5f} {summary}", flush=True)


if __name__ == "__main__":
    main()
