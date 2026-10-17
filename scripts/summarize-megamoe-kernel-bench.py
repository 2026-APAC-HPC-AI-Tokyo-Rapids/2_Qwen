"""Summarize a Firmus MegaMoE kernel bench: timing table and per-phase timeline.

Input: the job's results directory, holding bench.jsonl and timeline.jsonl from
bench-firmus-megamoe.py and timeline/<case>.npy dumps shaped
[calls, ranks, words] in the layout of sm90_fp8_mega_moe_timeline.cuh. Times are
microseconds since the rank's first CTA entry. Within a CTA, event times use
clock64 deltas from the CTA entry converted with that CTA's measured SM clock
(clock64 / %globaltimer over the CTA lifetime); CTA entries are aligned with
%globaltimer. The first call is skipped; the representative call is the one
with the median kernel time over the remaining calls.

Named-barrier handshakes between dispatch and math warps (barrier.sync 1)
compile to BAR.SYNC.DEFER_BLOCKING, and the %globaltimer read placed after them
retires before the barrier releases (run 39053820377: math "passes" its
handshake 9 us before dispatch arrives). Phases therefore never end at those
two events; the wait is attributed to the following phase instead. Grid-sync
and NVLink-barrier events are taken by the spinning thread and are exact.
"""
import argparse
import json
from pathlib import Path

import numpy as np

MAGIC = 0x314C544D4F45474D
EVENTS = ["Entry", "DispCount", "DispSendCount", "DispGridSync", "DispRecvCount", "DispPrePullBarrier", "DispEpilogueSync", "DispFirstPull", "DispLastPull", "DispLoopExit", "DispCombineSync", "DispClean", "DispFinalGridSync", "DispExit", "AStart", "AFirstBlock", "AFirstL1Ready", "ALastL1Issued", "AFirstL2Ready", "AExit", "BStart", "BFirstBlock", "BExit", "MStart", "MDispatchSync", "MFirstL1Full", "MLastL1End", "MFirstL2Full", "MLastL2End", "MLoopExit", "MCombineGridSync", "MCombineBarrier", "MDispatchSync2", "MCombineEnd"]
COUNTERS = ["DispTokensW0", "DispTokensW1", "DispPullWaitW0", "DispPullWaitW1", "AL1Blocks", "AL2Blocks", "AL1ArrivalWait", "AL2ArrivalWait", "AEmptyWait", "BEmptyWait", "ML1Blocks", "ML2Blocks", "ML1FullWait", "ML2FullWait", "ML1Busy", "ML2Busy", "MCombineTokensW0", "MCombineWaitW0", "BBlocks", "ML1Loop", "ML2Loop"]
CYCLE_COUNTERS = {"DispPullWaitW0", "DispPullWaitW1", "AL1ArrivalWait", "AL2ArrivalWait", "AEmptyWait", "BEmptyWait", "ML1FullWait", "ML2FullWait", "ML1Busy", "ML2Busy", "MCombineWaitW0", "ML1Loop", "ML2Loop"}
E = {name: index for index, name in enumerate(EVENTS)}
PHASES = [
    ("dispatch: count routed tokens", "Entry", "DispCount"),
    ("dispatch: per-expert send-count atomics", "DispCount", "DispSendCount"),
    ("dispatch: remote source-index writes + grid sync", "DispSendCount", "DispGridSync"),
    ("dispatch: publish recv counts (SM0)", "DispGridSync", "DispRecvCount"),
    ("dispatch: pre-pull NVLink barrier", "DispRecvCount", "DispPrePullBarrier"),
    ("dispatch: barrier -> first token pulled (warp 0)", "DispPrePullBarrier", "DispFirstPull"),
    ("dispatch: first -> last token pulled (warp 0)", "DispFirstPull", "DispLastPull"),
    ("math: entry -> first FC1 stage ready (incl. dispatch handshake)", "MStart", "MFirstL1Full"),
    ("math: FC1 span (first ready -> last end)", "MFirstL1Full", "MLastL1End"),
    ("math: last FC1 end -> first FC2 stage ready", "MLastL1End", "MFirstL2Full"),
    ("math: FC2 span (first ready -> last end)", "MFirstL2Full", "MLastL2End"),
    ("math: last block -> loop exit", "MLastL2End", "MLoopExit"),
    ("math: pre-combine grid sync (intra-rank straggler)", "MLoopExit", "MCombineGridSync"),
    ("math: pre-combine NVLink barrier + grid sync", "MCombineGridSync", "MCombineBarrier"),
    ("math: combine (after barrier)", "MCombineBarrier", "MCombineEnd"),
    ("dispatch: loop exit -> cleanup done (incl. combine handshake)", "DispLoopExit", "DispClean"),
    ("dispatch: final grid sync", "DispClean", "DispFinalGridSync"),
    ("dispatch: final NVLink barrier (SM0 only waits)", "DispFinalGridSync", "DispExit"),
]


def parse(words):
    assert words[0] == MAGIC, hex(int(words[0]))
    num_sms, num_events, num_counters = (int(x) for x in words[2:5])
    events = words[16:16 + num_sms * num_events * 2].reshape(num_sms, num_events, 2).astype(np.float64)
    counters = words[16 + num_sms * num_events * 2:16 + num_sms * num_events * 2 + num_sms * num_counters].reshape(num_sms, num_counters).astype(np.float64)
    ns, clk = events[:, :, 0].copy(), events[:, :, 1].copy()
    missing = ns == 0
    ns[missing], clk[missing] = np.nan, np.nan
    last = np.nanargmax(ns, axis=1)
    rows = np.arange(num_sms)
    ghz = (clk[rows, last] - clk[:, 0]) / (ns[rows, last] - ns[:, 0])
    start = np.nanmin(ns[:, 0])
    times = (ns[:, :1] - start + (clk - clk[:, :1]) / ghz[:, None]) / 1000.0
    counter_us = counters.copy()
    for name in CYCLE_COUNTERS:
        counter_us[:, COUNTERS.index(name)] = counters[:, COUNTERS.index(name)] / ghz / 1000.0
    end = np.nanmax(np.fmax(times[:, E["DispExit"]], times[:, E["MCombineEnd"]]))
    header = {"num_sms": num_sms, "num_tokens": int(words[5]), "rank": int(words[6]), "block_m": int(words[7]), "block_n": int(words[8]), "epilogue_threads": int(words[9]), "stages": int(words[10])}
    granularity = np.diff(np.unique(ns[~np.isnan(ns)]))
    return {"times": times, "counters": counter_us, "end": end, "ghz": ghz, "header": header, "granularity_ns": float(np.min(granularity)) if granularity.size else float("nan")}


def cell(values, suffix=""):
    values = values[~np.isnan(values)]
    if not values.size:
        return "-"
    return f"{np.median(values):.1f} / {np.max(values):.1f}{suffix}"


def report(path):
    data = np.load(path)
    calls = [[parse(data[call, rank]) for rank in range(data.shape[1])] for call in range(data.shape[0])]
    kernel = np.array([max(rank["end"] for rank in call) for call in calls[1:]])
    chosen = calls[1 + int(np.argsort(kernel)[len(kernel) // 2])]
    ranks = range(len(chosen))
    header = chosen[0]["header"]
    lines = [f"### {path.stem}", "", f"Geometry from the kernel header: BLOCK_M {header['block_m']}, BLOCK_N {header['block_n']}, {header['epilogue_threads']} math threads, {header['stages']} stages, {header['num_sms']} CTAs, {header['num_tokens']} tokens/rank.", f"Kernel time (max over ranks of last CTA exit since that rank's first CTA entry), calls 2-{len(calls)}: median {np.median(kernel):.1f} us, min {kernel.min():.1f}, max {kernel.max():.1f}. SM clock {np.nanmedian(np.concatenate([rank['ghz'] for rank in chosen])):.3f} GHz; %globaltimer step {min(rank['granularity_ns'] for rank in chosen):.0f} ns.", ""]
    lines += ["Per-phase duration, us (median over CTAs / max over CTAs):", "", "| Phase | " + " | ".join(f"rank {rank}" for rank in ranks) + " |", "| --- |" + " ---: |" * len(chosen)]
    for label, begin, finish in PHASES:
        lines.append(f"| {label} | " + " | ".join(cell(rank["times"][:, E[finish]] - rank["times"][:, E[begin]]) for rank in chosen) + " |")
    lines += ["", "Event time since the rank's kernel start, us (median over CTAs / max over CTAs):", "", "| Event | " + " | ".join(f"rank {rank}" for rank in ranks) + " |", "| --- |" + " ---: |" * len(chosen)]
    for name in EVENTS:
        lines.append(f"| {name} | " + " | ".join(cell(rank["times"][:, E[name]]) for rank in chosen) + " |")
    lines.append("| kernel end (max over CTAs) | " + " | ".join(f"{rank['end']:.1f}" for rank in chosen) + " |")
    lines += ["", "Per-CTA counters (median / max over CTAs; waits and busy time in us):", "", "| Counter | " + " | ".join(f"rank {rank}" for rank in ranks) + " |", "| --- |" + " ---: |" * len(chosen)]
    for index, name in enumerate(COUNTERS):
        lines.append(f"| {name} | " + " | ".join(cell(rank["counters"][:, index]) for rank in chosen) + " |")
    lines += ["", "Where the time goes (representative call):", ""]
    for rank_index, rank in zip(ranks, chosen):
        t, c = rank["times"], rank["counters"]
        busy = c[:, COUNTERS.index("ML1Busy")] + c[:, COUNTERS.index("ML2Busy")]
        waits = c[:, COUNTERS.index("ML1FullWait")] + c[:, COUNTERS.index("ML2FullWait")]
        first_fc1 = t[:, E["MFirstL1Full"]]
        last_fc2 = np.nanmax(t[:, E["MLastL2End"]])
        lines.append(f"- rank {rank_index}: kernel {rank['end']:.1f} us at SM clock {np.nanmedian(rank['ghz']):.3f} GHz; entry -> first FC1 stage ready median {np.nanmedian(first_fc1):.1f} / min {np.nanmin(first_fc1):.1f} us; GEMM window {np.nanmin(first_fc1):.1f} -> {last_fc2:.1f} us; math block time minus stage waits = {np.nansum(busy - waits) / (len(busy) * rank['end']):.1%} of CTA x kernel time (stage waits {np.nansum(waits) / (len(busy) * rank['end']):.1%}); last FC2 end -> kernel end {rank['end'] - last_fc2:.1f} us; combine phase (barrier done -> combine end) median {np.nanmedian(t[:, E['MCombineEnd']] - t[:, E['MCombineBarrier']]):.1f} us.")
    return "\n".join(lines) + "\n"


def timing(results):
    lines = ["### Kernel timing (pre-dispatch + fused kernel, max over ranks per iteration)", "", "| Case | Variant | Median ms | p10 ms | p90 ms | Rank load max/mean | Expert load max/mean | Deterministic | Note |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for name in ("bench.jsonl", "timeline.jsonl"):
        path = results / name
        if not path.exists():
            continue
        for row in map(json.loads, path.read_text().splitlines()):
            note = ""
            if "overhead_median" in row:
                note = f"bitwise equal to {row['reference_variant']}: {row['bitwise_equal_to_reference']}; overhead {row['overhead_median']:+.1%}"
            lines.append(f"| {row['case']} | {row['variant']} | {row['median_ms']:.4f} | {row['p10_ms']:.4f} | {row['p90_ms']:.4f} | {row['rank_load_max_over_mean']:.3f} | {row['expert_load_max_over_mean']:.2f} | {row['deterministic']} | {note} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(timing(args.results))
    print("\n".join(report(path) for path in sorted((args.results / "timeline").glob("*.npy"))))


if __name__ == "__main__":
    main()
