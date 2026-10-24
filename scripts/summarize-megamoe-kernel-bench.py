"""Summarize a Firmus kernel bench: candidate verdicts, timing, timeline, SGLang.

Input: the job's results directory (layout v2, since the multi-candidate bench):
variants.json from provision-firmus-kernel-bench.sh, steps.tsv from run.sbatch
(variant, step, exit code, seconds), <variant>/bench.jsonl and, when requested,
<variant>/timeline.jsonl and <variant>/timeline/<case>.npy from
bench-firmus-megamoe.py, and sglang/<image|candidate>.jsonl from
bench-firmus-sglang-kernels.py. Archives from the single-candidate layout need
the script as of commit 332e30c.

Exit status 1 marks any failure: a non-zero step (including a candidate's
numerical gate or SGLang check), a missing case, or a timeline patch that did
not apply. Replays must be deterministic. The four arithmetic-preserving Decode
candidates below must also match the control's output hashes. For other
candidates, changed hashes remain informational: their numerical gates decide
whether a deliberately changed accumulation order or quantization is acceptable.

Timeline dumps are shaped [calls, ranks, words] in the layout of
sm90_fp8_mega_moe_timeline.cuh. Times are
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

# These candidates promise to retain each output element's arithmetic. Use the
# recorded patch identity, so renaming a candidate directory cannot relax this.
EXACT_OUTPUT_PATCHES = {"sm90-decode-sparse-dispatch.patch", "sm90-decode-fc1-register.patch", "sm90-decode-independent-wg.patch", "stream-combine.patch"}

MAGIC = 0x314C544D4F45474D
EVENTS = ["Entry", "DispCount", "DispSendCount", "DispGridSync", "DispRecvCount", "DispPrePullBarrier", "DispEpilogueSync", "DispFirstPull", "DispLastPull", "DispLoopExit", "DispCombineSync", "DispClean", "DispFinalGridSync", "DispExit", "AStart", "AFirstBlock", "AFirstL1Ready", "ALastL1Issued", "AFirstL2Ready", "AExit", "BStart", "BFirstBlock", "BExit", "MStart", "MDispatchSync", "MFirstL1Full", "MLastL1End", "MFirstL2Full", "MLastL2End", "MLoopExit", "MCombineGridSync", "MCombineBarrier", "MDispatchSync2", "MCombineEnd"]
COUNTERS = ["DispTokensW0", "DispTokensW1", "DispPullWaitW0", "DispPullWaitW1", "AL1Blocks", "AL2Blocks", "AL1ArrivalWait", "AL2ArrivalWait", "AEmptyWait", "BEmptyWait", "ML1Blocks", "ML2Blocks", "ML1FullWait", "ML2FullWait", "ML1Busy", "ML2Busy", "MCombineTokensW0", "MCombineWaitW0", "BBlocks", "ML1Loop", "ML2Loop", "DispIndexW0", "DispRemoteW0", "DispStoreW0"]
CYCLE_COUNTERS = {"DispPullWaitW0", "DispPullWaitW1", "AL1ArrivalWait", "AL2ArrivalWait", "AEmptyWait", "BEmptyWait", "ML1FullWait", "ML2FullWait", "ML1Busy", "ML2Busy", "MCombineWaitW0", "ML1Loop", "ML2Loop", "DispIndexW0", "DispRemoteW0", "DispStoreW0"}
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


# v2 keeps synchronization intact and records both math WGs (up to four for
# archived 4-WG instantiations). Each row has a unique writer WG leader.
WG_EVENTS = ["Start", "FirstFC1", "LastFC1", "FirstFC2", "LastFC2", "LoopExit", "BarrierEnd", "CombineEnd"]
WG_COUNTERS = ["FC1Blocks", "FC2Blocks", "FC1Wait", "FC2Wait", "FC1Loop", "FC2Loop", "FC1Epilogue", "FC2Epilogue"]
V2_EVENTS = EVENTS[:14] + [f"WG{wg}{event}" for wg in range(4) for event in WG_EVENTS]
V2_COUNTERS = [f"WG{wg}{counter}" for wg in range(4) for counter in WG_COUNTERS]


def report_warpgroups(chosen, title, kernel, calls):
    header = chosen[0]["header"]
    groups = range(header["math_warpgroups"])
    lines = [f"### {title}", "", f"Geometry: BLOCK_M {header['block_m']}, BLOCK_N {header['block_n']}, {header['epilogue_threads']} math threads ({header['math_warpgroups']} WGs), {header['stages']} stages, {header['num_sms']} CTAs, {header['num_tokens']} tokens/rank.", f"Kernel time, calls 2-{calls}: median {np.median(kernel):.1f} us, min {kernel.min():.1f}, max {kernel.max():.1f}. SM clock {np.nanmedian(np.concatenate([rank['ghz'] for rank in chosen])):.3f} GHz.", "", "Per-CTA duration, us (median / max); counters sum all blocks in each WG. Mainloop includes setup, stage waits and FP32 promotion; the separate stage-wait row is a subset. Epilogue includes SwiGLU/FP8/store for FC1 and BF16/NVLink scatter for FC2, including existing epilogue barriers. Prefill FC1's deferred publication remains in the following mainloop or scheduler transition, as in serving. The first-ready interval includes dispatch handshake and loader/stage arrival; it is not pure dispatch time. Pre-combine wait includes intra-rank and inter-rank synchronization together. No barrier was split or added for measurement.", "", "| WG | Phase | " + " | ".join(f"rank {rank}" for rank in range(len(chosen))) + " |", "| --- | --- |" + " ---: |" * len(chosen)]
    for wg in groups:
        def event(rank, name, wg=wg):
            return rank["times"][:, V2_EVENTS.index(f"WG{wg}{name}")]

        def counter(rank, name, wg=wg):
            return rank["counters"][:, V2_COUNTERS.index(f"WG{wg}{name}")]

        phases = [
            ("dispatch/first-stage wait (WG start -> first FC1 ready)", lambda r: event(r, "FirstFC1") - event(r, "Start")),
            ("FC1 mainloop", lambda r: counter(r, "FC1Loop")),
            ("FC1 stage wait (within mainloop)", lambda r: counter(r, "FC1Wait")),
            ("FC1 epilogue (SwiGLU + FP8 + store)", lambda r: counter(r, "FC1Epilogue")),
            ("FC2 mainloop", lambda r: counter(r, "FC2Loop")),
            ("FC2 stage wait (within mainloop)", lambda r: counter(r, "FC2Wait")),
            ("FC2 epilogue (BF16 + NVLink scatter)", lambda r: counter(r, "FC2Epilogue")),
            ("last FC2 -> loop exit (scheduler/publication tail)", lambda r: event(r, "LoopExit") - event(r, "LastFC2")),
            ("pre-combine wait (grid + NVLink)", lambda r: event(r, "BarrierEnd") - event(r, "LoopExit")),
            ("combine (incl. dispatch cleanup handshake)", lambda r: event(r, "CombineEnd") - event(r, "BarrierEnd")),
        ]
        for label, values in phases:
            lines.append(f"| {wg} | {label} | " + " | ".join(cell(values(rank)) for rank in chosen) + " |")
        for name in ("FC1Blocks", "FC2Blocks"):
            lines.append(f"| {wg} | {name} (count) | " + " | ".join(cell(counter(rank, name)) for rank in chosen) + " |")
    lines += ["", "Event time since rank entry, us (median / max over CTAs):", "", "| Event | " + " | ".join(f"rank {rank}" for rank in range(len(chosen))) + " |", "| --- |" + " ---: |" * len(chosen)]
    names = V2_EVENTS[:12] + [f"WG{wg}{event}" for wg in groups for event in WG_EVENTS] + ["DispExit"]
    for name in names:
        lines.append(f"| {name} | " + " | ".join(cell(rank["times"][:, V2_EVENTS.index(name)]) for rank in chosen) + " |")
    lines += ["", "Instrumented durations include measurement overhead; use the uninstrumented bench for performance comparisons. BAR.SYNC.DEFER_BLOCKING can retire the immediate timestamp before release; no phase ends at the dispatch/math handshake timestamp. Missing event timestamps are shown as '-'."]
    return "\n".join(lines) + "\n"


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
    version = int(words[1])
    assert version in (1, 2), f"Unknown timeline version {version}"
    counter_us = counters.copy()
    if version == 2:
        assert num_events == len(V2_EVENTS) and num_counters == len(V2_COUNTERS)
        assert int(words[13]) in (2, 4), "Unexpected math WG count"
        for index, name in enumerate(V2_COUNTERS):
            if not name.endswith("Blocks"):
                counter_us[:, index] = counters[:, index] / ghz / 1000.0
        exits = [V2_EVENTS.index("DispExit")] + [V2_EVENTS.index(f"WG{wg}CombineEnd") for wg in range(int(words[13]))]
        end = np.nanmax(times[:, exits])
    else:
        for name in CYCLE_COUNTERS:
            counter_us[:, COUNTERS.index(name)] = counters[:, COUNTERS.index(name)] / ghz / 1000.0
        end = np.nanmax(np.fmax(times[:, E["DispExit"]], times[:, E["MCombineEnd"]]))
    header = {"num_sms": num_sms, "num_tokens": int(words[5]), "rank": int(words[6]), "block_m": int(words[7]), "block_n": int(words[8]), "epilogue_threads": int(words[9]), "stages": int(words[10])}
    header.update(version=version, math_warpgroups=int(words[13]) if version == 2 else int(words[9]) // 128)
    granularity = np.diff(np.unique(ns[~np.isnan(ns)]))
    return {"times": times, "counters": counter_us, "end": end, "ghz": ghz, "header": header, "granularity_ns": float(np.min(granularity)) if granularity.size else float("nan")}


def cell(values, suffix=""):
    values = values[~np.isnan(values)]
    if not values.size:
        return "-"
    return f"{np.median(values):.1f} / {np.max(values):.1f}{suffix}"


def report(path, title):
    data = np.load(path)
    calls = [[parse(data[call, rank]) for rank in range(data.shape[1])] for call in range(data.shape[0])]
    kernel = np.array([max(rank["end"] for rank in call) for call in calls[1:]])
    chosen = calls[1 + int(np.argsort(kernel)[len(kernel) // 2])]
    ranks = range(len(chosen))
    header = chosen[0]["header"]
    if header["version"] == 2:
        return report_warpgroups(chosen, title, kernel, len(calls))
    lines = [f"### {title}", "", f"Geometry from the kernel header: BLOCK_M {header['block_m']}, BLOCK_N {header['block_n']}, {header['epilogue_threads']} math threads, {header['stages']} stages, {header['num_sms']} CTAs, {header['num_tokens']} tokens/rank.", f"Kernel time (max over ranks of last CTA exit since that rank's first CTA entry), calls 2-{len(calls)}: median {np.median(kernel):.1f} us, min {kernel.min():.1f}, max {kernel.max():.1f}. SM clock {np.nanmedian(np.concatenate([rank['ghz'] for rank in chosen])):.3f} GHz; %globaltimer step {min(rank['granularity_ns'] for rank in chosen):.0f} ns.", ""]
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


def load_rows(path):
    return {row["case"]: row for row in map(json.loads, path.read_text().splitlines())} if path.exists() else {}


def slowest_clock(row):
    # Kernel time follows the slowest GPU. The two NVML samples bracket the
    # timed loop but are instantaneous: in run 39085655145 identical Prefill
    # kernels read 1545-1935 MHz at the same median, so they flag throttled
    # cases and are not used to normalize times.
    return f"{min(row['sm_clock_mhz_before']):.0f} / {min(row['sm_clock_mhz_after']):.0f}"


def step_cell(steps, variant, name):
    if (variant, name) not in steps:
        return "not run", False
    code = steps[(variant, name)]
    return ("pass" if code == 0 else f"FAIL (exit {code})"), code == 0


def deep_gemm(results, steps):
    variants = json.loads((results / "variants.json").read_text())["deep_gemm"]
    rows = {variant["name"]: load_rows(results / variant["name"] / "bench.jsonl") for variant in variants}
    image = rows["image"]
    cases = list(image)
    control_valid = steps.get(("image", "bench")) == 0 and bool(image) and all(row.get("deterministic") is True for row in image.values())
    failed = []
    lines = ["### DeepGEMM candidates", "", "| Variant | Patches | Gate | Bench | Output hash = image | Timeline | Verdict |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for variant in variants:
        name = variant["name"]
        ok = True
        gate, passed = ("- (image)", True) if name == "image" else step_cell(steps, name, "gate")
        ok &= passed
        bench, passed = step_cell(steps, name, "bench")
        ok &= passed and bool(cases) and set(rows[name]) == set(cases)
        deterministic = all(row.get("deterministic") is True for row in rows[name].values())
        ok &= deterministic
        if not deterministic:
            bench += " (non-deterministic or missing replay check)"
        hashes = "-"
        if name != "image":
            ok &= control_valid
            if not control_valid:
                bench += " (invalid control bench or replay)"
            matches = sum(rows[name].get(case, {}).get("output_sha256") == image[case]["output_sha256"] for case in cases)
            hashes = f"{matches}/{len(cases)}" + ("" if matches == len(cases) else " (rest differ)")
            if any(Path(patch["path"]).name in EXACT_OUTPUT_PATCHES for patch in variant["patches"]):
                ok &= matches == len(cases)
        timeline = variant["timeline"]
        if timeline == "apply-failed":
            timeline, ok = "timeline patch did not apply", False
        elif timeline == "ready":
            timeline, passed = step_cell(steps, name, "timeline")
            ok &= passed
        if not ok:
            failed.append(name)
        patches = ", ".join(patch["path"].rsplit("/", 1)[-1] for patch in variant["patches"]) or "-"
        # The candidate's env file (KEY=VALUE lines); archives staged before it existed have no "env".
        if variant.get("env"):
            patches += "; env " + " ".join(variant["env"])
        lines.append(f"| {name} | {patches} | {gate} | {bench} | {hashes} | {timeline} | {'OK' if ok else 'FAILED'} |")
    names = [variant["name"] for variant in variants]
    lines += ["", "### Kernel timing, median ms (delta vs image; * = output hash differs from image)", "", "| Case | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names)]
    for case in cases:
        cells = []
        for name in names:
            row = rows[name].get(case)
            if row is None:
                cells.append("-")
            elif name == "image":
                cells.append(f"{row['median_ms']:.4f}")
            else:
                mark = "" if row["output_sha256"] == image[case]["output_sha256"] else "*"
                cells.append(f"{row['median_ms']:.4f} ({row['median_ms'] / image[case]['median_ms'] - 1:+.1%}){mark}")
        lines.append(f"| {case} | " + " | ".join(cells) + " |")
    lines += ["", "Per variant detail (pre-dispatch + fused kernel, max over ranks per iteration). SM MHz is the lowest NVML SM clock over the GPUs, sampled right before / right after the timed iterations (instantaneous; the timeline's per-CTA clock is the accurate in-kernel value).", "", "| Case | Variant | Median ms | p10 ms | p90 ms | Delta | SM MHz before / after | Power W (max rank) | Rank load max/mean | Deterministic | Hash = image |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for case in cases:
        reference = image[case]
        for name in names:
            row = rows[name].get(case)
            if row is None:
                continue
            delta = "-" if name == "image" else f"{row['median_ms'] / reference['median_ms'] - 1:+.1%}"
            same = "-" if name == "image" else str(row["output_sha256"] == reference["output_sha256"])
            power = max(max(row["power_w_before"]), max(row["power_w_after"]))
            lines.append(f"| {case} | {name} | {row['median_ms']:.4f} | {row['p10_ms']:.4f} | {row['p90_ms']:.4f} | {delta} | {slowest_clock(row)} | {power:.0f} | {row['rank_load_max_over_mean']:.3f} | {row.get('deterministic', 'missing')} | {same} |")
    timelines = [(variant["name"], load_rows(results / variant["name"] / "timeline.jsonl")) for variant in variants if variant["timeline"] == "ready"]
    if any(timeline for _, timeline in timelines):
        lines += ["", "### Timeline runs (instrumented vs the same variant's bench)", "", "| Case | Variant | Median ms | Overhead | Bitwise equal | SM MHz before / after |", "| --- | --- | ---: | ---: | --- | ---: |"]
        for name, timeline in timelines:
            for case, row in timeline.items():
                lines.append(f"| {case} | {name} | {row['median_ms']:.4f} | {row['delta_median_vs_reference']:+.1%} | {row['bitwise_equal_to_reference']} | {slowest_clock(row)} |")
    reports = [report(path, f"{name}: {path.stem}") for name, _ in timelines for path in sorted((results / name / "timeline").glob("*.npy"))]
    return "\n".join(lines) + "\n\n" + "\n".join(reports), failed


def sglang(results, steps):
    config = json.loads((results / "variants.json").read_text())["sglang"]
    if config is None:
        return "", []
    failed = []
    lines = ["### SGLang JIT kernels (image tree vs patched tree)", "", "Patches: " + ", ".join(patch["path"].rsplit("/", 1)[-1] for patch in config["patches"]), "", "| Tree | Decode-kernel check | Bench |", "| --- | --- | --- |"]
    for tree in ("image", "candidate"):
        check, passed = step_cell(steps, f"sglang-{tree}", "check")
        if not passed:
            failed.append(f"sglang-{tree} check")
        bench = "no bench script"
        if config["bench_script"]:
            bench, passed = step_cell(steps, f"sglang-{tree}", "bench")
            if not passed:
                failed.append(f"sglang-{tree} bench")
        lines.append(f"| {tree} | {check} | {bench} |")
    if config["bench_script"]:
        image, candidate = load_rows(results / "sglang" / "image.jsonl"), load_rows(results / "sglang" / "candidate.jsonl")
        lines += ["", "| Case | Image median us | Candidate median us | Delta | Candidate p10 / p90 us | Hash equal |", "| --- | ---: | ---: | ---: | ---: | --- |"]
        for case in dict.fromkeys(list(image) + list(candidate)):
            base, row = image.get(case), candidate.get(case)
            if base is None or row is None:
                failed.append(f"sglang {case} missing")
                lines.append(f"| {case} | {base['median_us'] if base else '-'} | {row['median_us'] if row else '-'} | - | - | - |")
                continue
            same = base["output_sha256"] == row["output_sha256"]
            lines.append(f"| {case} | {base['median_us']:.2f} | {row['median_us']:.2f} | {row['median_us'] / base['median_us'] - 1:+.1%} | {row['p10_us']:.2f} / {row['p90_us']:.2f} | {same} |")
    return "\n".join(lines) + "\n", failed


def summarize(results):
    steps = {}
    if (results / "steps.tsv").exists():
        for line in (results / "steps.tsv").read_text().splitlines():
            variant, name, code, _ = line.split("\t")
            steps[(variant, name)] = int(code)
    text, failed = deep_gemm(results, steps)
    sglang_text, sglang_failed = sglang(results, steps)
    failed += sglang_failed
    verdict = "All variants passed." if not failed else "FAILED: " + ", ".join(failed)
    return f"### Verdict\n\n{verdict}\n\n{text}\n{sglang_text}", failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    text, failed = summarize(args.results)
    print(text)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
