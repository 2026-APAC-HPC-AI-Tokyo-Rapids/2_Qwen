#!/usr/bin/env python3
"""Read-only Nsight kernel budgets for the 94-layer target baseline.

One cycle runs from target MegaMoE 0 to MegaMoE 94. This rotation includes
the previous forward's tail and the next forward's prologue. It does not
separate draft work or infer tensor shapes. Count checks support alignment;
they cannot prove layer identity without an external layer marker.
"""
import argparse
import bisect
import collections
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics


LAYERS = 94
GEOMETRY = ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ",
            "clusterX", "clusterY", "clusterZ")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def columns(db, table):
    return {r[1] for r in db.execute(f'PRAGMA table_info("{table}")')}


def records(db, table, device=None):
    fields = columns(db, table)
    if not fields:
        return []
    query = f'SELECT * FROM "{table}"'
    parameters = ()
    if device is not None:
        query += ' WHERE deviceId = ?'
        parameters = (device,)
    return [dict(r) for r in db.execute(query, parameters)]


def union_ns(intervals):
    total = 0
    end = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
        elif b > end:
            total += b - end
        end = max(end, b) if end is not None else b
    return total


def percentile(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def marker_counts(rows):
    names = [r["name"] for r in rows]
    return {
        "target_megamoe": sum("sm90_fp8_mega_moe_impl" in n for n in names),
        "dense_fp8_qkv_o_family": sum("sm90_fp8_gemm_1d2d_impl" in n for n in names),
        "fa3_forward_family": sum("FlashAttnFwdSm90" in n for n in names),
    }


def contained(rows, lo, hi):
    return [r for r in rows if lo <= r["start"] < hi and r["end"] <= hi]


def summarize(rows, cycles, key_fields, overlap_rows, window):
    grouped = collections.defaultdict(list)
    overlapping = collections.defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in key_fields)].append(row)
    for row in overlap_rows:
        overlapping[tuple(row.get(k) for k in key_fields)].append(row)
    output = []
    for key in grouped.keys() | overlapping.keys():
        group = grouped[key]
        ns = [r["end"] - r["start"] for r in group]
        lo, hi = window
        overlap_ns = sum(min(r["end"], hi) - max(r["start"], lo) for r in overlapping[key])
        item = dict(zip(key_fields, key))
        item.update(calls=len(group), total_ms=sum(ns) / 1e6,
                    mean_us=statistics.mean(ns) / 1e3 if ns else None,
                    min_us=min(ns) / 1e3 if ns else None,
                    max_us=max(ns) / 1e3 if ns else None,
                    per_cycle_ms=sum(ns) / 1e6 / cycles,
                    window_overlap_calls=len(overlapping[key]),
                    window_overlap_ms=overlap_ns / 1e6,
                    per_cycle_overlap_ms=overlap_ns / 1e6 / cycles)
        if group and "bytes" in group[0]:
            item["total_bytes"] = sum(r.get("bytes") or 0 for r in group)
            item["bytes_per_cycle"] = item["total_bytes"] / cycles
        output.append(item)
    return sorted(output, key=lambda r: (-r["window_overlap_ms"], str(tuple(r.get(k) for k in key_fields))))


def clock_summary(db, device, windows):
    """Only accept an explicit device mapping; do not decode typeId bits."""
    metrics = records(db, "TARGET_INFO_GPU_METRICS")
    candidates = [r for r in metrics if "GPC Clock Frequency" in r.get("metricName", "")]
    raw_summaries = []
    for metric in candidates:
        values = [float(r[1]) for r in db.execute("SELECT timestamp, value FROM GPU_METRICS WHERE typeId=? AND metricId=?", (metric["typeId"], metric["metricId"]))
                  if any(a <= r[0] < b for a, b in windows)]
        if values:
            raw_summaries.append(dict(metric=metric, count=len(values), mean=statistics.mean(values),
                                      p10=percentile(values, .1), p90=percentile(values, .9),
                                      min=min(values), max=max(values),
                                      units="Raw GPU_METRICS.value; declared metric label preserved, no unit conversion"))
    mapped = []
    for row in candidates:
        explicit = row.get("deviceId", row.get("gpuId"))
        label = row.get("typeName", "")
        matches = re.findall(r"\b(?:GPU|Device)\s*[:#]?\s*(\d+)\b", label, re.I)
        if explicit == device or (len(matches) == 1 and int(matches[0]) == device):
            mapped.append(row)
    evidence = {"candidate_metrics": candidates, "raw_clock_series_summary": raw_summaries,
                "mapping_policy": "Explicit deviceId/gpuId or GPU/Device number in typeName; no encoded-typeId inference."}
    if len(mapped) != 1:
        return None, dict(evidence, reason="Unique explicit metric-to-device mapping unavailable")
    metric = mapped[0]
    values = list(db.execute("SELECT timestamp, value FROM GPU_METRICS WHERE typeId=? AND metricId=? ORDER BY timestamp", (metric["typeId"], metric["metricId"])))
    samples = [float(r[1]) for r in values if any(a <= r[0] < b for a, b in windows)]
    # Nsight exports may label MHz while storing Hz. Preserve raw data and
    # refuse ambiguous units rather than silently applying a scale.
    if not samples or max(samples) > 10000:
        return None, dict(evidence, selected_metric=metric,
                          reason="No window samples or metric values incompatible with declared MHz",
                          raw_sample_min=min(samples) if samples else None,
                          raw_sample_max=max(samples) if samples else None)
    return {"metric": metric, "sample_count": len(samples),
            "sample_mean_mhz": statistics.mean(samples),
            "sample_p10_mhz": percentile(samples, .1),
            "sample_p90_mhz": percentile(samples, .9),
            "averaging": "Arithmetic sample mean within selected windows, not time weighted"}, evidence


def extract(db, role, run, device=0, discard_cycles=2):
    if discard_cycles < 0:
        raise ValueError("discard-cycles must be nonnegative")
    db.row_factory = sqlite3.Row
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    kernels = records(db, "CUPTI_ACTIVITY_KIND_KERNEL", device)
    for row in kernels:
        row["name"] = strings.get(row["demangledName"], str(row["demangledName"]))
    kernels.sort(key=lambda r: (r["start"], r["end"]))
    moes = [r for r in kernels if "sm90_fp8_mega_moe_impl" in r["name"]]
    anchors = moes[::LAYERS]
    intervals = list(zip([r["start"] for r in anchors], [r["start"] for r in anchors[1:]]))
    if len(intervals) <= 2 * discard_cycles:
        raise ValueError("Insufficient complete 94-MegaMoE intervals after discarding both ends")
    prefix = marker_counts([r for r in kernels if r["start"] < moes[0]["start"]])
    checks = []
    starts = [r["start"] for r in kernels]
    for index, (lo, hi) in enumerate(intervals):
        rows = kernels[bisect.bisect_left(starts, lo):bisect.bisect_left(starts, hi)]
        counts = marker_counts(rows)
        checks.append(dict(index=index, start_ns=lo, end_ns=hi, duration_ms=(hi-lo)/1e6,
                           marker_counts_by_start=counts,
                           target_counts_ok=(counts["target_megamoe"] == LAYERS
                                             and counts["dense_fp8_qkv_o_family"] == 2 * LAYERS
                                             and counts["fa3_forward_family"] >= LAYERS)))
    selected_checks = checks[discard_cycles:len(checks)-discard_cycles if discard_cycles else None]
    windows = [(r["start_ns"], r["end_ns"]) for r in selected_checks]
    outer_lo, outer_hi = windows[0][0], windows[-1][1]

    def assign(rows):
        included, boundary, overlap = [], [], []
        for row in rows:
            if row["start"] < outer_hi and row["end"] > outer_lo:
                overlap.append(row)
                (included if outer_lo <= row["start"] and row["end"] <= outer_hi else boundary).append(row)
        return included, boundary, overlap

    selected, boundary, overlapping = assign(kernels)
    ncycles = len(windows)
    wall_ns = sum(b-a for a, b in windows)
    duration_sum_ns = sum(r["end"] - r["start"] for r in selected)
    busy_ns = union_ns((r["start"], r["end"]) for r in selected)
    # Internal anchors do not cut PDL work. Only the two outer boundaries clip.
    clipped = [(max(outer_lo, r["start"]), min(outer_hi, r["end"])) for r in overlapping]
    overlap_sum_ns = sum(b-a for a, b in clipped)
    clipped_union_ns = union_ns(clipped)
    transfers = {}
    for table, key in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        rows, excluded, copy_overlap = assign(records(db, table, device))
        transfers[key] = {"rows": summarize(rows, ncycles, ("copyKind", "srcKind", "dstKind") if key == "memcpy" else ("memKind", "value"), copy_overlap, (outer_lo, outer_hi)),
                          "calls": len(rows), "boundary_crossing_excluded_calls": len(excluded),
                          "busy_union_ms": union_ns((max(outer_lo, r["start"]), min(outer_hi, r["end"])) for r in copy_overlap) / 1e6}
    clock, clock_evidence = clock_summary(db, device, windows)
    return {
        "schema_version": "kernel-profile/2", "run": run, "role": role, "device": device,
        "method": {"target_layers_assumed": LAYERS,
                   "cycle_anchor": "Every 94th target sm90_fp8_mega_moe_impl start, in start order",
                   "discard_intervals_each_end": discard_cycles,
                   "budget_membership": "Main budget is clipped overlap with the single outer selected window. Calls/mean/min/max use full containment in that outer window; internal anchors never exclude work.",
                   "scope": "All recorded kernels including draft; no shape inference or removable-latency claim",
                   "alignment_limit": "Marker counts support but cannot prove absolute layer phase; 94 layers and first captured MoE being layer 0 remain baseline assumptions"},
        "alignment": {"prefix_before_first_moe": prefix,
                      "prefix_first_layer_counts_ok": prefix["dense_fp8_qkv_o_family"] == 2 and prefix["fa3_forward_family"] >= 1,
                      "all_interval_target_counts_ok": all(r["target_counts_ok"] for r in checks),
                      "selected_interval_target_counts_ok": all(r["target_counts_ok"] for r in selected_checks),
                      "capture_moe_calls": len(moes), "complete_intervals": len(intervals),
                      "trailing_moe_calls_from_last_anchor": len(moes) - (len(anchors)-1)*LAYERS,
                      "intervals": checks},
        "selected_cycles": ncycles, "windows_ns": windows,
        "outer_window_ns": [outer_lo, outer_hi],
        "window_total_ms": wall_ns / 1e6, "cycle_mean_ms": wall_ns / ncycles / 1e6,
        "kernel_calls": len(selected), "kernel_duration_sum_ms": duration_sum_ns / 1e6,
        "kernel_window_overlap_sum_ms": overlap_sum_ns / 1e6,
        "kernel_full_containment_union_busy_ms": busy_ns / 1e6,
        "kernel_full_containment_sum_minus_union_ms": (duration_sum_ns - busy_ns) / 1e6,
        "kernel_union_busy_ms": clipped_union_ns / 1e6,
        "kernel_sum_minus_union_ms": (overlap_sum_ns - clipped_union_ns) / 1e6,
        "kernel_union_clipped_including_boundary_ms": clipped_union_ns / 1e6,
        "kernel_idle_from_clipped_union_ms": (wall_ns - clipped_union_ns) / 1e6,
        "kernel_boundary_crossing_excluded_calls": len(boundary),
        "kernels": summarize(selected, ncycles, ("name",) + GEOMETRY, overlapping, (outer_lo, outer_hi)),
        "copies": transfers,
        "gpu_info": [r for r in records(db, "TARGET_INFO_GPU") if r.get("id") == device],
        "gpc_clock": clock, "gpc_clock_evidence": clock_evidence,
        "raw_clock_series_summary": clock_evidence["raw_clock_series_summary"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--role", required=True, choices=("prefill", "decode"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--discard-cycles", type=int, default=2)
    args = parser.parse_args()
    path = args.sqlite.resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        output = extract(db, args.role, args.run, args.device, args.discard_cycles)
    output["source"] = {"sqlite_path": str(path), "sqlite_sha256": sha256(path),
                        "sqlite_size_bytes": path.stat().st_size,
                        "extractor_sha256": sha256(Path(__file__).resolve())}
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
