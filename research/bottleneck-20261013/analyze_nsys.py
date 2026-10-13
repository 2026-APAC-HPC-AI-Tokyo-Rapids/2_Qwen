"""Audit recorded GPU work; do not equate duration sums with critical-path time.

Usage: python3 analyze_nsys.py exported.sqlite > analysis.json
The input is a CPU export of a separately collected diagnostic profile.
"""

import collections
import json
import sqlite3
import sys
from pathlib import Path


def union_ns(intervals):
    total = 0
    previous_end = None
    for start, end in sorted(intervals):
        if previous_end is None or start >= previous_end:
            total += end - start
        elif end > previous_end:
            total += end - previous_end
        previous_end = max(end, previous_end or end)
    return total


def analyze(path):
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = list(connection.execute(
        "SELECT k.start,k.end,k.deviceId,k.globalPid,k.graphNodeId,"
        "k.gridX,k.gridY,k.blockX,s.value AS name "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id"
    ))
    devices = sorted({r["deviceId"] for r in rows})
    # Intersect the four ranks' recorded windows; exclude asynchronous edges.
    start = max(min(r["start"] for r in rows if r["deviceId"] == d) for d in devices)
    end = min(max(r["end"] for r in rows if r["deviceId"] == d) for d in devices)
    if end <= start:
        raise ValueError("GPU recorded intervals have no common window")
    groups = collections.defaultdict(lambda: {"count": 0, "ns": 0})
    ranks = []
    for device in devices:
        selected = [r for r in rows if r["deviceId"] == device and r["end"] > start and r["start"] < end]
        intervals = [(max(start, r["start"]), min(end, r["end"])) for r in selected]
        ranks.append({"device": device, "kernel_count": len(selected),
                      "recorded_busy_union_pct": 100 * union_ns(intervals) / (end - start),
                      "duration_sum_ns": sum(b - a for a, b in intervals),
                      "graph_node_records_pct": 100 * sum(bool(r["graphNodeId"]) for r in selected) / len(selected)})
        for r, (a, b) in zip(selected, intervals, strict=True):
            item = groups[(r["name"], r["gridX"], r["gridY"], r["blockX"])]
            item["count"] += 1
            item["ns"] += b - a
    total = sum(r["duration_sum_ns"] for r in ranks)
    kernels = [{"name": k[0], "grid_x": k[1], "grid_y": k[2], "block_x": k[3],
                **v, "mean_us": v["ns"] / v["count"] / 1000,
                "kernel_duration_share_pct": 100 * v["ns"] / total}
               for k, v in sorted(groups.items(), key=lambda kv: kv[1]["ns"], reverse=True)]
    warnings = []
    if connection.execute("SELECT 1 FROM sqlite_master WHERE name='DIAGNOSTIC_EVENT'").fetchone():
        warnings = [dict(r) for r in connection.execute(
            "SELECT DISTINCT severity,text,globalPid FROM DIAGNOSTIC_EVENT WHERE severity > 1"
        )]
    summaries = collections.defaultdict(lambda: {"count": 0, "ns": 0})
    for kernel in kernels:
        item = summaries[kernel["name"]]
        item["count"] += kernel["count"]
        item["ns"] += kernel["ns"]
    names = [{"name": k, **v, "kernel_duration_share_pct": 100 * v["ns"] / total}
             for k, v in sorted(summaries.items(), key=lambda kv: kv[1]["ns"], reverse=True)]
    return {"common_window_ns": [start, end], "ranks": ranks,
            "kernel_names": names, "kernel_shape_count": len(kernels),
            "top_kernel_shapes": kernels[:20], "warnings": warnings,
            "interpretation": "Recorded kernel durations only. Concurrent kernels can overlap; these shares are not wall-time savings. Collection warnings invalidate attributing unrecorded gaps to CPU stalls. No hardware-counter proof of HBM or compute saturation."}


if __name__ == "__main__":
    print(json.dumps(analyze(Path(sys.argv[1])), indent=2))
