"""Summarize recorded target forwards without accepting capture completeness.

Match CPU NVTX launch context by thread/timestamp and GPU kernels by PID and
correlation ID. Select only ranges containing 94 main and 94 pre-dispatch
kernels. This is conditional recorded activity, not a pipeline critical path,
whole-capture acceptance, normal throughput or an achievable speedup estimate.
Use local exports from the pinned Nsight collector. Source databases are read
only; temporary indexes are in memory. No environment metadata is selected.
"""

import argparse
import collections
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path


def intervals(items):
    if not items:
        return {"sum_ns": 0, "union_ns": 0, "span_ns": 0}
    pairs = sorted(items)
    right = pairs[0][0]
    union = 0
    for a, b in pairs:
        assert b > a
        union += max(0, b - max(a, right))
        right = max(right, b)
    return {
        "sum_ns": sum(b - a for a, b in pairs),
        "union_ns": union,
        "span_ns": right - pairs[0][0],
    }


def stats(values):
    return {
        "p50": statistics.median(values),
        "p95": statistics.quantiles(values, n=100, method="inclusive")[94],
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def audit(database):
    database = database.resolve()
    with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as db:
        db.execute("PRAGMA temp_store=MEMORY")
        db.execute(
            "CREATE TEMP TABLE launches AS SELECT CAST((r.globalTid & -16777216) AS INTEGER) pid,r.globalTid tid,r.correlationId cid,r.start,r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id WHERE s.value LIKE '%Launch%'"
        )
        db.execute("CREATE INDEX launches_time ON launches(tid,start)")
        db.execute(
            "CREATE TEMP TABLE ranges AS SELECT n.eventId rid,n.globalTid tid,n.start,n.end,COALESCE(n.text,s.value) label FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE COALESCE(n.text,s.value) LIKE 'execute_context%'"
        )
        db.execute(
            "CREATE TEMP TABLE target_api AS SELECT t.rid,l.pid,l.cid FROM ranges t JOIN launches l ON l.tid=t.tid AND l.start>=t.start AND l.start<t.end"
        )
        db.execute("CREATE UNIQUE INDEX target_api_key ON target_api(pid,cid)")
        ranges = {
            rid: {"cpu_start": a, "cpu_end": b, "pid": tid & -16777216, "label": label}
            for rid, tid, a, b, label in db.execute(
                "SELECT rid,tid,start,end,label FROM ranges"
            )
        }
        names = dict(
            db.execute(
                "SELECT s.id,s.value FROM StringIds s JOIN (SELECT DISTINCT demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL) k ON k.demangledName=s.id"
            )
        )
        main = {n for n, name in names.items() if "sm90_fp8_mega_moe_impl" in name}
        pre = {
            n
            for n, name in names.items()
            if "sm90_mega_moe_pre_dispatch_kernel" in name
        }
        per_name = collections.defaultdict(lambda: {"count": 0, "sum_ns": 0})
        records = []
        rejected = []
        group = []
        previous = None

        def consume(rid, events):
            if rid is None:
                return
            base = ranges[rid]
            mm = [(a, b) for a, b, n in events if n in main]
            pp = [(a, b) for a, b, n in events if n in pre]
            if len(mm) != 94 or len(pp) != 94:
                rejected.append(
                    {
                        "range": rid,
                        "label": base["label"],
                        "main": len(mm),
                        "pre_dispatch": len(pp),
                    }
                )
                return
            all_times = intervals([(a, b) for a, b, n in events])
            m = intervals(mm)
            p = intervals(pp)
            records.append(
                {
                    "range": rid,
                    **base,
                    "kernel_count": len(events),
                    "gpu_span_ms": all_times["span_ns"] / 1e6,
                    "gpu_union_ms": all_times["union_ns"] / 1e6,
                    "main_sum_ms": m["sum_ns"] / 1e6,
                    "main_union_ms": m["union_ns"] / 1e6,
                    "pre_dispatch_sum_ms": p["sum_ns"] / 1e6,
                    "main_fraction_of_recorded_kernel_union": m["union_ns"]
                    / all_times["union_ns"],
                    "kernel_busy_fraction_within_target_span": all_times["union_ns"]
                    / all_times["span_ns"],
                    "cpu_range_ms": (base["cpu_end"] - base["cpu_start"]) / 1e6,
                    "main_overlap_excess_ms": (m["sum_ns"] - m["union_ns"]) / 1e6,
                }
            )
            for a, b, n in events:
                per_name[n]["count"] += 1
                per_name[n]["sum_ns"] += b - a

        for rid, a, b, n in db.execute(
            "SELECT t.rid,k.start,k.end,k.demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN target_api t ON t.pid=k.globalPid AND t.cid=k.correlationId ORDER BY t.rid,k.start"
        ):
            if previous is not None and rid != previous:
                consume(previous, group)
                group = []
            group.append((a, b, n))
            previous = rid
        consume(previous, group)
        all_empty = [
            {"range": rid, "label": r["label"]}
            for rid, r in ranges.items()
            if rid not in {r["range"] for r in records + rejected}
        ]
        metrics = [
            "gpu_span_ms",
            "gpu_union_ms",
            "main_sum_ms",
            "main_union_ms",
            "pre_dispatch_sum_ms",
            "main_fraction_of_recorded_kernel_union",
            "kernel_busy_fraction_within_target_span",
            "cpu_range_ms",
            "main_overlap_excess_ms",
        ]
        with database.open("rb") as stream:
            input_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "sqlite_sha256": input_sha,
            "selected_range_count": len(records),
            "no_recorded_kernel_ranges": all_empty,
            "rejected_nonempty_ranges": rejected,
            "summary": {k: stats([r[k] for r in records]) for k in metrics},
            "normalization_and_group_quant_duration_sum_ms_per_selected_range": sum(
                v["sum_ns"]
                for n, v in per_name.items()
                if "rms_norm" in names[n]
                or "per_token_group_quant_8bit_kernel" in names[n]
            )
            / 1e6
            / len(records),
            "normalization_and_group_quant_scope": "Broad kernel-name category, including ops not necessarily fusible; duration sums are not critical-path savings.",
            "top_kernel_names_by_duration_sum": [
                {"name": names[n], **v}
                for n, v in sorted(
                    per_name.items(), key=lambda t: t[1]["sum_ns"], reverse=True
                )[:15]
            ],
            "records": records,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    out = {}
    for worker in ("prefill-0", "prefill-1", "prefill-2", "decode-0"):
        out[worker] = audit(root / f"nsys-{worker}.sqlite")
        print(worker, out[worker]["selected_range_count"], flush=True)
    (root / "recorded-target-timings.json").write_text(
        json.dumps(
            {
                "scope": "Only recorded NVTX ranges with all 94 main and 94 pre-dispatch kernels; not complete-capture acceptance or a global critical path. Ratios are conditional on recorded kernel activity, not pipeline throughput.",
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "workers": out,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
