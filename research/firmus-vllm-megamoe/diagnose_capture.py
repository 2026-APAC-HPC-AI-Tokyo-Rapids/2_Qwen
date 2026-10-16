"""Audit recorded CUDA/NVTX structure without accepting an incomplete capture.

Use local SQLite exports from the pinned Nsight collector. No timing or speedup
is inferred, and the strict capture validator is neither changed nor bypassed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sqlite3
from pathlib import Path

WORKERS = ("prefill-0", "prefill-1", "prefill-2", "decode-0")
PRIMARY = {f"CUPTI_ACTIVITY_KIND_{kind}" for kind in ("RUNTIME", "DRIVER", "KERNEL", "MEMCPY", "MEMSET")}
META = ("CUDA_GRAPH_TRACE_OPTIONS:MODE", "CUDA_FLUSH_PERIODICALLY", "CUDA_FLUSH_PERIOD", "CUDA_FLUSH_ON_CUDA_PROFILER_STOP", "RUN_DURATION_MS")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def one_counter(texts: list[str], pattern: str) -> str:
    matches = [match[1] for text in texts if (match := re.search(pattern, text))]
    if len(matches) != 1:
        raise ValueError(f"expected one diagnostic matching {pattern}: {matches}")
    return matches[0]


def diagnose(database: Path) -> dict:
    with sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True) as db:
        db.execute("PRAGMA temp_store=MEMORY")
        identities = dict(db.execute("SELECT cuDevice,uuid FROM TARGET_INFO_GPU"))
        devices = [{"device": device, "uuid": identities[device], "pid": pid, "kernels": count} for device, pid, count in db.execute("SELECT deviceId,globalPid,COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY 1,2")]
        diagnostics = [dict(zip(("pid", "source", "severity", "text"), row)) for row in db.execute("SELECT globalPid,source,severity,text FROM DIAGNOSTIC_EVENT")]
        counts = {}
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'CUPTI_ACTIVITY_KIND_%'")]
        for table in tables:
            columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
            pid = "globalPid" if "globalPid" in columns else "(globalTid & -16777216)" if "globalTid" in columns else None
            if pid:
                counts[table] = dict(db.execute(f"SELECT {pid},COUNT(*) FROM {table} GROUP BY 1"))
        processes = {}
        for pid in sorted({device["pid"] for device in devices}):
            texts = [row["text"] for row in diagnostics if row["pid"] == pid]
            produced = int(one_counter(texts, r"Number of CUPTI events produced:\s*([\d,]+)").replace(",", ""))
            collected = int(one_counter(texts, r"Number of CUDA events collected:\s*([\d,]+)").replace(",", ""))
            per_table = {table: values.get(pid, 0) for table, values in counts.items()}
            primary = sum(value for table, value in per_table.items() if table in PRIMARY)
            auxiliary = sum(value for table, value in per_table.items() if table not in PRIMARY)
            processes[str(pid)] = {
                "produced_diagnostic": produced,
                "collected_diagnostic": collected,
                "buffer_diagnostic": one_counter(texts, r"CUPTI buffers:\s*(.+)\."),
                "table_counts": per_table,
                "collected_matches_primary_table_sum": collected == primary,
                "produced_minus_collected_minus_auxiliary": produced - collected - auxiliary,
                "warnings": [row["text"] for row in diagnostics if row["pid"] == pid and row["severity"] >= 2],
            }

        # Explicit INTEGER affinity is required. Without CAST, a correlated
        # lookup against an INTEGER pid scans the entire untyped launch index.
        db.execute("CREATE TEMP TABLE launches AS SELECT CAST((r.globalTid & -16777216) AS INTEGER) pid,r.globalTid tid,r.correlationId cid,r.start,r.end,s.value name FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id WHERE s.value LIKE '%Launch%'")
        db.execute("CREATE INDEX launches_key ON launches(pid,cid)")
        db.execute("CREATE INDEX launches_time ON launches(tid,start)")
        db.execute("CREATE TEMP TABLE kernel_counts AS SELECT globalPid pid,correlationId cid,count(*) n FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY 1,2")
        db.execute("CREATE INDEX kernel_counts_key ON kernel_counts(pid,cid)")
        missing_query = "SELECT COALESCE(SUM(k.n),0) FROM kernel_counts k WHERE NOT EXISTS(SELECT 1 FROM launches l WHERE l.pid=k.pid AND l.cid=k.cid)"
        plan = [row[3] for row in db.execute("EXPLAIN QUERY PLAN " + missing_query)]
        if not any("SEARCH l USING COVERING INDEX launches_key (pid=? AND cid=?)" in item for item in plan):
            raise ValueError(f"launch correlation requires indexed PID/correlation lookup: {plan}")
        missing = db.execute(missing_query).fetchone()[0]
        without_kernels = db.execute("SELECT name,count(*) FROM launches l WHERE NOT EXISTS(SELECT 1 FROM kernel_counts k WHERE l.pid=k.pid AND l.cid=k.cid) GROUP BY name").fetchall()
        db.execute("CREATE TEMP TABLE main_counts AS SELECT k.globalPid pid,k.correlationId cid,count(*) n FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE s.value LIKE '%sm90_fp8_mega_moe_impl%' GROUP BY 1,2")
        db.execute("CREATE INDEX main_counts_key ON main_counts(pid,cid)")
        db.execute("CREATE TEMP TABLE ranges AS SELECT n.eventId,n.globalTid tid,n.start,n.end,COALESCE(n.text,s.value) label FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE COALESCE(n.text,s.value) LIKE 'execute_context%'")
        ranges = db.execute("SELECT t.eventId,t.label,t.start,t.end,(t.tid & -16777216),COALESCE(SUM(m.n),0) FROM ranges t LEFT JOIN launches l ON l.tid=t.tid AND l.start>=t.start AND l.start<t.end LEFT JOIN main_counts m ON l.pid=m.pid AND l.cid=m.cid GROUP BY t.eventId").fetchall()
        by_pid = {}
        for pid in sorted({row[4] for row in ranges}):
            selected = [row for row in ranges if row[4] == pid]
            by_pid[str(pid)] = {
                "range_count": len(selected),
                "main_kernel_count_distribution": dict(collections.Counter(row[5] for row in selected)),
                "nonzero_non94_ranges": [{"event": row[0], "label": row[1], "start": row[2], "end": row[3], "main_count": row[5]} for row in selected if row[5] not in (0, 94)],
            }
        # Never select unrestricted metadata: capture metadata also contains
        # inherited environment variables, including ephemeral credentials.
        placeholders = ",".join("?" for _ in META)
        metadata = dict(db.execute(f"SELECT name,value FROM META_DATA_CAPTURE WHERE name IN ({placeholders})", META))
        profiler_calls = [{"pid": tid & -16777216, "name": name, "start_ns": start, "end_ns": end} for tid, name, start, end in db.execute("SELECT r.globalTid,s.value,r.start,r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id WHERE s.value LIKE '%ProfilerStart%' OR s.value LIKE '%ProfilerStop%'")]
        stops = [call["start_ns"] for call in profiler_calls if "ProfilerStop" in call["name"]]
        last_kernels = dict(db.execute("SELECT globalPid,MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY globalPid"))
        return {
            "sqlite_sha256": digest(database),
            "devices": devices,
            "active_processes": processes,
            "metadata_allowlist": metadata,
            "recorded_stop_boundary": {
                "scope": "Recorded activity ordering only; cannot exclude missing GPU activity or certify complete collection.",
                "profiler_API_calls": profiler_calls,
                "last_recorded_kernel_end_ns_by_pid": last_kernels,
                "first_recorded_stop_start_ns": min(stops) if stops else None,
                "gap_after_last_recorded_kernel_ns": min(stops) - max(last_kernels.values()) if stops else None,
            },
            "loss_or_compatibility_diagnostics": [row for row in diagnostics if "dropped" in row["text"].lower() or "Some CUPTI events might be missing" in row["text"] or ("CUDA driver version" in row["text"] and "not supported" in row["text"])],
            "kernels_without_recorded_launch_API": missing,
            "launch_APIs_without_recorded_kernels": without_kernels,
            "target_NVTX_launch_attribution": by_pid,
            "main_kernel_count": db.execute("SELECT COALESCE(SUM(n),0) FROM main_counts").fetchone()[0],
            "main_kernels_attributed_to_target_NVTX": sum(row[5] for row in ranges),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    workers = {worker: diagnose(args.directory / f"nsys-{worker}.sqlite") for worker in WORKERS}
    uuids = [device["uuid"] for result in workers.values() for device in result["devices"]]
    if len(uuids) != 16 or len(set(uuids)) != 16:
        raise ValueError("expected 16 distinct recorded GPUs")
    result = {
        "purpose": "Recorded-data structural diagnosis only; does not accept a capture or compute performance.",
        "analysis_script_sha256": digest(Path(__file__)),
        "workers": workers,
        "limitations": [
            "Unresolved completeness warnings still prevent strict complete-capture acceptance.",
            "Matching recorded APIs/kernels and 94-layer ranges cannot exclude jointly missing records.",
            "Produced/collected counters have different category scopes; their difference is not a dropped-event count.",
            "Attribution uses CPU launch API timestamps/threads, not containment of GPU execution in CPU ranges.",
            "NVTX-external kernels are not exclusively draft work; no critical path or speedup is inferred.",
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
