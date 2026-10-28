#!/usr/bin/env python3
"""Require nonempty CUDA captures from every TP rank; keep only native reports."""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sqlite3
import subprocess
from pathlib import Path


def validate_capture(database: Path, tp: int) -> dict:
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_KERNEL'").fetchone():
            raise ValueError(f"no CUDA kernels captured in {database.name}; expected kernels on {tp} GPUs")
        rows = connection.execute("SELECT deviceId, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId").fetchall()
    if len(rows) != tp or any(count <= 0 for _, count in rows):
        raise ValueError(f"expected kernels on {tp} GPUs, got {rows}")
    return {"gpu_count": len(rows), "kernel_count": sum(count for _, count in rows)}


def validate_cpu_traces(directory: Path, tp: int, pp: int = 1) -> dict:
    stages = set()
    by_rank = {}
    for path in directory.glob("*.trace.json.gz"):
        match = re.search(r"-TP-(\d+)((?:-(?:DP|PP|EP)-\d+)*)\.trace\.json\.gz$", path.name)
        if match:
            pipeline = re.search(r"-PP-(\d+)", match[2])
            key = int(match[1]), int(pipeline[1]) if pipeline else 0
            by_rank.setdefault(key, []).append(path)
    for rank, pipeline_rank in ((t, p) for p in range(pp) for t in range(tp)):
        traces = by_rank.get((rank, pipeline_rank), [])
        if len(traces) != 1:
            raise ValueError(f"expected one CPU trace for {directory.name} TP{rank}/PP{pipeline_rank}, got {len(traces)}")
        with gzip.open(traces[0], "rt") as handle:
            events = json.load(handle)["traceEvents"]
        annotations = {event.get("name", "") for event in events if event.get("cat") == "user_annotation"}
        if not annotations:
            raise ValueError(f"no CPU stage spans in {traces[0].name}")
        stages.update(name for name in annotations if name.startswith("step[") or "draft" in name.lower() or "verify" in name.lower() or "recv" in name or "send" in name or name == "run_batch")
    return {"cpu_trace_count": tp * pp, "cpu_stage_names": sorted(stages)}


def summarize_kernels(database: Path) -> dict:
    """Retain the same compact kernel evidence as the recovered Firmus profiles."""
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        devices = connection.execute("SELECT deviceId, COUNT(*), MIN(start), MAX(end), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId").fetchall()
        kernels = connection.execute("SELECT s.value, COUNT(*), SUM(k.end-k.start), AVG(k.end-k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT 40").fetchall()
    return {"time_unit": "ns", "interpretation": "Summed kernel durations overlap across streams and GPUs; envelope includes idle gaps and profiler export tails. Use the recorded capture_active/stop_requested window and CPU stage correlations for latency attribution.", "devices": [{"device_id": device, "kernel_count": count, "first_kernel_ns": start, "last_kernel_ns": end, "kernel_envelope_ns": end - start, "summed_kernel_duration_ns": duration} for device, count, start, end, duration in devices], "top_kernels": [{"name": name, "calls": count, "summed_duration_ns": duration, "mean_duration_ns": average} for name, count, duration, average in kernels]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    args = parser.parse_args()
    config = json.loads((args.runtime / "source/firmus-framework.json").read_text())
    separate = config["framework"] == "sglang" and config.get("workload") == "official"
    results = {}
    summaries = {}
    endpoints = (args.runtime / "source/worker-endpoints.tsv").read_text().splitlines()
    for row in endpoints:
        name, _host, _port, tp = row.split("\t")
        report = args.runtime / f"profile/nsys-{name}.nsys-rep"
        if not report.is_file() or report.stat().st_size == 0:
            raise ValueError(f"missing Nsight report: {report}")
        database = args.runtime / f"profile/nsys-{name}.sqlite"
        try:
            subprocess.run(["/opt/nsys/bin/nsys", "export", "--type=sqlite", f"--output={database}", str(report)], check=True)
            results[name] = {"report": report.name, **validate_capture(database, int(tp))}
            summaries[name] = summarize_kernels(database)
        finally:
            # Keep native reports only, including after a failed export or check.
            database.unlink(missing_ok=True)
        if separate:
            pp = config.get("prefill", {}).get("pp", 1) if name.startswith("prefill-") else 1
            results[name].update(validate_cpu_traces(args.runtime / "profile/torch" / name, int(tp) // pp, pp))
    if not endpoints or len(results) != len(endpoints):
        raise ValueError("expected a capture for every distinct worker endpoint")
    (args.runtime / "profile/validation.json").write_text(json.dumps({"valid": True, "workers": results}, indent=2) + "\n")
    (args.runtime / "profile/kernel-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
