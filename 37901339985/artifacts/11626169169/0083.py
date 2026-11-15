#!/usr/bin/env python3
"""Require nonempty CUDA captures from every TP rank; keep only native reports."""
from __future__ import annotations

import argparse
import json
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


def summarize_kernels(database: Path) -> dict:
    """Kernel time sums can overlap across streams; they are not GPU busy time."""
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        devices = connection.execute("SELECT deviceId, COUNT(*), MIN(start), MAX(end), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId").fetchall()
        kernels = connection.execute("SELECT s.value, COUNT(*), SUM(k.end-k.start), AVG(k.end-k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT 40").fetchall()
    return {"time_unit": "ns", "interpretation": "Summed kernel durations overlap across streams and GPUs; envelope includes idle gaps. Compare per-device and per-call costs, not raw totals across different traffic volumes.", "devices": [{"device_id": device, "kernel_count": count, "first_kernel_ns": start, "last_kernel_ns": end, "kernel_envelope_ns": end - start, "summed_kernel_duration_ns": duration} for device, count, start, end, duration in devices], "top_kernels": [{"name": name, "calls": count, "summed_duration_ns": duration, "mean_duration_ns": average} for name, count, duration, average in kernels]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    args = parser.parse_args()
    results = {}
    summaries = {}
    for row in (args.runtime / "source/worker-endpoints.tsv").read_text().splitlines():
        name, _host, _port, tp = row.split("\t")
        report = args.runtime / f"profile/nsys-{name}.nsys-rep"
        if not report.is_file() or report.stat().st_size == 0:
            raise ValueError(f"missing Nsight report: {report}")
        database = args.runtime / f"profile/nsys-{name}.sqlite"
        subprocess.run(["/opt/nsys/bin/nsys", "export", "--type=sqlite", f"--output={database}", str(report)], check=True)
        results[name] = {"report": report.name, **validate_capture(database, int(tp))}
        summaries[name] = summarize_kernels(database)
        database.unlink()
    if len(results) != 6:
        raise ValueError("expected six worker captures")
    (args.runtime / "profile/validation.json").write_text(json.dumps({"valid": True, "workers": results}, indent=2) + "\n")
    (args.runtime / "profile/kernel-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
