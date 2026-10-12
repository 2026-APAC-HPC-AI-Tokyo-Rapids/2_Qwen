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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    args = parser.parse_args()
    results = {}
    for row in (args.runtime / "source/worker-endpoints.tsv").read_text().splitlines():
        name, _host, _port, tp = row.split("\t")
        report = args.runtime / f"profile/nsys-{name}.nsys-rep"
        if not report.is_file() or report.stat().st_size == 0:
            raise ValueError(f"missing Nsight report: {report}")
        database = args.runtime / f"profile/nsys-{name}.sqlite"
        subprocess.run(["/opt/nsys/bin/nsys", "export", "--type=sqlite", f"--output={database}", str(report)], check=True)
        results[name] = {"report": report.name, **validate_capture(database, int(tp))}
        database.unlink()
    expected_gpus = sum(int(row.split("\t")[3]) for row in (args.runtime / "source/worker-endpoints.tsv").read_text().splitlines())
    if expected_gpus != 16 or sum(value["gpu_count"] for value in results.values()) != 16:
        raise ValueError("expected complete 16-GPU worker coverage")
    (args.runtime / "profile/validation.json").write_text(json.dumps({"valid": True, "workers": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
