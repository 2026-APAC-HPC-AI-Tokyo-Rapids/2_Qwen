#!/usr/bin/env python3
"""Require nonempty CUDA captures from every TP rank; keep only native reports."""
from __future__ import annotations

import argparse
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
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='DIAGNOSTIC_EVENT'").fetchone():
            # Explicit loss is also emitted at informational severity. A generic
            # forced-flush warning alone does not prove loss, but is unresolved
            # evidence and must still prevent accepting a complete capture.
            warnings = [row[0] for row in connection.execute("SELECT DISTINCT text FROM DIAGNOSTIC_EVENT")]
            incomplete = []
            for text in warnings:
                dropped = re.search(r"Number of incomplete CUPTI events dropped\s*:\s*([\d,]+)", text)
                if "Not all CUDA events" in text or ("CUDA driver version" in text and "not supported" in text) or (dropped and int(dropped[1].replace(",", "")) > 0):
                    incomplete.append(text)
            if incomplete:
                raise ValueError(f"incomplete CUDA capture in {database.name}: {incomplete}")
    if len(rows) != tp or any(count <= 0 for _, count in rows):
        raise ValueError(f"expected kernels on {tp} GPUs, got {rows}")
    return {"gpu_count": len(rows), "kernel_count": sum(count for _, count in rows)}


def worker_gpu_count(runtime: Path, name: str, tp: int) -> int:
    info = runtime / f"results/server-info-{name}.json"
    if not info.is_file():
        return tp
    config = json.loads(info.read_text())
    if "vllm_config" not in config:
        return tp
    parallel = config["vllm_config"]["parallel_config"]
    if parallel["tensor_parallel_size"] != tp:
        raise ValueError(f"endpoint TP differs from exported configuration: {name}")
    return tp * parallel["data_parallel_size"] * parallel.get("pipeline_parallel_size", 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--nsys", type=Path, default=Path("/opt/nsys/bin/nsys"))
    args = parser.parse_args()
    results = {}
    endpoints = args.runtime / "source/worker-endpoints.tsv"
    selected = args.runtime / "source/profile-worker-endpoints.tsv"
    if selected.is_file():
        rows = selected.read_text().splitlines()
        if len(rows) != 1 or rows[0].split("\t")[0] != "decode-0" or worker_gpu_count(args.runtime, "decode-0", int(rows[0].split("\t")[-1])) != 4:
            raise ValueError("decoder-only capture requires exactly decode-0 with four GPUs")
        if rows[0] not in endpoints.read_text().splitlines():
            raise ValueError("profile worker is not a serving endpoint")
        endpoints = selected
    for row in endpoints.read_text().splitlines():
        name, _host, _port, tp = row.split("\t")
        report = args.runtime / f"profile/nsys-{name}.nsys-rep"
        if not report.is_file() or report.stat().st_size == 0:
            raise ValueError(f"missing Nsight report: {report}")
        database = args.runtime / f"profile/nsys-{name}.sqlite"
        try:
            subprocess.run([str(args.nsys), "export", "--type=sqlite", f"--output={database}", str(report)], check=True)
            results[name] = {"report": report.name, **validate_capture(database, worker_gpu_count(args.runtime, name, int(tp)))}
        finally:
            database.unlink(missing_ok=True)
    expected_gpus = sum(worker_gpu_count(args.runtime, row.split("\t")[0], int(row.split("\t")[3])) for row in endpoints.read_text().splitlines())
    required_gpus = 4 if selected.is_file() else 16
    if expected_gpus != required_gpus or sum(value["gpu_count"] for value in results.values()) != required_gpus:
        raise ValueError("incomplete selected worker GPU coverage")
    (args.runtime / "profile/validation.json").write_text(json.dumps({"valid": True, "workers": results, "scope": "decode-0" if selected.is_file() else "all-workers", "cuda_event_completion_trace": False, "limitations": "Kernel timing diagnostic only; no clean throughput or complete critical-path claim."}, indent=2) + "\n")


if __name__ == "__main__":
    main()
