#!/usr/bin/env python3
"""Summarize SGLang JSONL results into stable CSV rows."""
from __future__ import annotations
import argparse, csv, json, statistics
from pathlib import Path

METRICS = ("request_throughput", "input_throughput", "output_throughput", "total_throughput")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    rows = []
    for run in args.runs:
        files = list(run.glob("**/results.jsonl")) if run.is_dir() else [run]
        records = [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
        if not records: continue
        row = {"run": str(run), "samples": len(records)}
        for metric in METRICS:
            values = [float(r[metric]) for r in records if metric in r]
            row[metric] = statistics.median(values) if values else ""
        rows.append(row)
    out = args.output.open("w", newline="") if args.output else None
    target = out or __import__("sys").stdout
    writer = csv.DictWriter(target, fieldnames=("run", "samples", *METRICS))
    writer.writeheader(); writer.writerows(rows)
    if out: out.close()

if __name__ == "__main__": main()
