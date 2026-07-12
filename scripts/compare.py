#!/usr/bin/env python3
"""Compare two summary CSV rows and report percentage changes."""
from __future__ import annotations
import argparse, csv
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.summary.open()))
    find = lambda name: next((r for r in rows if name in r["run"]), None)
    base, cand = find(args.baseline), find(args.candidate)
    if not base or not cand: raise SystemExit("baseline or candidate not found")
    print("metric,baseline,candidate,change_percent")
    for key in ("request_throughput", "input_throughput", "output_throughput", "total_throughput"):
        if not base[key] or not cand[key]: continue
        b, c = float(base[key]), float(cand[key])
        print(f"{key},{b:.6g},{c:.6g},{(c / b - 1) * 100:.3f}")

if __name__ == "__main__": main()
