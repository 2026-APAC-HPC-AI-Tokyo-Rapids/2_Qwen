#!/usr/bin/env python3
"""Render and optionally submit exactly one experiment."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--submit", action="store_true", help="call pjsub; otherwise only render")
    args = ap.parse_args()
    proc = subprocess.run([sys.executable, str(root / "scripts/render-job.py"), str(args.experiment)], check=True, text=True, capture_output=True)
    job = Path(proc.stdout.strip())
    print(f"rendered={job}")
    if not args.submit:
        print("dry-run: pass --submit to call pjsub")
        return
    answer = input(f"Submit {job.name} with pjsub? Type the experiment name: ")
    expected = args.experiment.stem
    if answer != expected:
        raise SystemExit("submission cancelled")
    subprocess.run(["pjsub", str(job)], check=True)

if __name__ == "__main__": main()

