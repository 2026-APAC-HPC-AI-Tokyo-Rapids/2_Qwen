#!/usr/bin/env python3
"""Write a small Actions summary from collected artifacts, including failed runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_artifact(root: Path, filename: str) -> dict:
    paths = sorted(root.rglob(filename)) if root.exists() else []
    return json.loads(paths[0].read_text()) if paths else {}


def summarize(root: Path, job_id: str, status: str) -> str:
    identity = read_artifact(root, "provenance.json")
    config = read_artifact(root, "firmus-framework.json")
    score = read_artifact(root, "score.json")
    capture = read_artifact(root, "validation.json")
    profile = identity.get("profile", False)
    reports = list(root.rglob("*.nsys-rep")) if root.exists() else []
    run_id = os.environ.get("GITHUB_RUN_ID", "unknown")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "unknown")
    lines = ["### Firmus Qwen", "", f"- Status: **{status}**; Slurm `{job_id}`; completion marker: **{'present' if list(root.rglob('run-success')) else 'missing'}**.", f"- Framework: **{identity.get('framework', 'unavailable')}**; H200 × 16; P4 × TP2 / D2 × TP4 EP4; concurrency **{config.get('concurrency', 'unavailable')}**.", "- Workload: reference-07, ISL 4000 / OSL 200, warmup / benchmark / grace **20 / 60 / 15 s**; **not an official submission**."]
    if score:
        lines += ["", "| Metric | Value |", "| --- | ---: |", f"| Requests / good requests | {score['request_count']} / {score['good_request_count']} |", f"| Good request fraction | {score['good_request_fraction']:.2%} |", f"| Goodput | {score['goodput_requests_per_second']:.3f} req/s |", f"| Reference score | {score['score_output_tokens_per_second']:.3f} output tok/s |", f"| SLA eligible | {score['eligible']} |"]
    else:
        lines += ["", "Validated AIPerf result unavailable; inspect the logs."]
    if profile:
        workers = capture.get("workers", {})
        gpu_coverage = f"{sum(worker['gpu_count'] for worker in workers.values())} / 16 GPUs" if capture else "GPU coverage unverified"
        notes = sorted(root.rglob("interpretation.txt"))
        capture_note = notes[0].read_text().strip() if notes else "Capture timing unavailable; inspect profile-capture.log."
        lines += ["", f"Nsight: **{len(reports)} / 6 reports**; CUDA validation **{capture.get('valid', False)}**; **{gpu_coverage}**. {capture_note} **Instrumented score; do not use for performance comparison.**"]
    else:
        lines += ["", "Nsight: disabled."]
    lines += ["", f"Archive after the archive job succeeds: `reschene-tuat-server:/mnt/net_share/qwen-artifacts/{run_id}/{attempt}/`.", "Reports: `profile/`; conditions and image hashes: `source/`; raw requests and score: `results/`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--status", required=True)
    args = parser.parse_args()
    print(summarize(args.artifacts, args.job_id, args.status))


if __name__ == "__main__":
    main()
