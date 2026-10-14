#!/usr/bin/env python3
"""Write a small Actions summary from collected artifacts, including failed runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_artifact(root: Path, filename: str) -> dict:
    paths = sorted(root.rglob(filename), key=lambda path: (len(path.parts), str(path))) if root.exists() else []
    return json.loads(paths[0].read_text()) if paths else {}


def summarize(root: Path, job_id: str, status: str) -> str:
    identity = read_artifact(root, "provenance.json")
    config = read_artifact(root, "firmus-framework.json")
    score = read_artifact(root, "score.json")
    export = read_artifact(root, "profile_export_aiperf.json")
    capture = read_artifact(root, "validation.json")
    profile = identity.get("profile", False)
    profile_only = identity.get("measurement_mode") == "profile-only"
    reports = list(root.rglob("*.nsys-rep")) if root.exists() else []
    run_id = os.environ.get("GITHUB_RUN_ID", "unknown")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "unknown")
    workload = identity.get("workload", "reference-07")
    phases = "60 / 120 / 30" if workload == "official" else "20 / 60 / 15"
    if profile_only:
        phases = "20 / 30 / 120 (diagnostic only)"
    topology = "P2 × TP4 / D2 × TP4 EP1" if identity.get("framework") == "sglang" and workload == "official" else "P4 × TP2 / D2 × TP4 EP4"
    if workload == "official" and config.get("concurrency") == 64 and config.get("speculation", {}).get("algorithm") == "EAGLE3":
        topology = "P4 × TP2 / D2 × TP4 EP1"
    if workload == "official" and identity.get("framework") == "vllm" and config.get("concurrency") == 104:
        topology = "P6 × TP2 / D1 × TP4 EP1"
    expected_reports = 7 if topology == "P6 × TP2 / D1 × TP4 EP1" else 6
    lines = ["### Firmus Qwen", "", f"- Status: **{status}**; Slurm `{job_id}`; completion marker: **{'present' if list(root.rglob('run-success')) else 'missing'}**.", f"- Framework: **{identity.get('framework', 'unavailable')}**; H200 × 16; {topology}; concurrency **{config.get('concurrency', 'unavailable')}**.", f"- Workload: {workload}, ISL 4000 / OSL 200, warmup / benchmark / grace **{phases} s**; **not an official submission**."]
    if identity.get("dynamo"):
        dynamo = identity["dynamo"]
        lines += [f"- Dynamo **{dynamo['version']}**; engine **{dynamo['engine_version']}**; frontend → TCP → PD workers; NIXL KV transfer. Engine version differs from the standalone baseline."]
    if score:
        good_count = score.get("good_request_count", (export.get("good_request_count") or {}).get("avg", "unavailable"))
        lines += ["", "| Metric | Value |", "| --- | ---: |", f"| Requests / good requests | {score['request_count']} / {good_count} |", f"| Good request fraction (required ≥90%) | {score['good_request_fraction']:.2%} |", f"| Goodput | {score['goodput_requests_per_second']:.3f} req/s |", f"| Local score (goodput × 200; zero if ineligible) | {score['score_output_tokens_per_second']:.3f} output tok/s |", f"| SLA eligible | {score['eligible']} |", f"| Official workload validation | {score.get('official_validation', {}).get('valid', False)} |"]
        for name, label in (("time_to_first_token", "TTFT"), ("inter_token_latency", "ITL")):
            latency = export.get(name, {})
            if "avg" in latency and "p99" in latency:
                lines.append(f"| {label} mean / p99 | {latency['avg']:.3f} / {latency['p99']:.3f} ms |")
    elif profile_only:
        lines += ["", "Diagnostic profile only; no official benchmark or score was requested."]
    else:
        lines += ["", "Validated AIPerf result unavailable; inspect the logs."]
    if profile:
        workers = capture.get("workers", {})
        gpu_coverage = f"{sum(worker['gpu_count'] for worker in workers.values())} / 16 GPUs" if capture else "GPU coverage unverified"
        notes = sorted(root.rglob("interpretation.txt"))
        capture_note = notes[0].read_text().strip() if notes else "Capture timing unavailable; inspect profile-capture.log."
        score_note = "Diagnostic only; no official score." if profile_only else "Instrumented score; do not use for performance comparison."
        lines += ["", f"Nsight: **{len(reports)} / {expected_reports} reports**; CUDA validation **{capture.get('valid', False)}**; **{gpu_coverage}**. {capture_note} **{score_note}**"]
    else:
        lines += ["", "Nsight: disabled."]
    lines += ["", f"Archive after the archive job succeeds: `archive-host:/srv/qwen-artifacts/{run_id}/{attempt}/`.", "Reports: `profile/`; conditions and image hashes: `source/`; raw requests and score: `results/`.", ""]
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
