#!/usr/bin/env python3
"""Write a small Actions summary from collected artifacts, including failed runs."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def read_artifact(root: Path, filename: str) -> dict:
    paths = sorted((path for path in root.rglob(filename) if not (path.parent.name == "client" and path.parent.parent.name == "profile")), key=lambda path: (len(path.parts), str(path))) if root.exists() else []
    return json.loads(paths[0].read_text()) if paths else {}


def summarize(root: Path, job_id: str, status: str, cluster: str = "firmus") -> str:
    identity = read_artifact(root, "provenance.json")
    aspire = cluster == "aspire2ap"
    config = read_artifact(root, "aspire-serving.json" if aspire else "firmus-framework.json")
    score = read_artifact(root, "score.json")
    export = read_artifact(root, "profile_export_aiperf.json")
    capture = read_artifact(root, "validation.json")
    routing = read_artifact(root, "prefill-routing-summary.json")
    profile = identity.get("profile", False)
    reports = list(root.rglob("*.nsys-rep")) if root.exists() else []
    run_id = os.environ.get("GITHUB_RUN_ID", "unknown")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "unknown")
    workload = identity.get("workload", "official" if aspire else "reference-07")
    phases = "20 / 30 / 90 (profile only)" if identity.get("profile_only") else "60 / 120 / 30" if workload == "official" else "20 / 60 / 15"
    topology = "P2 × TP4 / D2 × TP4 EP1" if identity.get("framework") == "sglang" and workload == "official" else "P4 × TP2 / D2 × TP4 EP4"
    endpoint_files = list(root.rglob("worker-endpoints.tsv")) if root.exists() else []
    expected_reports = 6
    if endpoint_files and not aspire:
        endpoints = [line.split("\t") for line in endpoint_files[0].read_text().splitlines()]
        expected_reports = len(endpoints)
        groups = Counter(("P" if name.startswith("prefill-") else "D", int(tp)) for name, _host, _port, tp in endpoints)
        topology = " / ".join(f"{role}{count} × TP{tp}" for (role, tp), count in sorted(groups.items(), reverse=True))
    if config.get("prefill"):
        p = config["prefill"]
        topology = f"P{p['workers']} × (Attention TP{p['tp'] // p['dp']} DP{p['dp']} / MoE EP{p['ep']}) / D1 × TP4 EP1"
        if p.get("pp", 1) > 1:
            topology = f"P{p['workers']} × PP{p['pp']} TP{p['tp']} EP{p['ep']} / D1 × TP4 DP{config['decode_dp_size']} EP4"
    if aspire:
        workers = config.get("workers", [])
        expected_reports = len(workers)
        groups = Counter(("P" if worker["role"] == "prefill" else "D", worker["tp"], worker["ep"]) for worker in workers)
        topology = " / ".join(f"{role}{count} × TP{tp} DP{config['dp_size']} EP{ep}" for (role, tp, ep), count in sorted(groups.items(), reverse=True)) or "topology unavailable"
    title, scheduler, hardware = ("ASPIRE 2A+", "PBS", "H100 × 16") if aspire else ("Firmus", "Slurm", "H200 × 16")
    lines = [f"### {title} Qwen", "", f"- Status: **{status}**; {scheduler} `{job_id}`; completion marker: **{'present' if list(root.rglob('run-success')) else 'missing'}**.", f"- Framework: **{identity.get('framework', 'sglang' if aspire else 'unavailable')}**; {hardware}; {topology}; concurrency **{config.get('concurrency', 'unavailable')}**.", f"- Workload: {workload}, ISL 4000 / OSL 200, warmup / benchmark / grace **{phases} s**; **not an official submission**."]
    if identity.get("dynamo"):
        dynamo = identity["dynamo"]
        lines += [f"- Dynamo **{dynamo['version']}**; engine **{dynamo['engine_version']}**; frontend → TCP → PD workers; NIXL KV transfer. Engine version differs from the standalone baseline."]
    if score:
        good_count = score.get("good_request_count", (export.get("good_request_count") or {}).get("avg", "unavailable"))
        lines += ["", "| Metric | Value |", "| --- | ---: |", f"| Requests / good requests | {score['request_count']} / {good_count} |", f"| Request errors | {score.get('error_request_count', 'unavailable')} |", f"| Good request fraction (required ≥90%) | {score['good_request_fraction']:.2%} |", f"| Goodput | {score['goodput_requests_per_second']:.3f} req/s |", f"| Local score (good output tokens; zero if ineligible) | {score['score_output_tokens_per_second']:.3f} output tok/s |", f"| SLA eligible | {score['eligible']} |", f"| Official workload validation | {score.get('official_validation', {}).get('valid', False)} |"]
        for name, label in (("time_to_first_token", "TTFT"), ("inter_token_latency", "ITL")):
            latency = export.get(name, {})
            if "avg" in latency and "p99" in latency:
                lines.append(f"| {label} mean / p99 | {latency['avg']:.3f} / {latency['p99']:.3f} ms |")
    elif identity.get("profile_only"):
        lines += ["", "Profile-only diagnostic: ordinary benchmark omitted; no official score was produced."]
    else:
        lines += ["", "Validated AIPerf result unavailable; inspect the logs."]
    if profile:
        workers = capture.get("workers", {})
        gpu_coverage = f"{sum(worker['gpu_count'] for worker in workers.values())} / 16 GPUs" if capture else "GPU coverage unverified"
        notes = sorted(root.rglob("interpretation.txt"))
        capture_note = notes[0].read_text().strip() if notes else "Capture timing unavailable; inspect profile-capture.log."
        score_note = "Normal score is from the earlier tracing-disabled pass; the separate profile/client pass is instrumented and must not be used for performance comparison." if identity.get("normal_benchmark_before_profile") else "Instrumented score; do not use for performance comparison."
        if identity.get("profile_only"):
            score_note = "Profile-only diagnostic; no official score. Do not use instrumented client timings for adoption."
        lines += ["", f"Nsight: **{len(reports)} / {expected_reports} reports**; CUDA validation **{capture.get('valid', False)}**; **{gpu_coverage}**. {capture_note} **{score_note}**"]
    else:
        lines += ["", "Nsight: disabled."]
    if config.get("prefill", {}).get("routing_capture_samples", 0):
        lines += ["", f"Prefill source-rank routing diagnostic: **{'valid' if routing.get('valid') else 'missing or invalid'}**; {routing.get('expected_workers', 'unavailable')} workers × {routing.get('expected_ranks', 'unavailable')} ranks × {routing.get('samples_per_rank', 'unavailable')} samples. Raw captures: `results/prefill-routing/`; aggregate: `results/prefill-routing-summary.json`. Routing copies affect the separate profile pass and its Nsight timings; do not use those timings for performance adoption. The score above remains from the normal benchmark pass."]
    lines += ["", f"Archive after the archive job succeeds: `archive-host:/srv/qwen-artifacts/{run_id}/{attempt}/`.", "Reports: `profile/`; conditions and image hashes: `source/`; raw requests and score: `results/`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--cluster", choices=("firmus", "aspire2ap"), default="firmus")
    args = parser.parse_args()
    print(summarize(args.artifacts, args.job_id, args.status, args.cluster))


if __name__ == "__main__":
    main()
