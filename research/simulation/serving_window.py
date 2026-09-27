"""Evaluate a warmed simulation window; this is not the official AIPerf parser."""
import argparse
import json
import statistics
from pathlib import Path


def summarize(requests, warmup_seconds=60, duration_seconds=120, grace_seconds=30):
    start = warmup_seconds * 1000
    stop = start + duration_seconds * 1000
    deadline = stop + grace_seconds * 1000
    selected = [r for r in requests if start <= r["arrival_time_ms"] < stop]
    if not selected:
        raise ValueError("simulation has no requests in the measurement window")
    if any(r["arrival_time_ms"] >= stop for r in requests):
        raise ValueError("rerun with the count of arrivals before measurement end to stop background arrivals during drain")
    if max(r["arrival_time_ms"] for r in selected) < stop - max(r["e2e_latency_ms"] for r in selected):
        raise ValueError("simulation ends before the measurement window")
    complete = [r for r in selected if r["terminal_status"] == "completed" and r["terminal_time_ms"] <= deadline]
    good = sum(r["ttft_ms"] <= 3000 and r["itl_ms"] <= 20 for r in complete)
    first = min(r["arrival_time_ms"] for r in selected)
    last = min(deadline, max(r["terminal_time_ms"] for r in selected))
    fraction = good / len(selected)
    return {"official_result": False, "kind": "synthetic_warmed_window", "warmup_seconds": warmup_seconds, "duration_seconds": duration_seconds, "grace_seconds": grace_seconds, "requests": len(selected), "completed": len(complete), "good_requests": good, "joint_sla_fraction": fraction, "elapsed_seconds": (last - first) / 1000, "good_tokens_per_second": good * 200 / ((last - first) / 1000) if fraction >= .9 else 0, "mean_ttft_ms": statistics.mean(r["ttft_ms"] for r in complete), "mean_itl_ms": statistics.mean(r["itl_ms"] for r in complete), "limitations": ["synthetic fixed lengths, not AIPerf tokenized prompts", "closed-loop warmup selection approximates the client protocol", "latency and goodput use native simulated request boundaries"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.requests.read_text().splitlines()]
    args.output.write_text(json.dumps(summarize(rows), indent=2) + "\n")
