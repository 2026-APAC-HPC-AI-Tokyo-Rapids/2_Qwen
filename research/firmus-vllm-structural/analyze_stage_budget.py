"""Read canonical AIPerf exports to budget the client-visible serving stages.

No scoring changes: stationary slices are diagnostic, never replacement scores.
Before-first-token time includes prefill, queues, transfer, and protocol overhead.
After-first-token occupancy is a client-visible proxy, not engine batch size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def overlap_seconds(start: int, end: int, left: int, right: int) -> float:
    return max(0, min(end, right) - max(start, left)) / 1e9


def analyze(directory: Path) -> dict:
    raw = directory / "profile_export.jsonl"
    export_path = directory / "profile_export_aiperf.json"
    score_path = directory / "score.json"
    export = json.loads(export_path.read_text())
    score = json.loads(score_path.read_text())
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line]
    measured = [r for r in rows if r["metadata"]["benchmark_phase"] == "profiling"]
    if len(measured) != score["request_count"]:
        raise ValueError("raw measured count differs from score")
    first = min(r["metadata"]["request_start_ns"] for r in measured)

    def metric(row, name):
        return row["metrics"][name]["value"]

    def summarize(left_s, right_s):
        left = first + int(left_s * 1e9)
        right = first + int(right_s * 1e9)
        started = [
            r for r in measured if left <= r["metadata"]["request_start_ns"] < right
        ]
        before = after = 0.0
        for r in measured:
            start = r["metadata"]["request_start_ns"]
            end = r["metadata"]["request_end_ns"]
            token = start + round(metric(r, "time_to_first_token") * 1e6)
            if not start <= token <= end + 1000000:
                raise ValueError("first token lies outside request interval")
            token = min(token, end)
            before += overlap_seconds(start, token, left, right)
            after += overlap_seconds(token, end, left, right)
        return {
            "start_seconds_inclusive": left_s,
            "end_seconds_exclusive": right_s,
            "requests_started": len(started),
            "request_start_rate_per_second": len(started) / (right_s - left_s),
            "client_before_first_token_mean_inflight": before / (right_s - left_s),
            "client_after_first_token_mean_inflight": after / (right_s - left_s),
            "ttft_mean_ms": mean(metric(r, "time_to_first_token") for r in started),
            "decode_duration_mean_ms": mean(
                metric(r, "decode_duration") for r in started
            ),
            "request_latency_mean_ms": mean(
                metric(r, "request_latency") for r in started
            ),
            "itl_mean_ms": mean(metric(r, "inter_token_latency") for r in started),
            "ttft_misses": sum(
                metric(r, "time_to_first_token") > 3000 for r in started
            ),
            "itl_misses": sum(metric(r, "inter_token_latency") > 20 for r in started),
        }

    latency = mean(metric(r, "request_latency") for r in measured) / 1000
    good_tps = export["goodput"]["avg"] * 200
    targets = []
    for output_tps in (7000, 2 * good_tps, 10000):
        # A zero measured rate cannot define a meaningful doubling target.
        if output_tps <= 0:
            continue
        good_rps = output_tps / 200
        fraction = score["good_request_fraction"]
        required_rps = good_rps / fraction if fraction > 0 else None
        targets.append(
            {
                "target_good_output_tokens_per_second": output_tps,
                "minimum_good_requests_per_second": good_rps,
                "minimum_configured_input_tokens_per_second": good_rps * 4000,
                "input_tokens_per_second_at_current_good_fraction": required_rps * 4000
                if required_rps is not None
                else None,
                "mean_request_seconds_at_current_concurrency_and_good_fraction": score[
                    "concurrency"
                ]
                / required_rps
                if required_rps is not None
                else None,
                "needed_concurrency_at_current_mean_latency_and_good_fraction": required_rps
                * latency
                if required_rps is not None
                else None,
            }
        )
    return {
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (raw, export_path, score_path)
        },
        "concurrency": score["concurrency"],
        "measured_requests": len(measured),
        "good_output_tokens_per_second": good_tps,
        "local_gated_value": score["score_output_tokens_per_second"],
        "sla_eligible": score["eligible"],
        "good_request_fraction": score["good_request_fraction"],
        "benchmark_duration_seconds": export["benchmark_duration"]["avg"],
        "observed_input_processing_tokens_per_second": score[
            "observed_input_sequence_length_sum"
        ]
        / export["benchmark_duration"]["avg"],
        "raw_output_tokens_per_second": export["output_token_throughput"]["avg"],
        "windows": [summarize(0, 120), summarize(30, 120)],
        "target_sensitivity": targets,
        "limitations": [
            "Client-visible intervals cannot separate prefill execution, prefill queue, NIXL, decoder admission, or HTTP overhead.",
            "After-first-token inflight is not measured engine batch size.",
            "Window origin is first measured request start; stationary slices never modify the stored raw result.",
            "A zero good fraction makes current-fraction targets unattainable; their required rates and occupancy are null.",
            "Little-law target calculations hold current latency or good fraction hypothetically; they are not throughput predictions.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.results), indent=2))
