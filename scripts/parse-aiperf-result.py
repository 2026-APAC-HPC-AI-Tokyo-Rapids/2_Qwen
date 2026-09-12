#!/usr/bin/env python3
"""Validate AIPerf's official Qwen metrics and emit the competition score."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def metric(metrics: dict[str, Any], name: str, *, nullable: bool = False) -> float:
    value = metrics.get(name)
    if value is None and nullable:
        return 0.0
    if not isinstance(value, dict) or not isinstance(value.get("avg"), (int, float)):
        raise TypeError(f"missing numeric {name}.avg")
    result = float(value["avg"])
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}.avg")
    return result


def parse_result(metrics: dict[str, Any], concurrency: int) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    goodput = metric(metrics, "goodput")
    fraction = metric(metrics, "good_request_fraction")
    requests = metric(metrics, "request_count")
    errors = metric(metrics, "error_request_count", nullable=True)
    input_length = metric(metrics, "input_sequence_length")
    output_length = metric(metrics, "output_sequence_length")
    if requests <= 0 or goodput < 0 or not 0 <= fraction <= 1 or errors < 0:
        raise ValueError("invalid request or goodput metrics")
    eligible = fraction >= 0.90
    return {
        "schema_version": "qwen3-pd-score/v1",
        "concurrency": concurrency,
        "goodput_requests_per_second": goodput,
        "good_request_fraction": fraction,
        "eligible": eligible,
        "score_output_tokens_per_second": goodput * 200 if eligible else 0.0,
        "request_count": requests,
        "error_request_count": errors,
        "observed_input_sequence_length_avg": input_length,
        "observed_output_sequence_length_avg": output_length,
        "sla": {"time_to_first_token_ms": 3000, "inter_token_latency_ms": 20},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--require-eligible", action="store_true")
    args = parser.parse_args()
    metrics = json.loads(args.export.read_text())
    result = parse_result(metrics, args.concurrency)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if args.require_eligible and not result["eligible"]:
        raise SystemExit("official run did not meet good_request_fraction >= 0.90")


if __name__ == "__main__":
    main()
