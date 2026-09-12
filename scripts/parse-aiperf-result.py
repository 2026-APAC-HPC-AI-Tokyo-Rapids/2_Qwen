#!/usr/bin/env python3
"""Validate AIPerf's official Qwen metrics and emit the competition score."""

from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


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


TOKENIZER_REVISION = "ac9c66cc9b46af7306746a9250f23d47083d689e"


def validate_official_workload(command: Path, provenance: Path, concurrency: int) -> None:
    """Require the fixed command and measured runtime identity, without tolerance."""
    identity = dict(line.split("=", 1) for line in provenance.read_text().splitlines())
    gateway = identity.get("gateway", "")
    url = urlsplit(gateway)
    if url.scheme != "http" or not url.hostname or url.path != "/v1/chat/completions" or url.query or url.fragment or url.username or url.password:
        raise ValueError("invalid gateway in AIPerf provenance")
    required = {"aiperf_version": "0.12.0", "mode": "official", "concurrency": str(concurrency), "tokenizer_revision": TOKENIZER_REVISION}
    if any(identity.get(key) != value for key, value in required.items()):
        raise ValueError("AIPerf provenance does not match the pinned official workload")
    expected = ["/opt/sglang/venv/bin/aiperf", "profile", "--url", f"http://{url.netloc}"]
    expected += shlex.split('--endpoint-type chat --endpoint /v1/chat/completions --model qwen --tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507 --streaming -H "Accept: text/event-stream"')
    expected += ["--concurrency", str(concurrency), "--workers-max", str(concurrency)]
    expected += shlex.split('--api-key NOT_USED --warmup-duration 60 --benchmark-duration 120 --benchmark-grace-period 30 --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 --output-tokens-mean 200 --output-tokens-stddev 0 --extra-inputs min_tokens:200 --extra-inputs max_tokens:200 --extra-inputs ignore_eos:true --extra-inputs temperature:0.0 --extra-inputs repetition_penalty:1.0 --num-dataset-entries 12800 --random-seed 2026 --cache-bust first_turn_prefix --goodput "time_to_first_token:3000 inter_token_latency:20" --ui simple')
    if shlex.split(command.read_text()) != expected:
        raise ValueError("AIPerf command does not match the fixed official workload")


def validate_official_result(result: dict[str, Any], command: Path | None, provenance: Path | None) -> dict[str, Any]:
    if command is None or provenance is None:
        raise ValueError("official validation requires --command and --provenance")
    validate_official_workload(command, provenance, result["concurrency"])
    if not result["eligible"]:
        raise ValueError("good_request_fraction is below 0.90")
    # AIPerf 0.12.0 subtracts the rounded 14-token cache-bust estimate from
    # synthetic ISL. Its client export measures that generated prompt (3986).
    # Keep the original 4000 observation valid too, but admit no nearby values.
    input_length = result["observed_input_sequence_length_avg"]
    if input_length not in (4000.0, 3986.0):
        raise ValueError("average input sequence length is neither 4000 nor pinned AIPerf's 3986 tokens")
    if result["observed_output_sequence_length_avg"] != 200.0:
        raise ValueError("average output sequence length is not 200 tokens")
    return {"valid": True, "configured_input_sequence_length": 4000, "configured_output_sequence_length": 200, "client_input_token_adjustment": 4000 - input_length, "aiperf_version": "0.12.0", "tokenizer_revision": TOKENIZER_REVISION}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--require-eligible", action="store_true")
    parser.add_argument("--command", type=Path)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    metrics = json.loads(args.export.read_text())
    result = parse_result(metrics, args.concurrency)
    validation_error = None
    if args.require_eligible:
        try:
            result["official_validation"] = validate_official_result(result, args.command, args.provenance)
        except (ValueError, OSError) as error:
            validation_error = str(error)
            result["official_validation"] = {"valid": False, "error": validation_error}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if validation_error is not None:
        raise SystemExit(f"invalid official result: {validation_error}")


if __name__ == "__main__":
    main()
