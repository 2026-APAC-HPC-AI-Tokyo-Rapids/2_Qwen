#!/usr/bin/env python3
"""Validate the unchanged reference-07 workload; never label it an official run."""

from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path


def parse_reference(export: dict, records: list[dict], concurrency: int) -> dict:
    if export.get("aiperf_version") != "0.12.0":
        raise ValueError("expected pinned AIPerf 0.12.0")
    if export.get("was_cancelled") is not False or export.get("error_summary") != []:
        raise ValueError("benchmark was cancelled or contains request errors")
    config = export["input_config"]
    phases = [
        {"name": "warmup", "kind": "warmup", "type": "concurrency", "exclude_from_results": True, "duration": 20.0, "concurrency": concurrency},
        {"name": "profiling", "kind": "profiling", "type": "concurrency", "duration": 60.0, "concurrency": concurrency, "grace_period": 15.0},
    ]
    if config["phases"] != phases:
        raise ValueError("reference-07 phases or concurrency differ")
    endpoint = config["endpoint"]
    expected_extra = {"min_tokens": 200, "max_tokens": 200, "ignore_eos": True, "temperature": 0.0, "repetition_penalty": 1.0}
    if endpoint["type"] != "chat" or endpoint["path"] != "/v1/chat/completions" or endpoint["streaming"] is not True or endpoint["extra"] != expected_extra:
        raise ValueError("reference endpoint or generation settings differ")
    dataset = config["datasets"]
    expected_dataset = [{"name": "main", "type": "synthetic", "entries": 12800, "random_seed": 2026, "prompts": {"isl": {"mean": 4000.0, "stddev": 0.0}, "osl": {"mean": 200.0, "stddev": 0.0}, "cache_bust": {"target": "first_turn_prefix"}}}]
    if dataset != expected_dataset or config["models"] != {"items": [{"name": "qwen"}]}:
        raise ValueError("reference model or synthetic dataset differs")
    if Path(config["tokenizer"]["name"]).name != "ac9c66cc9b46af7306746a9250f23d47083d689e":
        raise ValueError("tokenizer revision differs")
    measured = [row for row in records if row["metadata"]["benchmark_phase"] == "profiling"]
    count = export["request_count"]["avg"]
    duration = export["benchmark_duration"]["avg"]
    if not measured or len(measured) != count or not math.isfinite(duration) or duration < 60:
        raise ValueError("incomplete benchmark accounting")
    good = 0
    for row in measured:
        if row["metadata"].get("was_cancelled") is not False:
            raise ValueError("cancelled measured request")
        metrics = row["metrics"]
        ttft = metrics["time_to_first_token"]["value"]
        itl = metrics["inter_token_latency"]["value"]
        if metrics["time_to_first_token"]["unit"] != "ms" or metrics["inter_token_latency"]["unit"] != "ms" or not all(math.isfinite(value) and value >= 0 for value in (ttft, itl)):
            raise ValueError("invalid per-request latency")
        if not 3984 <= metrics["input_sequence_length"]["value"] <= 3988 or not 190 <= metrics["output_sequence_length"]["value"] <= 210:
            raise ValueError("request token lengths differ from pinned workload")
        good += ttft <= 3000 and itl <= 20
    enriched = dict(export)
    enriched["goodput"] = {"avg": good / duration}
    enriched["good_request_fraction"] = {"avg": good / count}
    parse_result = runpy.run_path(str(Path(__file__).with_name("parse-aiperf-result.py")))["parse_result"]
    result = parse_result(enriched, concurrency)
    if result["error_request_count"] != 0 or result["osl_mismatch_count"] != 0:
        raise ValueError("request errors or output-length mismatches")
    result.update(workload="reference-07", official_submission=False, benchmark_duration_seconds=duration, good_request_count=good, reference_validation={"valid": True}, official_validation={"valid": False, "reason": "Reference timings 20/60/15; full official workload is 60/120/30."})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--concurrency", type=int, required=True)
    args = parser.parse_args()
    export = json.loads((args.results / "profile_export_aiperf.json").read_text())
    records = [json.loads(line) for line in (args.results / "profile_export.jsonl").read_text().splitlines() if line]
    result = parse_reference(export, records, args.concurrency)
    (args.results / "aiperf-command.txt").write_text(export["run_info"]["cli_command"] + "\n")
    (args.results / "score.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
