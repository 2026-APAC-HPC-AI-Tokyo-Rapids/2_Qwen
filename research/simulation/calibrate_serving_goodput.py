"""Fit one serving-goodput scale on one run and evaluate disjoint runs.

This is an empirical baseline, not a correction to GPU forward measurements,
request latencies, eligibility, or the native AISimulate engine.
"""
import argparse
import json
import math
import statistics
from pathlib import Path


def evaluate(data):
    train = data["calibration"]
    holdout = data["holdout"]
    if not holdout or train["run_id"] in {row["run_id"] for row in holdout}:
        raise ValueError("calibration and evaluation runs must be disjoint")
    if len({row["run_id"] for row in holdout}) != len(holdout):
        raise ValueError("duplicate evaluation run")
    for row in [train, *holdout]:
        if not row["parser_valid"]:
            raise ValueError("only validated ordinary runs can calibrate goodput")
        for key in ("predicted_goodput", "measured_goodput"):
            if not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError("positive finite goodput is required")
    factor = train["measured_goodput"] / train["predicted_goodput"]
    results = []
    for row in holdout:
        calibrated = row["predicted_goodput"] * factor
        results.append({**row, "calibrated_goodput": calibrated, "original_ape_percent": abs(row["predicted_goodput"] / row["measured_goodput"] - 1)*100, "calibrated_ape_percent": abs(calibrated / row["measured_goodput"] - 1)*100})
    return {"kind": "empirical_goodput_scale_not_native_engine_calibration", "official_result": False, "calibration": train, "scale": factor, "holdout": results, "original_median_ape_percent": statistics.median(r["original_ape_percent"] for r in results), "calibrated_median_ape_percent": statistics.median(r["calibrated_ape_percent"] for r in results), "limitations": ["one calibration run; uncertainty not estimated", "restricted to target-only BF16 KV and the fixed image/TP4 EP1 layout", "does not correct TTFT, ITL, routing, or predicted SLA", "does not validate extrapolation or alternate backend/speculation rankings"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(evaluate(json.loads(args.dataset.read_text())), indent=2) + "\n")
