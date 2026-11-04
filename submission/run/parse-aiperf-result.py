"""Score an official AIPerf export.

Good output tok/s = goodput (requests/s that met TTFT <= 3000 ms and ITL <= 20 ms) x 200 output tokens.
A run is eligible when at least 90% of its requests met that SLA.

usage: parse-aiperf-result.py profile_export_aiperf.json score.json
"""
import json
import sys
from pathlib import Path

OUTPUT_TOKENS = 200


def score(metrics):
    def avg(name):
        return metrics[name]["avg"]

    goodput = avg("goodput")
    fraction = avg("good_request_fraction")
    result = {
        "good_output_tokens_per_second": goodput * OUTPUT_TOKENS,
        "goodput_requests_per_second": goodput,
        "good_request_fraction": fraction,
        "eligible": fraction >= 0.90,
        "request_count": avg("request_count"),
        # AIPerf leaves this metric out of the export when no request failed.
        "error_request_count": metrics.get("error_request_count", {"avg": 0.0})["avg"],
        "osl_mismatch_count": avg("osl_mismatch_count"),
        "sla": {"time_to_first_token_ms": 3000, "inter_token_latency_ms": 20},
    }
    for kind in ("input", "output"):
        for statistic in ("avg", "min", "max", "count", "sum"):
            result[f"observed_{kind}_sequence_length_{statistic}"] = metrics[f"{kind}_sequence_length"][statistic]
    return result


if __name__ == "__main__":
    export, output = map(Path, sys.argv[1:])
    result = score(json.loads(export.read_text()))
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
