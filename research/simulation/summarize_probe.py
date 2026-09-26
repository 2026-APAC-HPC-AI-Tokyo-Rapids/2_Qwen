"""Summarize a complete hardware probe without creating AISimulate calibration data."""

import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(root):
    records = [json.loads(p.read_text()) for p in sorted(root.rglob("rank-*.json"))]
    if len(records) != 16 or {r["rank"] for r in records} != set(range(16)):
        raise ValueError("exactly one result from each of 16 ranks required")
    if len({r["job_id"] for r in records}) != 1 or any(r["cluster"] != "aspire2ap" for r in records):
        raise ValueError("mixed allocation/cluster")
    hosts = {r["hostname"] for r in records}
    if len(hosts) != 2 or any(sum(r["hostname"] == h for r in records) != 8 for h in hosts):
        raise ValueError("requires two distinct hosts with eight ranks each")
    expected = {( "device_copy", None), ("bf16_gemm", None)} | {(op, size) for op in ("nccl_intra_node_all_reduce", "nccl_inter_node_pair_all_reduce") for size in (1048576, 16777216, 268435456)}
    groups = {}
    for record in records:
        measured = record["measurements"]
        keys = [(m["operation"], m.get("payload_bytes") if m["operation"].startswith("nccl_") else None) for m in measured]
        if len(keys) != 8 or set(keys) != expected:
            raise ValueError("incomplete measurement set")
        for key, measurement in zip(keys, measured, strict=True):
            values = measurement["latency_ms_samples"]
            if len(values) != 5 or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError("five finite positive latency samples required")
            groups.setdefault(key, []).append((record["rank"], measurement))
    summary = []
    for (operation, size), measurements in sorted(groups.items()):
        latencies = [statistics.median(m["latency_ms_samples"]) for _, m in measurements]
        row = {"operation": operation, "payload_bytes": size, "rank_median_latency_ms": {"min": min(latencies), "median": statistics.median(latencies), "max": max(latencies)}}
        if operation == "device_copy":
            rates = [m["accounted_bytes_per_iteration"] / statistics.median(m["latency_ms_samples"]) / 1e6 for _, m in measurements]
            row["read_plus_write_GB_per_second"] = {"min": min(rates), "median": statistics.median(rates), "max": max(rates)}
        elif operation == "bf16_gemm":
            rates = [m["flops_per_iteration"] / statistics.median(m["latency_ms_samples"]) / 1e9 for _, m in measurements]
            row["TFLOP_per_second"] = {"min": min(rates), "median": statistics.median(rates), "max": max(rates)}
        else:
            row["algorithm_GB_per_second_slowest_rank"] = size / max(latencies) / 1e6
        summary.append(row)
    nixl = []
    if any('nixl_measurements' in record for record in records):
        destination_tps = (4, 8) if records[0].get('nixl_geometry_version') == 2 else (8,)
        expected_nixl = {(tp, destination_tp, tokens) for tp in (4, 8) for destination_tp in destination_tps for tokens in (128, 1024, 4000, 4200)}
        for record in records:
            points = record.get('nixl_measurements', [])
            if len(points) != len(expected_nixl) or {(p['source_tp'], p.get('destination_tp', 8), p['prompt_tokens']) for p in points} != expected_nixl:
                raise ValueError('incomplete NIXL measurements')
            for point in points:
                if not point['integrity_verified'] or len(point['latency_ms_samples']) != 5 or any(not math.isfinite(x) or x <= 0 for x in point['latency_ms_samples']):
                    raise ValueError('invalid NIXL measurement')
        for point in records[0]['nixl_measurements']:
            ms = statistics.median(point['latency_ms_samples'])
            nixl.append({**point, 'median_latency_ms': ms, 'physical_GB_per_second': point['physical_bytes'] / ms / 1e6, 'effective_unique_KV_GB_per_second': point['unique_kv_bytes'] / ms / 1e6})
    return {"nixl_segmented_writes": nixl, "schema_version": "qwen-hardware-probe-summary/v1", "job_id": records[0]["job_id"], "hosts": sorted(hosts), "rank_count": 16, "measurements": summary, "nixl_bandwidth_gb_per_second": None, "qwen_forward_pass_calibration": None, "accuracy_improvement_demonstrated": False, "limitations": records[0]["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
