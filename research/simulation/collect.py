"""Read archived runs without modifying them; never infer ASPIRE from GPU type."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

METRICS = {"time_to_first_token": "ms", "inter_token_latency": "ms", "request_throughput": "requests/sec", "goodput": "requests/sec", "good_request_fraction": "ratio", "request_count": "requests"}


def evidence(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_json(path):
    return json.loads(path.read_text())


def counter_rates(path):
    """IB port_*_data units are four octets; counter reset is not traffic."""
    previous, samples, resets = {}, {}, Counter()
    with path.open() as stream:
        for row in csv.DictReader(stream):
            device = row["device"]
            time = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).timestamp()
            current = (time, int(row["port_xmit_data"]), int(row["port_rcv_data"]))
            prior = previous.get(device)
            previous[device] = current
            if prior is None:
                continue
            dt = time - prior[0]
            if dt <= 0 or current[1] < prior[1] or current[2] < prior[2]:
                resets[device] += 1
                continue
            samples.setdefault(device, []).append((dt, 4 * (current[1] - prior[1]), 4 * (current[2] - prior[2])))
    return {"source": evidence(path), "kind": "observed_port_traffic_not_link_capacity_or_nixl_bandwidth", "devices": {d: {"valid_intervals": len(s), "discarded_intervals": resets[d], "observed_seconds": sum(x[0] for x in s), "tx_bytes_per_second": sum(x[1] for x in s) / sum(x[0] for x in s), "rx_bytes_per_second": sum(x[2] for x in s) / sum(x[0] for x in s)} for d, s in samples.items()}}


def export_metrics(path):
    data = read_json(path)
    metrics, errors = {}, []
    for name, unit in METRICS.items():
        metric = data.get(name)
        if not isinstance(metric, dict) or metric.get("unit") != unit:
            errors.append("missing_or_wrong_unit:" + name)
            continue
        values = {k: metric[k] for k in ("avg", "p50", "p99", "count") if k in metric}
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in values.values()) or "avg" not in values:
            errors.append("invalid_value:" + name)
            continue
        metrics[name] = {"unit": unit, **values}
    config = data.get("input_config", {})
    workload = {k: config.get(k) for k in ("datasets", "phases", "slos", "tokenizer")}
    phases = config.get("phases", [])
    if not any(p.get("kind") == "profiling" and p.get("type") == "concurrency" for p in phases):
        errors.append("missing_closed_loop_profiling_phase")
    if data.get("was_cancelled") is not False:
        errors.append("cancelled_or_unknown")
    if data.get("error_summary"):
        errors.append("request_errors")
    return {"source": evidence(path), "benchmark_id": data.get("benchmark_id"), "aiperf_version": data.get("aiperf_version"), "metrics": metrics, "workload": workload, "issues": errors}


def worker_shapes(root):
    shapes = []
    for path in sorted(root.rglob("*.log")):
        if not re.fullmatch(r"(?:prefill|decode)-\d+\.log", path.name):
            continue
        with path.open(errors="replace") as stream:
            line = next((line for line in stream if "server_args=ServerArgs(" in line), "")
        fields = {}
        for key in ("tp_size", "ep_size", "pp_size", "dp_size", "context_length", "max_total_tokens", "max_running_requests", "chunked_prefill_size", "mem_fraction_static", "kv_cache_dtype", "dtype", "attention_backend", "moe_runner_backend"):
            match = re.search(r"\b" + key + r"=('[^']*'|[^,)]+)", line)
            fields[key] = match[1].strip("'") if match else None
        if line:
            shapes.append({"worker": path.stem, "settings": fields, "source": evidence(path)})
    return shapes


def collect_run(root, provenance_path, target):
    provenance = read_json(provenance_path)
    source_dir = provenance_path.parent
    model_path, sidecar_path = source_dir / "model.json", source_dir / "container-sidecar.json"
    model = read_json(model_path) if model_path.exists() else {}
    sidecar = read_json(sidecar_path) if sidecar_path.exists() else {}
    shapes = worker_shapes(root)
    exports, hashes, duplicates = [], set(), []
    for path in sorted(root.rglob("profile_export_aiperf.json"), key=lambda p: (len(p.parts), str(p))):
        if "warmup" in str(path.relative_to(root)).lower() or "phases" in path.relative_to(root).parts:
            continue
        fingerprint = evidence(path)["sha256"]
        if fingerprint in hashes:
            duplicates.append(str(path))
            continue
        hashes.add(fingerprint)
        exports.append(export_metrics(path))
    scores = [read_json(p) for p in root.rglob("score.json")]
    problems = []
    if not any(p.is_file() and not any(part.startswith("rank-") for part in p.relative_to(root).parts) for p in root.rglob("run-success")):
        problems.append("missing_terminal_success_marker")
    if len(scores) != 1 or scores[0].get("official_validation", {}).get("valid") is not True:
        problems.append("missing_or_invalid_local_validation")
    if provenance.get("cluster") != target:
        problems.append("different_or_unknown_cluster")
    if provenance.get("profile") is not False:
        problems.append("instrumented_or_unknown_profile_mode")
    if len(exports) != 1:
        problems.append("missing_or_ambiguous_benchmark_export")
    if not shapes:
        problems.append("missing_worker_settings")
    if not model.get("model", {}).get("revision"):
        problems.append("missing_model_revision")
    if not provenance.get("container", {}).get("sha256"):
        problems.append("missing_container_identity")
    if not sidecar.get("source_commits", {}).get("sglang"):
        problems.append("missing_sglang_source")
    for export in exports:
        problems.extend(export["issues"])
    logs = list(root.rglob("check-env.log")) + list(root.rglob("environment.log")) + list(root.rglob("node-*-environment.log"))
    hardware = []
    for path in sorted(logs):
        lines = [line for line in path.read_text(errors="replace").splitlines() if re.match(r"^(GPU \d|CUDA Driver Version:|PyTorch:|sglang:|NIXL:|hostname=|job_id=|rank=|node_rank=)", line)]
        hardware.append({"source": evidence(path), "observations": lines})
    if not hardware:
        problems.append("missing_hardware_observation")
    condition = {"cluster": provenance.get("cluster"), "model": model.get("model", {}).get("revision"), "container": provenance.get("container", {}).get("sha256"), "sglang": sidecar.get("source_commits", {}).get("sglang"), "profile": provenance.get("profile"), "workers": [{"worker": w["worker"], "settings": w["settings"]} for w in shapes], "workload": exports[0]["workload"] if len(exports) == 1 else None}
    condition_key = hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest() if not problems else None
    return {"condition_key": condition_key, "root": str(root), "provenance": provenance, "source": evidence(provenance_path), "model": model.get("model"), "sglang_commit": sidecar.get("source_commits", {}).get("sglang"), "worker_settings": shapes, "hardware": hardware, "exports": exports, "duplicate_exports": duplicates, "port_traffic": [counter_rates(p) for p in sorted(root.rglob("infiniband-node-*.csv"))], "comparison_blockers": sorted(set(problems)), "candidate_for_matched_comparison": not problems}


def collect(root, target):
    records, failures, covered = [], [], set()
    for path in sorted(root.rglob("*.json")):
        if path.name not in ("provenance.json", "check-env-source.json"):
            continue
        try:
            data = read_json(path)
            if data.get("schema_version") not in ("qwen3-pd-provenance/v1", "qwen-check-env-source/v1"):
                continue
            run_root = path.parent
            while run_root.name == "source":
                run_root = run_root.parent
            records.append(collect_run(run_root, path, target))
            covered.update(run_root.rglob("profile_export_aiperf.json"))
        except (ValueError, KeyError, OSError) as error:
            failures.append({"path": str(path), "error": str(error)})
    orphan_exports = []
    for path in sorted(root.rglob("profile_export_aiperf.json")):
        if path not in covered and "warmup" not in str(path.relative_to(root)).lower() and "phases" not in path.relative_to(root).parts:
            try:
                orphan_exports.append(export_metrics(path))
            except (ValueError, KeyError, OSError) as error:
                failures.append({"path": str(path), "error": str(error)})
    return {"schema_version": "qwen-calibration-inventory/v1", "target_cluster": target, "archive_root": str(root), "runs": records, "unattributed_exports": orphan_exports, "read_errors": failures, "summary": {"runs": len(records), "target_runs": sum(r["provenance"].get("cluster") == target for r in records), "target_exports": sum(len(r["exports"]) for r in records if r["provenance"].get("cluster") == target), "candidate_runs": sum(r["candidate_for_matched_comparison"] for r in records)}, "accuracy_improvement_demonstrated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cluster", default="aspire2ap")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("archive root must exist")
    result = collect(args.root.resolve(), args.cluster)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["summary"]))
    return 1 if result["read_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
