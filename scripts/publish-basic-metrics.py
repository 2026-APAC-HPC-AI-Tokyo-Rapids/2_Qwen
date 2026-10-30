"""Publish a terminal Basic archive through the existing run/key/value sender."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from qwen_experiment_metrics.canonical import encode_logical_value, fact_row_bytes
from qwen_experiment_metrics.publish import RawRow, send_raw


def collect(archive_root: Path, measurement: str, repository: str):
    if not re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", measurement):
        raise ValueError("measurement must be a positive Actions run/attempt")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid repository identity")
    run, attempt = measurement.split("/")
    archive = archive_root / run / attempt
    result = archive / "results/results"
    if not (result / "score.json").exists():
        return None
    subprocess.run(["sha256sum", "--check", "--quiet", "SHA256SUMS"], cwd=archive, check=True)
    accounting = (archive / "logs/logs/slurm-accounting.txt").read_text().strip().splitlines()[0].split("|")
    if len(accounting) != 5 or accounting[1].split()[0] not in {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"}:
        raise ValueError("scheduler state is not terminal")
    config = json.loads((archive / "source/source/firmus-framework.json").read_text())
    provenance = json.loads((archive / "source/source/provenance.json").read_text())
    score = json.loads((result / "score.json").read_text())
    metrics = json.loads((result / "profile_export_aiperf.json").read_text())
    if config.get("tier") != "basic" or config.get("workload") != "official" or provenance.get("profile") is not False:
        raise ValueError("only normal Basic official measurements can be published")
    if provenance.get("run_id") != run or metrics.get("aiperf_version") != "0.12.0" or score.get("concurrency") != config["concurrency"]:
        raise ValueError("archive measurement identity does not match")
    config.pop("images", None)
    values = {
        "aiperf.provenance": {"repository": repository, "measurement": measurement, "prepared": provenance, "slurm_job_id": accounting[0], "slurm_state": accounting[1], "archive": f"qwen-artifacts/{measurement}"},
        "aiperf.configuration": config,
        "aiperf.score": score,
    }
    for key in ("time_to_first_token", "inter_token_latency", "output_token_throughput", "input_token_throughput", "request_latency", "benchmark_duration"):
        values[f"aiperf.{key}"] = metrics[key]
    identity = f"github:{repository}:{measurement}"
    rows = [RawRow(identity, key, encode_logical_value(value)) for key, value in values.items()]
    # Keep the derived producer fragment beside, not inside, the immutable raw archive.
    return archive.parent / f"metrics-{attempt}.jsonl", rows


def publish(archive_root: Path, measurement: str, repository: str, sender=send_raw):
    collected = collect(archive_root, measurement, repository)
    if collected is None:
        print(f"No archived AIPerf result for {measurement}; nothing to publish")
        return True
    destination, rows = collected
    payload = b"".join(fact_row_bytes(row.run, row.key, row.value) + b"\n" for row in rows)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError("completed metrics fragment differs from the archived source")
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    success = sender(rows[0].run, rows)
    print(f"BigQuery publication {'succeeded' if success else 'failed'}: {measurement}, {len(rows)} rows, fragment={destination}")
    return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("measurement", help="Actions run ID/attempt, not an experiment option")
    args = parser.parse_args()
    try:
        return 0 if publish(args.archive_root, args.measurement, os.environ["GITHUB_REPOSITORY"]) else 1
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"Basic metrics publication warning: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
