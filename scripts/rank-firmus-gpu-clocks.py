"""Build configs/firmus-gpu-sm-clock.json from archived Firmus GPU telemetry.

Under the 700 W cap every Prefill GPU draws ~692 W, so its SM clock is its
speed. In the EP4 Prefill runs the four GPUs of a worker do the same work in
lockstep, so a lower clock there is a property of the GPU (2026-10-30 research
note: g4:2 and g1:3 are the slowest in every run). The table keeps the mean
clock of each Prefill GPU over the busy window (g1 GPU 0 above 550 W, first 10
s and last 5 s dropped) of the given runs, keyed by GPU UUID. The Decode GPUs
(tail GPUs 4-7) never ran Prefill and are left out.

Usage: rank-firmus-gpu-clocks.py OUTPUT RUN...  (reads /srv/qwen-artifacts/RUN/1/logs/logs/gpu-telemetry-*.csv)
"""
import csv
import datetime
import itertools
import json
import os
import statistics
import sys
from pathlib import Path

ARCHIVE = Path(os.environ.get("QWEN_ARTIFACT_ARCHIVE_ROOT", "/srv/qwen-artifacts"))
PREFILL = {"g1": range(8), "g4": range(4)}


def rows(path):
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for x in reader:
            if len(x) < 8 or not x[3].strip().endswith("W"):
                continue
            # nvidia-smi prints node-local time; only differences are used.
            t = datetime.datetime.strptime(x[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.UTC).timestamp()
            yield t, int(x[1]), x[2].strip(), float(x[3].split()[0]), float(x[4].split()[0])


def busy_window(head):
    hot = sorted(t for t, i, _, w, _ in head if i == 0 and w > 550)
    spans, start = [], hot[0]
    for a, b in itertools.pairwise(hot):
        if b - a > 5:
            spans.append((start, a))
            start = b
    spans.append((start, hot[-1]))
    a, b = max(spans, key=lambda s: s[1] - s[0])
    return a + 10, b - 5


def main(output, runs):
    clocks = {}
    for run in runs:
        logs = ARCHIVE / run / "1/logs/logs"
        data = {host: list(rows(logs / f"gpu-telemetry-{host}.csv")) for host in PREFILL}
        a, b = busy_window(data["g1"])
        for host, indices in PREFILL.items():
            for index in indices:
                sample = [(u, c) for t, i, u, _, c in data[host] if i == index and a <= t <= b]
                uuid = sample[0][0]
                entry = clocks.setdefault(uuid, {"host": host, "index": index, "runs": []})
                assert (entry["host"], entry["index"]) == (host, index)
                entry["runs"].append(statistics.fmean(c for _, c in sample))
    gpus = {u: {"host": e["host"], "index": e["index"], "sm_mhz": round(statistics.fmean(e["runs"]), 1), "sd_mhz": round(statistics.stdev(e["runs"]), 1)} for u, e in clocks.items()}
    table = {"schema": "firmus-gpu-sm-clock/v1", "load": "EP4 Prefill at the 700 W cap (four GPUs per worker in lockstep)", "runs": runs, "gpus": dict(sorted(gpus.items(), key=lambda kv: (kv[1]["host"], kv[1]["index"])))}
    Path(output).write_text(json.dumps(table, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
