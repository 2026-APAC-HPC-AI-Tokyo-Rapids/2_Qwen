"""Summarize native SGLang histogram deltas from a separate profile run."""
import argparse
import json
import re
from pathlib import Path

LINE = re.compile(r'^(\w[\w:]*)(\{.*\})?\s+([-+0-9.eE]+)(?:\s+.*)?$')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def read_samples(path):
    samples = {}
    for line in path.read_text().splitlines():
        match = LINE.fullmatch(line)
        if not match:
            continue
        name, labels, value = match.groups()
        if name.endswith(("_sum", "_count")):
            key = (name, tuple(sorted(LABEL.findall(labels or ""))))
            if key in samples:
                raise ValueError(f"duplicate series in {path}: {key}")
            samples[key] = float(value)
    return samples


def summarize(directory):
    workers = {}
    for after_path in sorted(directory.glob("metrics-after-*.prom")):
        worker = after_path.name.removeprefix("metrics-after-").removesuffix(".prom")
        before = read_samples(directory / f"metrics-before-{worker}.prom")
        after = read_samples(after_path)
        rows = []
        for (name, labels), value in sorted(after.items()):
            if not name.endswith("_sum"):
                continue
            count_key = (name.removesuffix("_sum") + "_count", labels)
            if count_key not in after:
                continue
            total = value - before.get((name, labels), 0)
            count = after[count_key] - before.get(count_key, 0)
            if total < -1e-9 or count < 0:
                raise ValueError(f"counter reset for {worker}: {name}")
            if count:
                rows.append({"metric": name.removesuffix("_sum"), "labels": dict(labels), "count": count, "sum": total, "mean": total / count})
        workers[worker] = rows
    if not workers:
        raise ValueError("no profile histogram snapshots")
    return {"official_result": False, "scope": "before AIPerf warmup through benchmark drain; excludes later generic probes", "limitations": ["native host timestamps, not isolated GPU kernel times", "stage intervals may overlap; do not add unrelated histograms", "means do not explain individual request tails", "native counters can include health requests during the capture"], "workers": workers}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.profile), indent=2) + "\n")
