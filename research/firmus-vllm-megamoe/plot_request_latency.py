"""Plot canonical normal-run metrics and complete measurement-phase request data.

Run with an analysis-only matplotlib environment; serving images stay unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

LABELS = {
    "38664373784": "Komori SGLang",
    "38821794697": "vLLM serial",
    "38850339264": "vLLM concurrent Push",
    "38872232338": "vLLM MP readiness",
}


def load_run(root: Path, run: str) -> dict:
    directory = root / run / "1/results/results"
    raw = directory / "profile_export.jsonl"
    rows = []
    with raw.open() as source:
        for line in source:
            row = json.loads(line)
            if row["metadata"]["benchmark_phase"] == "profiling":
                rows.append(row)
    export = json.loads((directory / "profile_export_aiperf.json").read_text())
    score = json.loads((directory / "score.json").read_text())
    if score["concurrency"] != 192 or score["sla"] != {"time_to_first_token_ms": 3000, "inter_token_latency_ms": 20} or score["error_request_count"] != 0 or score["osl_mismatch_count"] != 0:
        raise ValueError(f"unexpected scored workload/SLA/errors: {run}")
    if not rows or any(row["metadata"]["was_cancelled"] for row in rows):
        raise ValueError(f"empty or cancelled measurement requests: {run}")
    starts = np.array([row["metadata"]["request_start_ns"] for row in rows], dtype=np.int64)
    ttft = np.array([row["metrics"]["time_to_first_token"]["value"] for row in rows])
    itl = np.array([row["metrics"]["inter_token_latency"]["value"] for row in rows])
    for row in rows:
        for name in ("time_to_first_token", "inter_token_latency"):
            if row["metrics"][name]["unit"] != "ms":
                raise ValueError(f"unexpected latency unit: {run}/{name}")
    if not (np.isfinite(ttft).all() and np.isfinite(itl).all() and (ttft >= 0).all() and (itl >= 0).all()):
        raise ValueError(f"invalid request latency: {run}")
    good = (ttft <= 3000) & (itl <= 20)
    if len(rows) != score["request_count"] or len(rows) != export["request_count"]["avg"] or int(good.sum()) != export["good_request_count"]["avg"]:
        raise ValueError(f"raw request counts do not reproduce scored counts: {run}")
    fraction = float(good.mean())
    if abs(fraction - score["good_request_fraction"]) > 1e-12 or bool(score["eligible"]) != (fraction >= 0.9):
        raise ValueError(f"SLA accounting differs from score: {run}")
    digests = {}
    for path in (raw, directory / "profile_export_aiperf.json", directory / "score.json"):
        with path.open("rb") as source:
            digests[path.name] = hashlib.file_digest(source, "sha256").hexdigest()
    return {"run": run, "label": LABELS[run], "start_s": (starts - starts.min()) / 1e9, "ttft_ms": ttft, "itl_ms": itl, "good_fraction": fraction, "eligible": score["eligible"], "good_tps": score["goodput_requests_per_second"] * 200, "overall_tps": export["output_token_throughput"]["avg"], "source_directory": str(directory), "source_sha256": digests}


def plot(runs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    bar, ttft, bins, itl = axes.flat
    x = np.arange(len(runs))
    width = 0.36
    overall_bars = bar.bar(x - width / 2, [r["overall_tps"] for r in runs], width, color="#b5bdc4", label="Overall")
    good_bars = bar.bar(x + width / 2, [r["good_tps"] for r in runs], width, color="#2878b5", label="SLA passing")
    bar.axhline(7000, color="#a33", ls="--", lw=1, label="7k target")
    bar.set_xticks(x, [f'{r["label"].replace(" ", chr(10), 1)}\n{r["good_fraction"]:.2%}' + (" eligible" if r["eligible"] else " ineligible") for r in runs], fontsize=8)
    bar.set(title="Official output throughput", ylabel="Tokens / s")
    bar.set_ylim(0, max(8000, max(r["overall_tps"] for r in runs) * 1.15))
    bar.bar_label(overall_bars, fmt="{:,.0f}", fontsize=8, padding=3)
    bar.bar_label(good_bars, fmt="{:,.0f}", fontsize=8, padding=3)
    bar.legend(fontsize=8, loc="upper left", ncol=3)
    end = max(120, np.ceil(max(float(r["start_s"].max()) for r in runs) / 5) * 5)
    edges = np.arange(0, end + 5, 5)
    for run in runs:
        label = f'{run["label"]} ({run["run"]})'
        for ax, key, scale in ((ttft, "ttft_ms", 1000), (itl, "itl_ms", 1)):
            values = np.sort(run[key]) / scale
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=label)
        counts, _ = np.histogram(run["start_s"], edges)
        misses, _ = np.histogram(run["start_s"][run["ttft_ms"] > 3000], edges)
        rates = np.divide(misses, counts, out=np.full(len(counts), np.nan), where=counts > 0)
        bins.stairs(rates, edges, label=label)
    for ax in (ttft, itl, bins):
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=0.2)
    ttft.axvline(3, color="#a33", ls="--", lw=1)
    itl.axvline(20, color="#a33", ls="--", lw=1)
    ttft.set(title="TTFT: all measurement requests", xlabel="TTFT (s)", ylabel="Cumulative request fraction")
    itl.set(title="ITL: all measurement requests", xlabel="Request mean ITL (ms)", ylabel="Cumulative request fraction")
    bins.set(title="TTFT misses by request-start time (5 s bins)", xlabel="Seconds since first measurement request", ylabel="Requests with TTFT > 3 s / bin requests", xlim=(0, end))
    ttft.legend(fontsize=7, loc="lower right")
    fig.suptitle("Matched official workload: throughput and request latency", fontsize=14)
    note = "c192; ISL4000 / OSL200; 60 s warmup / 120 s measurement / 30 s grace. SLA: TTFT <= 3 s and request mean ITL <= 20 ms.\nSingle observations, not isolated/repeated causal speedup estimates. Full measurement is scored; time bins are descriptive only.\nSLA-passing TPS = official goodput requests/s x 200; ineligibility does not zero measured TPS; the local gated score is separate."
    fig.text(0.02, 0.015, note, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    provenance = json.dumps({"matplotlib": matplotlib.__version__, "numpy": np.__version__, "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "sources": [{k: r[k] for k in ("run", "source_directory", "source_sha256")} for r in runs]})
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, metadata={"Description": provenance})
    fig.savefig(output.with_suffix(".pdf"), metadata={"Subject": provenance})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", choices=LABELS, help="audited canonical normal-run IDs")
    parser.add_argument("--archive-root", type=Path, default=Path("/srv/qwen-artifacts"))
    parser.add_argument("--output", type=Path, required=True, help="output prefix for PNG and PDF")
    args = parser.parse_args()
    plot([load_run(args.archive_root, run) for run in args.runs], args.output)
