"""Replot archived 40071892077 Nsight data without a new GPU measurement.

Usage: uv run --with matplotlib --with numpy python THIS.py ARCHIVE OUTPUT
ARCHIVE contains the three SQLite exports and existing launch-correlated analysis.json.
Metric stream IDs are intentionally not mapped to CUDA device IDs by bit patterns.
"""

import argparse
import collections
import hashlib
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(a.mean()), "p50": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def category(name):
    if "mega_moe_impl" in name:
        return "MegaMoE"
    if "FlashAttn" in name or "flash_fwd" in name or "combine_fixed" in name:
        return "Attention"
    if "sm90_fp8_gemm_1d2d" in name:
        return "Dense GEMM"
    return "Other kernels"


def union_ns(intervals):
    total = 0
    end = -1
    for a, b in sorted(intervals):
        total += max(0, b - max(end, a))
        end = max(end, b)
    return total


def membership(ts, intervals):
    starts = np.asarray([a for a, b in intervals])
    ends = np.asarray([b for a, b in intervals])
    indices = np.searchsorted(starts, ts, side="right") - 1
    return (indices >= 0) & (ts < ends[np.maximum(indices, 0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    analysis_path = args.archive / "analysis.json"
    analysis = json.loads(analysis_path.read_text())["profile"]
    result = {
        "run": "40071892077", "hardware": "H200", "new_gpu_measurements": 0,
        "analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "scope": "Interior complete target EXTEND stages from existing CPU launch correlation; first/last two complete stages removed per worker.",
        "limitations": [
            "1 ms sampled throughput percentages, not instantaneous values or exact kernel counters.",
            "No explicit metric typeId to CUDA device mapping. Four streams shown separately with shared capture timestamps.",
            "NVLink request/response submetrics are not summed or converted to bytes/s or total link utilization.",
            "Sample timestamp inside MegaMoE does not imply its complete sampling interval lies inside it.",
            "Fused MegaMoE has no FC1/FC2/communication markers in this capture.",
            "Low average throughput cannot rule out short bursts, latency, dependency or synchronization stalls.",
            "This is the older 8854.80 serving configuration, not the latest 9235 configuration."
        ],
        "captures": [],
    }
    representative = None
    for worker in range(3):
        name = f"nsys-prefill-{worker}"
        path = args.archive / f"{name}.sqlite"
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        c.execute("pragma query_only=1")
        meta = c.execute("select name,value from META_DATA_CAPTURE where value like 'GITHUB_RUN_ID=%' or name='GPU_METRICS_OPTIONS:SAMPLING_FREQUENCY'").fetchall()
        assert any("40071892077" in v for k, v in meta)
        assert ("GPU_METRICS_OPTIONS:SAMPLING_FREQUENCY", "1000") in meta
        rows = c.execute("select k.start,k.end,k.deviceId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start").fetchall()
        by_dev = {d: [r for r in rows if r[2] == d] for d in range(4)}
        targets = [d["stages"]["prefill_target"] for d in analysis[name]["devices"]]
        # Corresponding stages across ranks must share the same batch, not merely ordinal counts.
        assert len({len(s) for s in targets}) == 1
        intervals, moe_common, cycles, layer_rows = [], [], [], []
        for idx in range(2, len(targets[0]) - 2):
            starts = [s[idx]["start_ns"] for s in targets]
            ends = [s[idx]["end_ns"] for s in targets]
            assert max(starts) < min(ends)
            lo, hi = min(starts), max(ends)
            intervals.append((lo, hi))
            ks = [[r for r in by_dev[d] if starts[d] <= r[0] and r[1] <= ends[d]] for d in range(4)]
            moes = [[r for r in k if category(r[3]) == "MegaMoE"] for k in ks]
            assert [len(m) for m in moes] == [94] * 4
            layers = []
            for layer in range(94):
                a = [m[layer][0] for m in moes]
                b = [m[layer][1] for m in moes]
                assert max(a) < min(b)
                moe_common.append((max(a), min(b)))
                row = {"stage_index": idx, "layer": layer, "starts_ns": a, "ends_ns": b,
                       "entry_spread_us": (max(a) - min(a)) / 1e3,
                       "exit_spread_us": (max(b) - min(b)) / 1e3,
                       "last_entry_to_last_exit_us": (max(b) - max(a)) / 1e3}
                layers.append(row)
            layer_rows.extend(layers)
            cats = collections.defaultdict(list)
            for k in ks:
                for cat in ["MegaMoE", "Attention", "Dense GEMM", "Other kernels"]:
                    cats[cat].append(sum(b-a for a,b,d,n in k if category(n) == cat) / 1e6)
            cyc = {"stage_index": idx, "start_ns": lo, "end_ns": hi,
                   "target_span_ms": (hi-lo) / 1e6,
                   "period_ms": (min(s[idx+1]["start_ns"] for s in targets)-lo) / 1e6,
                   "kernel_ms_mean_across_ranks": {k: float(np.mean(v)) for k,v in cats.items()},
                   "idle_ms_by_rank": [(ends[d]-starts[d]-union_ns([(a,b) for a,b,_,_ in ks[d]]))/1e6 for d in range(4)],
                   "entry_spread_sum_ms": sum(r["entry_spread_us"] for r in layers) / 1000}
            cycles.append(cyc)
            if worker == 0 and idx == len(targets[0]) // 2:
                representative = {"lo": lo, "hi": min(s[idx+1]["start_ns"] for s in targets), "target_end": hi,
                                  "rows": rows, "layers": layers, "cycle": cyc, "worker": worker}
        definitions = dict(c.execute("select distinct metricId,metricName from TARGET_INFO_GPU_METRICS"))
        for metric, expected in {4: "SM Issue", 5: "Tensor Active", 18: "DRAM Read Bandwidth", 19: "DRAM Write Bandwidth", 25: "NVLink TX Requests User Data", 27: "NVLink TX Responses User Data"}.items():
            assert definitions[metric] == f"{expected} [Throughput %]", (name, metric, definitions[metric])
        metric_rows = c.execute("select timestamp,typeId,metricId,value from GPU_METRICS where metricId in (0,3,4,5,18,19,21,23,25,27) order by timestamp").fetchall()
        streams = sorted({r[1] for r in metric_rows})
        assert len(streams) == 4
        arrays = {}
        metrics = {}
        for metric in [0,3,4,5,18,19,21,23,25,27]:
            entries = []
            for stream in streams:
                arr = np.asarray([(t,v) for t,s,m,v in metric_rows if s == stream and m == metric])
                arrays[stream, metric] = arr
                t, v = arr[:,0], arr[:,1]
                entries.append({"type_id": stream, "target": stats(v[membership(t, intervals)]),
                                "all_four_in_megamoe": stats(v[membership(t, moe_common)]),
                                "median_sample_gap_ns": float(np.median(np.diff(t)))})
            metrics[str(metric)] = {"name": definitions[metric], "streams": entries}
        # Co-occurrence is descriptive only. No sub-ms interpolation or phase attribution.
        joint = []
        for stream in streams:
            base = arrays[stream,5]
            assert all(np.array_equal(base[:,0], arrays[stream,m][:,0]) for m in [25,18,19])
            mask = membership(base[:,0], moe_common)
            tensor, tx, rd, wr = (arrays[stream,m][mask,1] for m in [5,25,18,19])
            low = tensor < 30
            joint.append({"type_id": stream, "samples": int(mask.sum()), "tensor_below_30_samples": int(low.sum()),
                          "tx_request_at_low_tensor": stats(tx[low]) if low.any() else None,
                          "read_at_low_tensor": stats(rd[low]) if low.any() else None,
                          "write_at_low_tensor": stats(wr[low]) if low.any() else None,
                          "tensor_tx_request_pearson": float(np.corrcoef(tensor, tx)[0,1])})
        record = {"capture": name, "sqlite_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "metadata": meta, "complete_target_count": len(cycles), "cycles": cycles,
                  "layer_entry_spread_us": stats([r["entry_spread_us"] for r in layer_rows]),
                  "layer_exit_spread_us": stats([r["exit_spread_us"] for r in layer_rows]),
                  "layer_last_entry_to_last_exit_us": stats([r["last_entry_to_last_exit_us"] for r in layer_rows]),
                  "metrics": metrics, "joint_samples": joint}
        result["captures"].append(record)
        if worker == 0:
            representative["arrays"] = arrays
            representative["streams"] = streams
        c.close()
    result["representative_cycle"] = representative["cycle"]
    plot(representative, args.output, zoom=False)
    plot(representative, args.output, zoom=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    compact = [{"capture": r["capture"], "targets": r["complete_target_count"],
                "target_ms": stats([v["target_span_ms"] for v in r["cycles"]]),
                "period_ms": stats([v["period_ms"] for v in r["cycles"]]),
                "entry_spread_us": r["layer_entry_spread_us"], "joint_samples": r["joint_samples"]} for r in result["captures"]]
    print(json.dumps(compact, indent=2))


def plot(rep, output, zoom):
    origin = rep["lo"]
    a, b = rep["lo"], rep["hi"]
    if zoom:
        a = min(rep["layers"][46]["starts_ns"]) - 1_000_000
        b = max(rep["layers"][49]["ends_ns"]) + 200_000
    fig, axes = plt.subplots(4, 1, figsize=(13,9), sharex=True,
                             gridspec_kw={"height_ratios": [1.2,1,1,1.15]})
    metric_groups = [[(5,"Tensor active","#1764a0"),(4,"SM issue","#de7d22")],
                     [(18,"HBM read","#a02972"),(19,"HBM write","#297942")],
                     [(25,"NVLink TX request user data","#d04f35"),(27,"NVLink TX response user data","#6853a3")]]
    for ax, group in zip(axes[:3], metric_groups):
        for metric, label, color in group:
            for i, stream in enumerate(rep["streams"]):
                ar = rep["arrays"][stream,metric]
                selected = ar[(ar[:,0] >= a) & (ar[:,0] <= b)]
                ax.plot((selected[:,0]-origin)/1e6, selected[:,1], color=color, alpha=.60,
                        marker="o" if zoom else None, ms=3, lw=1, label=label if i==0 else None)
        ax.set_ylim(0,100)
        ax.set_ylabel("Stored throughput %")
        ax.legend(loc="upper right", ncol=2, fontsize=9, framealpha=.95)
        ax.grid(alpha=.2)
        if not zoom:
            ax.axvspan((rep["target_end"]-origin)/1e6, (b-origin)/1e6, color="#b4b4b4", alpha=.18)
    colors = {"MegaMoE":"#245b80", "Attention":"#32a3a4", "Dense GEMM":"#e1a33b", "Other kernels":"#b6bfc7"}
    ax = axes[3]
    for dev in range(4):
        for cat,color in colors.items():
            spans = [((max(t,a)-origin)/1e6,(min(e,b)-max(t,a))/1e6) for t,e,d,n in rep["rows"] if d==dev and t<b and e>a and category(n)==cat]
            ax.broken_barh(spans, (dev-.32,.64), facecolors=color)
    ax.set_yticks(range(4), [f"CUDA GPU {i}" for i in range(4)])
    ax.set_ylim(3.6,-.9)
    ax.set_xlabel("Time since this target Prefill began (ms)")
    ax.set_xlim((a-origin)/1e6,(b-origin)/1e6)
    ax.legend(handles=[Patch(facecolor=c,label=n) for n,c in colors.items()],loc="upper center", bbox_to_anchor=(.5,-.28),ncol=4,frameon=False)
    if zoom:
        for row in rep["layers"][46:50]:
            center=(min(row["starts_ns"])+max(row["ends_ns"]))/2
            ax.text((center-origin)/1e6,-.43,f"MoE layer {row['layer']}",ha="center",va="bottom",fontsize=9)
    fig.suptitle("Run 40071892077 | H200 Prefill | " + ("4-layer zoom" if zoom else "One complete Prefill cycle"),fontsize=16,weight="bold",y=.98)
    fig.text(.09,.93,"Four metric streams per color, at original ~1 ms sampling. Shared capture clock; no guessed GPU mapping.",fontsize=10)
    fig.text(.09,.035,"NVLink lines are request/response user-data submetrics, not total link saturation. No FC1/FC2 markers in this capture.",fontsize=10)
    fig.subplots_adjust(top=.89,bottom=.15,hspace=.15,left=.09,right=.98)
    stem = "prefill-zoom" if zoom else "prefill-cycle"
    fig.savefig(output / f"{stem}.png",dpi=170)
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
