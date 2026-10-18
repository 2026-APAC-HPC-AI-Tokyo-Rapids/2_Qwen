"""Attribute the archived profile on a CPU host with nsys-exported SQLite files."""

import argparse
import bisect
import collections
import datetime as dt
import gzip
import itertools
import json
import re
import sqlite3
import statistics as st
from pathlib import Path


def union(intervals):
    total, end = 0, -1
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def stats(x):
    x = sorted(x)
    return (
        {
            "n": len(x),
            "mean": st.mean(x),
            "p50": st.median(x),
            "p95": x[int(0.95 * (len(x) - 1))],
            "min": x[0],
            "max": x[-1],
        }
        if x
        else {}
    )


def category(name):
    for needle, label in [
        ("sm90_fp8_mega_moe_impl<", "megamoe"),
        ("sm90_mega_moe_pre_dispatch", "pre_dispatch"),
        ("FlashAttnFwd", "attention"),
        ("sm90_fp8_gemm_1d2d_impl<", "dense_gemm"),
        ("qknorm_rope", "qknorm_rope"),
        ("fused_qk", "qknorm_rope"),
        ("fused_rope", "rope"),
        ("per_token_group_quant", "group_quant"),
        ("RMSNorm", "norm"),
        ("rmsnorm", "norm"),
        ("topk", "topk"),
        ("_router_triton", "topk"),
        ("store_kv", "kv_store"),
    ]:
        if needle.lower() in name.lower():
            return label
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("databases", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    times = {}
    for line in (
        (args.archive / "logs/logs/profile-capture.log").read_text().splitlines()
    ):
        for key in ["capture_active", "stop_requested"]:
            if line.startswith(key + " "):
                times[key] = int(
                    dt.datetime.fromisoformat(
                        line.split("utc=")[1].rstrip("Z")[:26] + "+00:00"
                    ).timestamp()
                    * 1e9
                )
    out = {
        "scope": "Separate instrumented five-second profile; complete steps only, no speedup inferred from normal-vs-profile timing.",
        "workers": {},
    }
    for worker in ["decode-0", "prefill-0"]:
        c = sqlite3.connect(
            f"file:{args.databases / (worker + '.sqlite')}?mode=ro", uri=True
        )
        epoch = c.execute(
            "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
        ).fetchone()[0]
        lo, hi = times["capture_active"] - epoch, times["stop_requested"] - epoch
        names = dict(c.execute("SELECT id,value FROM StringIds"))
        apis = {
            ((tid >> 24) & 0xFFFFFF, corr): (a, b, names[nid])
            for a, b, tid, corr, nid in c.execute(
                "SELECT start,end,globalTid,correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
            )
        }
        ranks = {}
        for trace in sorted(
            (args.archive / "profile/profile/torch" / worker).glob("*.gz")
        ):
            with gzip.open(trace, "rt") as stream:
                data = json.load(stream)
            base = data["baseTimeNanoseconds"] - epoch
            rank = int(re.search(r"-DP-(\d+)-", trace.name)[1])
            annotations = [
                e
                for e in data["traceEvents"]
                if e.get("cat") == "user_annotation" and e.get("dur")
            ]
            stages = []
            for e in annotations:
                if e["name"] in ("draft", "draft_extend") or e["name"].startswith(
                    ("step[TARGET_VERIFY", "step[EXTEND", "step[IDLE")
                ):
                    a = base + round(e["ts"] * 1000)
                    kind = e["name"].split(" ")[0].split("[")[-1].rstrip("]")
                    stages.append((a, a + round(e["dur"] * 1000), kind, e["name"]))
            stages.sort()
            starts = [s[0] for s in stages]
            groups = collections.defaultdict(list)
            kernels = []
            for a, b, nid, pid, corr in c.execute(
                "SELECT start,end,demangledName,globalPid,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=? ORDER BY start",
                (rank,),
            ):
                kernels.append((a, b))
                api = apis.get(((pid >> 24) & 0xFFFFFF, corr))
                if api is None:
                    continue
                i = bisect.bisect_right(starts, api[0]) - 1
                if i >= 0 and api[0] < stages[i][1]:
                    groups[i].append((a, b, names[nid]))
            rows = collections.defaultdict(list)
            cycles = []
            incomplete = collections.Counter()
            for i, ks in groups.items():
                a, b = min(k[0] for k in ks), max(k[1] for k in ks)
                if not lo <= a <= b <= hi:
                    continue
                kind = stages[i][2]
                cats = collections.Counter()
                for x, y, name in ks:
                    cats[category(name)] += (y - x) / 1e6
                row = {
                    "span_ms": (b - a) / 1e6,
                    "busy_ms": union((x, y) for x, y, _ in ks) / 1e6,
                    "categories_ms": dict(cats),
                    "kernels": len(ks),
                }
                if kind in ("TARGET_VERIFY", "EXTEND"):
                    moe = [k for k in ks if category(k[2]) == "megamoe"]
                    if kind == "EXTEND" and not moe:
                        # The dense one-layer EAGLE draft uses the same EXTEND label.
                        kind = "DRAFT_EXTEND"
                    elif len(moe) != 94:
                        incomplete[f"{kind}:{len(moe)}_layers"] += 1
                        continue
                    else:
                        row["batch"] = int(re.search(r"bs=(\d+)", stages[i][3])[1])
                        row["moe_layers_ms"] = [(y - x) / 1e6 for x, y, _ in moe]
                rows[kind].append(row)
                if kind == ("draft" if worker.startswith("decode") else "EXTEND"):
                    cycles.append(a)
            cycle_rows = []
            cycles.sort()
            for a, b in itertools.pairwise(cycles):
                busy = union(
                    (max(a, x), min(b, y)) for x, y in kernels if x < b and y > a
                )
                cycle_rows.append(((b - a) / 1e6, busy / 1e6))
            stage_stats = {}
            for kind, rs in rows.items():
                cats = sorted({k for r in rs for k in r["categories_ms"]})
                s = {
                    k: stats([r[k] for r in rs])
                    for k in ["span_ms", "busy_ms", "kernels"]
                }
                s["categories_ms"] = {
                    k: stats([r["categories_ms"].get(k, 0) for r in rs]) for k in cats
                }
                if kind in ("TARGET_VERIFY", "EXTEND"):
                    s["batch"] = stats([r["batch"] for r in rs])
                    s["moe_layers_ms"] = [
                        st.mean(r["moe_layers_ms"][i] for r in rs) for i in range(94)
                    ]
                    s["by_batch"] = {
                        bs: {
                            "n": len(sub),
                            "span_ms": st.mean(r["span_ms"] for r in sub),
                            "megamoe_ms": st.mean(
                                r["categories_ms"]["megamoe"] for r in sub
                            ),
                        }
                        for bs in sorted({r["batch"] for r in rs})
                        for sub in [[r for r in rs if r["batch"] == bs]]
                    }
                stage_stats[kind] = s
            busy = union(
                (max(lo, a), min(hi, b)) for a, b in kernels if a < hi and b > lo
            )
            assert rows["TARGET_VERIFY" if worker.startswith("decode") else "EXTEND"], (
                worker,
                rank,
            )
            ranks[rank] = {
                "window_busy_fraction": busy / (hi - lo),
                "incomplete_stages": dict(incomplete),
                "stages": stage_stats,
                "cycle_ms": stats([a for a, b in cycle_rows]),
                "cycle_idle_ms": stats([a - b for a, b in cycle_rows]),
            }
        out["workers"][worker] = {
            "window_ns": [lo, hi],
            "epoch_ns": epoch,
            "ranks": ranks,
            # Device-wide samples, not per-kernel counters. Exclude the clock
            # fields: their exported values do not match the MHz label.
            "gpu_throughput_metrics_all_four_devices": {
                name: {
                    "samples": count,
                    "mean_percent": mean,
                    "min_percent": minimum,
                    "max_percent": maximum,
                }
                for name, count, mean, minimum, maximum in c.execute(
                    "SELECT n.metricName, count(*), avg(g.value), min(g.value), max(g.value) "
                    "FROM GPU_METRICS g JOIN TARGET_INFO_GPU_METRICS n "
                    "ON n.typeId=g.typeId AND n.metricId=g.metricId "
                    "WHERE timestamp >= ? AND timestamp < ? AND n.metricName LIKE '%[Throughput %]' "
                    "GROUP BY n.metricName",
                    (lo, hi),
                )
            },
        }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for w, d in out["workers"].items():
        for r, s in d["ranks"].items():
            print(
                w,
                r,
                "busy",
                s["window_busy_fraction"],
                "cycle",
                s["cycle_ms"],
                "idle",
                s["cycle_idle_ms"],
            )


if __name__ == "__main__":
    assert union([(0, 2), (1, 3), (5, 6)]) == 4
    main()
