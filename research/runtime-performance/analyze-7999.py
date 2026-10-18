"""Recompute run 39114949128 from its archived normal pass; no remote execution."""

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
import statistics as st
from pathlib import Path


def stats(values):
    x = sorted(values)
    if not x:
        return {"n": 0}

    def q(p):
        i = (len(x) - 1) * p
        lo = int(i)
        return x[lo] + (x[min(lo + 1, len(x) - 1)] - x[lo]) * (i - lo)

    return {
        "n": len(x),
        "mean": st.mean(x),
        "min": x[0],
        "p50": q(0.5),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": x[-1],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    root = args.archive
    verified = 0
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = root / name.lstrip("*")
        assert hashlib.file_digest(path.open("rb"), "sha256").hexdigest() == expected, (
            path
        )
        verified += 1
    normal = root / "results/results"
    summary = json.loads((normal / "profile_export_aiperf.json").read_text())
    score = json.loads((normal / "score.json").read_text())
    rows = []
    for line in (normal / "profile_export.jsonl").open():
        x = json.loads(line)
        if x["metadata"]["benchmark_phase"] == "profiling":
            m = x["metadata"]
            v = {k: val["value"] for k, val in x["metrics"].items()}
            rows.append(
                dict(
                    start=m["request_start_ns"] / 1e9,
                    ack=m["request_ack_ns"] / 1e9,
                    end=m["request_end_ns"] / 1e9,
                    credit=m["credit_issued_ns"] / 1e9,
                    **v,
                )
            )
    rows.sort(key=lambda x: x["start"])
    t0 = rows[0]["start"]
    # AIPerf goodput uses the credit-phase start/end, not benchmark_duration
    # (first request start to last request end). Their difference is 2.683 ms.
    duration = summary["good_request_count"]["avg"] / summary["goodput"]["avg"]
    good = [
        x
        for x in rows
        if x["time_to_first_token"] <= 3000 and x["inter_token_latency"] <= 20
    ]
    assert len(rows) == score["request_count"] == 5051
    assert len(good) == summary["good_request_count"]["avg"]
    assert (
        abs(len(good) * 200 / duration - score["score_output_tokens_per_second"]) < 1e-7
    )
    metric_names = [
        "time_to_first_token",
        "inter_token_latency",
        "request_latency",
        "http_req_connecting",
        "http_req_sending",
        "http_req_connection_overhead",
    ]

    def summarize(sub):
        return dict(
            requests=len(sub),
            ttft_fail=sum(x["time_to_first_token"] > 3000 for x in sub),
            itl_fail=sum(x["inter_token_latency"] > 20 for x in sub),
            **{k: stats([x[k] for x in sub]) for k in metric_names},
        )

    out = {
        "run": 39114949128,
        "verified_archive_files": verified,
        "duration_s": duration,
        "start_utc": dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat(),
        "score": score,
        "all": summarize(rows),
        "after20": summarize([x for x in rows if x["start"] - t0 >= 20]),
        "cohorts": {
            f"{lo}-{hi}": summarize([x for x in rows if lo <= x["start"] - t0 < hi])
            for lo, hi in [(0, 1), (1, 5), (5, 20), (20, 60), (60, 115), (115, 120)]
        },
    }
    out["failure_start_s"] = stats(
        [x["start"] - t0 for x in rows if x["time_to_first_token"] > 3000]
    )
    out["credit_to_start_ms"] = stats([(x["start"] - x["credit"]) * 1000 for x in rows])
    out["client_osl_counts"] = dict(
        collections.Counter(x["output_sequence_length"] for x in rows)
    )
    lo, hi = t0 + 20, t0 + 115
    out["steady"] = {
        "window_s": 95,
        "starts_per_s": sum(lo <= x["start"] < hi for x in rows) / 95,
        "acks_per_s": sum(lo <= x["ack"] < hi for x in rows) / 95,
        "completions_per_s": sum(lo <= x["end"] < hi for x in rows) / 95,
    }
    for label, left, right in [
        ("client_active", "start", "end"),
        ("client_before_first_token", "start", "ack"),
        ("client_after_first_token", "ack", "end"),
    ]:
        out["steady"][label] = (
            sum(max(0, min(hi, x[right]) - max(lo, x[left])) for x in rows) / 95
        )
    out["ceilings"] = {
        "all_sla_pass_same_duration": len(rows) * 200 / duration,
        "gain_needed_8500": 8500 / score["score_output_tokens_per_second"] - 1,
        "additional_good_requests_same_duration": 8500 * duration / 200 - len(good),
        "required_raw_rps_same_good_fraction": 42.5 / score["good_request_fraction"],
    }
    # Server logs are UTC+08 (confirmed against the archived nsys session epoch).
    tz = dt.timezone(dt.timedelta(hours=8))
    logs = {}
    for path in sorted((root / "logs/logs").glob("*-0.log")) + sorted(
        (root / "logs/logs").glob("prefill-[12].log")
    ):
        if not path.name.startswith(("decode", "prefill")):
            continue
        by_rank = collections.defaultdict(list)
        for line in path.read_text(errors="replace").splitlines():
            match = re.search(
                r"\[(2026-\d\d-\d\d \d\d:\d\d:\d\d) DP(\d).*?\] (Decode|Prefill) batch, (.*)",
                line,
            )
            if not match:
                continue
            when = (
                dt.datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=tz)
                .timestamp()
            )
            if not lo <= when < hi:
                continue
            fields = {
                k: float(v)
                for k, v in re.findall(r"(?:^|, )([^:,]+): ([\d.]+)", match[4])
            }
            by_rank[int(match[2])].append((when, fields))
        ranks = {}
        for rank, entries in sorted(by_rank.items()):
            keys = sorted({k for _, fields in entries for k in fields})
            result = {
                "log_rows": len(entries),
                "first": entries[0][0] - t0,
                "last": entries[-1][0] - t0,
                "fields": {
                    k: stats([f[k] for _, f in entries if k in f]) for k in keys
                },
            }
            if path.name.startswith("decode"):
                result["cycle_ms_from_40_iteration_log_cadence"] = (
                    (entries[-1][0] - entries[0][0]) * 1000 / (40 * (len(entries) - 1))
                )
                result["at_cap_fraction"] = st.mean(
                    f["#running-req"] == 32 for _, f in entries
                )
            else:
                result["new_seq_per_s"] = sum(f["#new-seq"] for _, f in entries) / 95
                result["new_token_counts"] = dict(
                    collections.Counter(f["#new-token"] for _, f in entries)
                )
            ranks[rank] = result
        logs[path.stem] = ranks
    out["normal_server_logs"] = logs
    # Kernel totals are attribution only: they sum multiple devices and overlapping streams.
    kernels = json.loads((root / "profile/profile/kernel-summary.json").read_text())
    out["profile_kernel_summary"] = {
        worker: data["top_kernels"][:12] for worker, data in kernels.items()
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: out[k]
                for k in [
                    "verified_archive_files",
                    "duration_s",
                    "start_utc",
                    "steady",
                    "ceilings",
                    "failure_start_s",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    assert stats([1, 2, 3])["p50"] == 2
    main()
