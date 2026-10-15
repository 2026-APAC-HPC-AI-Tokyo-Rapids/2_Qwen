"""Summarize four exported MegaMoE Nsight reports without changing raw artifacts.

Export native reports with the pinned collector's nsys, then pass their directory.
Reject incomplete captures using the same gate as the scheduler workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATE = runpy.run_path(str(ROOT / "scripts/validate-firmus-profile.py"))["validate_capture"]
WORKERS = ("prefill-0", "prefill-1", "prefill-2", "decode-0")


def interval_totals(intervals: list[tuple[int, int]]) -> dict:
    """Union accounts for overlapping streams; sums count their work separately."""
    if not intervals or any(end <= start for start, end in intervals):
        raise ValueError("expected nonempty positive kernel intervals")
    ordered = sorted(intervals)
    first = end = ordered[0][0]
    union = 0
    gaps = []
    for left, right in ordered:
        if left > end:
            gaps.append(left - end)
        union += max(0, right - max(left, end))
        end = max(end, right)
    span = end - first
    total = sum(right - left for left, right in ordered)
    return {
        "observed_span_ns": span,
        "kernel_union_ns": union,
        "kernel_sum_ns": total,
        "overlap_excess_ns": total - union,
        "between_kernel_gap_ns": span - union,
        "largest_between_kernel_gap_ns": max(gaps, default=0),
        "observed_kernel_busy_fraction": union / span,
    }


def analyze(directory: Path) -> dict:
    # Validate every report before producing any timing summary.
    databases = {worker: directory / f"nsys-{worker}.sqlite" for worker in WORKERS}
    validation = {worker: VALIDATE(path, 4) for worker, path in databases.items()}
    workers = {}
    all_uuids = set()
    for worker, path in databases.items():
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True) as db:
            identity = dict(db.execute("SELECT cuDevice, uuid FROM TARGET_INFO_GPU"))
            rows = db.execute("SELECT k.deviceId, k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id ORDER BY k.deviceId, k.start").fetchall()
        if len(rows) != validation[worker]["kernel_count"]:
            raise ValueError(f"missing kernel name metadata: {worker}")
        devices = defaultdict(list)
        for device, start, end, name in rows:
            devices[device].append((start, end, name))
        summaries = []
        for device, events in sorted(devices.items()):
            uuid = identity[device]
            if not uuid or uuid in all_uuids:
                raise ValueError(f"missing or repeated GPU UUID: {worker}/{device}")
            all_uuids.add(uuid)
            by_name = defaultdict(list)
            for start, end, name in events:
                by_name[name].append((start, end))
            ranked = sorted(by_name, key=lambda name: sum(end - start for start, end in by_name[name]), reverse=True)
            selected = [name for name in ranked if "sm90_fp8_mega_moe_impl" in name or "sm90_mega_moe_pre_dispatch_kernel" in name]
            if not any("sm90_fp8_mega_moe_impl" in name for name in selected) or not any("sm90_mega_moe_pre_dispatch_kernel" in name for name in selected):
                raise ValueError(f"missing MegaMoE main/pre-dispatch kernel: {worker}/{device}")
            summaries.append({
                "device": device,
                "uuid": uuid,
                "kernels": len(events),
                **interval_totals([(start, end) for start, end, _ in events]),
                "top_kernel_names_by_duration_sum": [{"name": name, "count": len(by_name[name]), "sum_ns": sum(end - start for start, end in by_name[name])} for name in ranked[:10]],
                "megamoe_specializations": [{"name": name, "count": len(by_name[name]), "sum_ns": sum(end - start for start, end in by_name[name])} for name in selected],
            })
        workers[worker] = {"sqlite_sha256": digest, "validation": validation[worker], "devices": summaries}
    return {
        "workers": workers,
        "limitations": [
            "Instrumented kernel timings are not official throughput or an end-to-end critical path.",
            "Each GPU uses its first-to-last recorded kernel span, excluding capture-edge idle time.",
            "GPU spans are local to each report and are not aligned across hosts.",
            "Kernel sums can overlap; duration shares are not recoverable speedup estimates.",
            "A gap in kernel activity can include copies or communication; it is not necessarily an idle GPU.",
            "Passing the existing warning/coverage gate does not prove complete event dependencies.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(analyze(parser.parse_args().directory), indent=2))
