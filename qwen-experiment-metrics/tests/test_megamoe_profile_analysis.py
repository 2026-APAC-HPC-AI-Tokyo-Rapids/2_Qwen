from __future__ import annotations

import hashlib
import runpy
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = runpy.run_path(str(ROOT / "research/firmus-vllm-megamoe/analyze_kernel_intervals.py"))


@pytest.mark.parametrize("intervals,union,span", [([(0, 10), (8, 12), (14, 18)], 16, 18), ([(5, 7), (0, 10), (0, 10)], 10, 10), ([(0, 2), (2, 4)], 4, 4)])
def test_interval_union_handles_overlaps_and_adjacency(intervals, union, span):
    result = ANALYSIS["interval_totals"](intervals)
    assert result["kernel_union_ns"] == union
    assert result["observed_span_ns"] == span
    assert result["between_kernel_gap_ns"] == span - union
    assert result["kernel_sum_ns"] == sum(end - start for start, end in intervals)


@pytest.mark.parametrize("intervals", [[], [(0, 0)], [(1, 0)]])
def test_interval_union_rejects_invalid_input(intervals):
    with pytest.raises(ValueError, match="positive kernel intervals"):
        ANALYSIS["interval_totals"](intervals)


def make_reports(directory):
    for worker in ANALYSIS["WORKERS"]:
        with sqlite3.connect(directory / f"nsys-{worker}.sqlite") as db:
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER, start INTEGER, end INTEGER, demangledName INTEGER)")
            db.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
            db.executemany("INSERT INTO StringIds VALUES (?, ?)", [(0, "sm90_fp8_mega_moe_impl<test>"), (1, "sm90_mega_moe_pre_dispatch_kernel<test>"), (2, "dense")])
            db.execute("CREATE TABLE TARGET_INFO_GPU (cuDevice INTEGER, uuid TEXT)")
            db.execute("CREATE TABLE DIAGNOSTIC_EVENT (text TEXT)")
            for device in range(4):
                db.execute("INSERT INTO TARGET_INFO_GPU VALUES (?, ?)", (device, f"{worker}-{device}"))
                db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)", [(device, 0, 10, 0), (device, 8, 12, 1), (device, 14, 18, 2)])


def test_analyze_all_workers_without_modifying_exports(tmp_path):
    make_reports(tmp_path)
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*.sqlite")}
    result = ANALYSIS["analyze"](tmp_path)
    assert len(result["workers"]) == 4
    for worker, summary in result["workers"].items():
        assert summary["sqlite_sha256"] == hashes[tmp_path / f"nsys-{worker}.sqlite"]
        assert len(summary["devices"]) == 4
        for device in summary["devices"]:
            assert device["kernel_union_ns"] == 16
            assert device["overlap_excess_ns"] == 2
            assert len(device["megamoe_specializations"]) == 2
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes}


@pytest.mark.parametrize("failure", ["incomplete", "duplicate_gpu", "missing_name", "missing_moe"])
def test_analyze_rejects_invalid_evidence(tmp_path, failure):
    make_reports(tmp_path)
    with sqlite3.connect(tmp_path / "nsys-prefill-1.sqlite") as db:
        if failure == "incomplete":
            db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES ('Not all CUDA events might have been collected')")
        elif failure == "duplicate_gpu":
            db.execute("UPDATE TARGET_INFO_GPU SET uuid = 'prefill-0-0' WHERE cuDevice = 0")
        elif failure == "missing_name":
            db.execute("DELETE FROM StringIds WHERE id = 0")
        else:
            db.execute("UPDATE StringIds SET value = 'ordinary_gemm' WHERE id = 0")
    with pytest.raises(ValueError):
        ANALYSIS["analyze"](tmp_path)
