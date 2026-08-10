from __future__ import annotations

from pathlib import Path

import pytest
from conftest import RUN_ID, make_manifest

from qwen_experiment_metrics.profile_sources import (
    capture_profile_hardware,
    validate_profile_hardware_capture,
)
from qwen_experiment_metrics.projections import project_profile_hardware


def test_profile_hardware_capture_is_closed_and_projectable(
    runtime: Path, tmp_path: Path
) -> None:
    manifest = make_manifest(runtime, "profile")
    proc = tmp_path / "proc"
    sys_nodes = tmp_path / "sys/devices/system/node"
    proc.mkdir(parents=True)
    (proc / "meminfo").write_text(
        "MemTotal: 100 kB\nMemAvailable: 75 kB\n", encoding="ascii"
    )
    for node, cpulist in ((0, "0-3\n"), (2, "8,10-11\n")):
        root = sys_nodes / f"node{node}"
        root.mkdir(parents=True)
        (root / "cpulist").write_text(cpulist, encoding="ascii")
        (root / "meminfo").write_text(
            f"Node {node} MemTotal: 50 kB\nNode {node} MemFree: 25 kB\n",
            encoding="ascii",
        )
    capture = capture_profile_hardware(
        manifest,
        runtime,
        node_rank=1,
        proc_root=proc,
        sys_node_root=sys_nodes,
        hostname="node-b",
    )
    value = validate_profile_hardware_capture(capture, runtime, RUN_ID, 1)
    assert [member["role"] for member in value["members"]] == [
        "proc_meminfo",
        "numa_cpulist",
        "numa_cpulist",
        "numa_meminfo",
        "numa_meminfo",
    ]
    rows = project_profile_hardware(runtime, RUN_ID, 1)
    columns = [dict(row.columns) for row in rows]
    assert [(row["record_kind"], row["numa_node_id"]) for row in columns] == [
        ("node_summary", None),
        ("numa_node", 0),
        ("numa_node", 2),
    ]
    assert columns[0]["postrun_memory_available_bytes"] == 75 * 1024


def test_profile_hardware_capture_rejects_bench(runtime: Path, tmp_path: Path) -> None:
    manifest = make_manifest(runtime, "bench")
    with pytest.raises(ValueError, match="profile-only"):
        capture_profile_hardware(
            manifest,
            runtime,
            node_rank=0,
            proc_root=tmp_path,
            sys_node_root=tmp_path,
        )
