from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import RUN_ID, make_manifest

from qwen_experiment_metrics.canonical import jcs_bytes
from qwen_experiment_metrics.profile_contract import parse_phase_summary
from qwen_experiment_metrics.system_sources import (
    capture_environment_snapshot,
    create_scheduler_accounting,
    environment_facts,
    finish_run_lifecycle,
    start_run_lifecycle,
    validate_environment_snapshot,
    validate_lifecycle_summary,
    validate_node_lifecycle,
    validate_scheduler_accounting,
)
from qwen_experiment_metrics.workload import validate_ordered_workload


def test_environment_snapshot_is_byte_sorted_nul_pairs(tmp_path: Path) -> None:
    path = tmp_path / "environment.snapshot"
    capture_environment_snapshot(path, {b"Z": b"last", b"A": b"first"})
    assert path.read_bytes() == b"A\0first\0Z\0last\0"
    assert validate_environment_snapshot(path) == (
        (b"A", b"first"),
        (b"Z", b"last"),
    )
    assert [(fact.key, fact.logical_value) for fact in environment_facts(
        path, key_prefix="environment.job_entry"
    )] == [
        ("environment.job_entry.A", "Zmlyc3Q="),
        ("environment.job_entry.Z", "bGFzdA=="),
    ]


def test_run_lifecycle_uses_one_monotonic_clock(
    manifest_path: Path, runtime: Path
) -> None:
    start_run_lifecycle(manifest_path, runtime)
    path, facts = finish_run_lifecycle(manifest_path, runtime, 17)
    assert path == runtime / "source/run-lifecycle.json"
    values = {fact.key: fact.logical_value for fact in facts}
    assert values["run.exit_code"] == 17
    assert values["run.elapsed_sec"] >= 0


def test_scheduler_accounting_binds_exact_tsv_and_job(
    manifest_path: Path, runtime: Path
) -> None:
    scheduler = runtime / "source/scheduler"
    scheduler.mkdir(parents=True)
    stdout = scheduler / "accounting.txt"
    stderr = scheduler / "accounting.stderr"
    stdout.write_text(
        "H\tJOB ID\tSTATE\tJOB START DATE\tELAPSE TIME (USE)\tEXIT CODE\t"
        "SIGNAL NO\tPJM CODE\tNODE NUM (USE)\tMAX MEMORY SIZE (USE)\t"
        "USER CPU TIME (USE)\tSYSTEM CPU TIME (USE)\n"
        "\t123\tEXT\t2026/08/10 09:00:00\t00:01:02 (62)\t0\t-\t0\t2\t"
        "4.0 KiB (4096)\t12000 ms\t3000 ms\n",
        encoding="utf-8",
    )
    stderr.write_bytes(b"")
    path, facts = create_scheduler_accounting(
        manifest_path,
        runtime,
        job_id="123",
        stdout_path=stdout,
        stderr_path=stderr,
        exit_code=0,
    )
    assert validate_scheduler_accounting(path, runtime, RUN_ID) == facts
    values = {fact.key: fact.logical_value for fact in facts}
    assert values["scheduler.accounting.elp.seconds"] == 62
    assert values["scheduler.accounting.mmszu.bytes"] == 4096


def test_ordered_workload_is_closed_and_order_sensitive(tmp_path: Path) -> None:
    dataset_identity = "a" * 64
    value = {
        "schema_version": "ordered-workload/v1",
        "dataset_manifest_sha256": dataset_identity,
        "kind": "measured",
        "requests": [
            {
                "source": {"kind": "dataset_row", "array_index": 9},
                "prompt_payload": {
                    "kind": "text",
                    "text": "hello",
                    "input_token_count": 1,
                },
                "requested_output_token_count": 2,
                "sampling_params": [
                    {"name": "ignore_eos", "declared_type": "bool", "value": True},
                    {"name": "temperature", "declared_type": "int", "value": 0},
                ],
            }
        ],
    }
    path = tmp_path / "workload-measured.json"
    path.write_bytes(jcs_bytes(value) + b"\n")
    facts = validate_ordered_workload(
        path, kind="measured", dataset_manifest_sha256=dataset_identity
    )
    assert facts[0].logical_value == hashlib.sha256(jcs_bytes(value)).hexdigest()
    value["requests"][0]["sampling_params"].reverse()  # type: ignore[index]
    path.write_bytes(jcs_bytes(value) + b"\n")
    with pytest.raises(ValueError, match="sorted"):
        validate_ordered_workload(
            path, kind="measured", dataset_manifest_sha256=dataset_identity
        )


def test_node_lifecycle_preserves_raw_wait_status(tmp_path: Path) -> None:
    path = tmp_path / "node.json"
    path.write_bytes(
        jcs_bytes(
            {
                "schema_version": "node-lifecycle/v1",
                "clock_id": "boot:CLOCK_MONOTONIC",
                "spawn_attempt_monotonic_ns": 1,
                "child_started_monotonic_ns": 2,
                "child_finished_monotonic_ns": 3,
                "termination": {"kind": "waited", "raw_wait_status": 7 << 8},
            }
        )
        + b"\n"
    )
    facts = validate_node_lifecycle(path, node_rank=1)
    assert facts[-1].logical_value == 7 << 8


def test_lifecycle_summary_rejects_cross_clock_shutdown(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_bytes(
        jcs_bytes(
            {
                "schema_version": "lifecycle-summary/v1",
                "run_id": RUN_ID,
                "nodes": [
                    {
                        "node_rank": 0,
                        "ready": {
                            "sequence": 0,
                            "clock_id": "a:CLOCK_MONOTONIC",
                            "received_monotonic_ns": 1,
                            "ack_monotonic_ns": 2,
                        },
                        "shutdown": {
                            "sequence": 1,
                            "invocation_id": "measured-generation",
                            "clock_id": "b:CLOCK_MONOTONIC",
                            "received_monotonic_ns": 3,
                            "ack_monotonic_ns": 4,
                        },
                    }
                ],
            }
        )
        + b"\n"
    )
    with pytest.raises(ValueError, match="clocks"):
        validate_lifecycle_summary(path, RUN_ID, 1)


def test_two_node_agent_records_ready_shutdown_and_raw_wait(
    manifest_path: Path, runtime: Path
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings_path = runtime / "control/execution-settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings["node_agent_port"] = port
    settings["rendezvous_timeout_sec"] = 5
    settings["ready_timeout_sec"] = 5
    settings_path.write_bytes(jcs_bytes(settings) + b"\n")
    common = [
        sys.executable,
        "-m",
        "qwen_experiment_metrics.node_agent",
        "--manifest",
        str(manifest_path),
        "--runtime-dir",
        str(runtime),
        "--master-host",
        "127.0.0.1",
    ]
    coordinator = subprocess.Popen(
        [*common, "--node-rank", "0", "--", "/bin/true"]
    )
    time.sleep(0.1)
    participant = subprocess.run(
        [
            *common,
            "--node-rank",
            "1",
            "--",
            "/bin/sh",
            "-c",
            "trap 'exit 0' TERM; while :; do sleep 0.1; done",
        ],
        check=False,
        timeout=10,
    )
    assert participant.returncode == 0
    assert coordinator.wait(timeout=10) == 0
    assert validate_lifecycle_summary(
        runtime / "control/lifecycle/summary.json", RUN_ID, 2
    )
    for node in range(2):
        validate_node_lifecycle(
            runtime / f"source/lifecycle/node-{node}.json", node_rank=node
        )


def test_participant_waits_for_shutdown_after_its_child_exits(
    manifest_path: Path, runtime: Path
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings_path = runtime / "control/execution-settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings["node_agent_port"] = port
    settings["rendezvous_timeout_sec"] = 5
    settings["ready_timeout_sec"] = 2
    settings_path.write_bytes(jcs_bytes(settings) + b"\n")
    common = [
        sys.executable,
        "-m",
        "qwen_experiment_metrics.node_agent",
        "--manifest",
        str(manifest_path),
        "--runtime-dir",
        str(runtime),
        "--master-host",
        "127.0.0.1",
    ]
    coordinator = subprocess.Popen(
        [*common, "--node-rank", "0", "--", "/bin/sleep", "0.5"]
    )
    time.sleep(0.1)
    participant = subprocess.run(
        [*common, "--node-rank", "1", "--", "/bin/true"],
        check=False,
        timeout=10,
    )
    assert participant.returncode == 0
    assert coordinator.wait(timeout=10) == 0


def test_profile_node_agents_broadcast_measured_phase(
    runtime: Path, tmp_path: Path
) -> None:
    manifest_path = make_manifest(runtime, "profile")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings_path = runtime / "control/execution-settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings["node_agent_port"] = port
    settings["rendezvous_timeout_sec"] = 5
    settings["ready_timeout_sec"] = 5
    settings_path.write_bytes(jcs_bytes(settings) + b"\n")
    phase_socket = tmp_path / "phase.sock"
    child = """
import json, socket, sys, time
def request(value):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.connect(sys.argv[1])
        stream.sendall(json.dumps(value, separators=(\",\", \":\")).encode() + b\"\\n\")
        payload = b\"\"
        while not payload.endswith(b\"\\n\"):
            payload += stream.recv(4096)
        return json.loads(payload)
identity = {\"invocation_id\": \"measured-generation\", \"kind\": \"measured\"}
assert request({\"type\": \"phase_start\", \"sequence\": 0, **identity})[\"type\"] == \"phase_start_ack\"
started = time.monotonic_ns()
time.sleep(0.01)
finished = time.monotonic_ns()
assert request({\"type\": \"phase_end\", \"sequence\": 1, **identity, \"generation_start_ns\": started, \"generation_end_ns\": finished})[\"type\"] == \"phase_end_ack\"
"""
    common = [
        sys.executable,
        "-m",
        "qwen_experiment_metrics.node_agent",
        "--manifest",
        str(manifest_path),
        "--runtime-dir",
        str(runtime),
        "--master-host",
        "127.0.0.1",
    ]
    coordinator = subprocess.Popen(
        [
            *common,
            "--node-rank",
            "0",
            "--phase-socket",
            str(phase_socket),
            "--",
            sys.executable,
            "-c",
            child,
            str(phase_socket),
        ]
    )
    time.sleep(0.1)
    participant = subprocess.run(
        [
            *common,
            "--node-rank",
            "1",
            "--",
            "/bin/sh",
            "-c",
            "trap 'exit 0' TERM; while :; do sleep 0.1; done",
        ],
        check=False,
        timeout=10,
    )
    assert participant.returncode == 0
    assert coordinator.wait(timeout=10) == 0
    phase = parse_phase_summary(
        runtime / "control/phase/summary.json", run_id=RUN_ID, expected_nodes=2
    )
    assert phase.invocation_id == "measured-generation"
    assert phase.kind == "measured"
    assert all(node.clock_id for node in phase.nodes)
