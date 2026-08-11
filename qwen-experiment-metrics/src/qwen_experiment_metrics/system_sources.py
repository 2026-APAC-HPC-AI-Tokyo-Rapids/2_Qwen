from __future__ import annotations

import base64
import hashlib
import os
import re
import socket
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath

from .canonical import jcs_bytes, load_json_strict
from .domain import Fact
from .manifest import load_manifest

_SAFE_KEY = re.compile(rb"^[A-Za-z0-9_-]+$")
_INT = re.compile(r"^-?[0-9]+$")
_UINT = re.compile(r"^[0-9]+$")
_GPU_QUERY_FIELDS = (
    "index",
    "name",
    "uuid",
    "pci.bus_id",
    "memory.total",
    "driver_version",
    "persistence_mode",
    "compute_mode",
    "power.limit",
    "clocks.max.sm",
    "clocks.max.memory",
    "mig.mode.current",
    "ecc.mode.current",
)
_ACCOUNTING_COLUMNS = (
    "jid",
    "st",
    "sdt",
    "elp",
    "ec",
    "sn",
    "pc",
    "nnumu",
    "mmszu",
    "uctmut",
    "sctmut",
)


def capture_environment_snapshot(
    destination: Path, environment: dict[bytes, bytes] | None = None
) -> Path:
    values = os.environb if environment is None else environment
    rows = []
    for name, value in values.items():
        if name in {
            b"EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_FILE",
            b"EXPERIMENT_METRICS_BIGQUERY_CONFIG",
            b"EXPERIMENT_METRICS_DIRECT_SEND",
        }:
            continue
        if not name or b"\0" in name or b"=" in name or b"\0" in value:
            raise ValueError("environment contains an invalid name or NUL value")
        rows.append((name, value))
    rows.sort(key=lambda item: item[0])
    payload = b"".join(name + b"\0" + value + b"\0" for name, value in rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_bytes(destination, payload)
    validate_environment_snapshot(destination)
    return destination


def validate_environment_snapshot(path: Path) -> tuple[tuple[bytes, bytes], ...]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\0"):
        raise ValueError("environment snapshot must be nonempty and NUL terminated")
    fields = payload[:-1].split(b"\0")
    if len(fields) % 2:
        raise ValueError("environment snapshot has an incomplete pair")
    rows = tuple(zip(fields[::2], fields[1::2], strict=True))
    names = [name for name, _ in rows]
    if (
        any(not name or b"=" in name for name in names)
        or names != sorted(names)
        or len(names) != len(set(names))
    ):
        raise ValueError("environment names must be byte-sorted and unique")
    return rows


def environment_facts(path: Path, *, key_prefix: str) -> tuple[Fact, ...]:
    return tuple(
        Fact(
            f"{key_prefix}.{_escape_environment_name(name)}",
            base64.b64encode(value).decode("ascii"),
        )
        for name, value in validate_environment_snapshot(path)
    )


def start_run_lifecycle(manifest_path: Path, runtime_dir: Path) -> Path:
    manifest = load_manifest(manifest_path)
    clock_id = _clock_id()
    value = {
        "schema_version": "run-lifecycle-start/v1",
        "run_id": manifest.run_id,
        "clock_id": clock_id,
        "started_at": _utc_now(),
        "started_monotonic_ns": time.monotonic_ns(),
    }
    destination = runtime_dir / "control/lifecycle/run-start.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_json(destination, value)
    return destination


def finish_run_lifecycle(
    manifest_path: Path, runtime_dir: Path, exit_code: int
) -> tuple[Path, tuple[Fact, ...]]:
    manifest = load_manifest(manifest_path)
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise TypeError("run exit code must be an integer")
    start_path = runtime_dir / "control/lifecycle/run-start.json"
    start = load_json_strict(start_path.read_bytes())
    if not isinstance(start, dict) or set(start) != {
        "schema_version",
        "run_id",
        "clock_id",
        "started_at",
        "started_monotonic_ns",
    }:
        raise ValueError("run lifecycle start is not closed")
    if (
        start["schema_version"] != "run-lifecycle-start/v1"
        or start["run_id"] != manifest.run_id
        or start["clock_id"] != _clock_id()
    ):
        raise ValueError("run lifecycle start identity differs")
    started_ns = _uint(start["started_monotonic_ns"], "started monotonic ns")
    finished_ns = time.monotonic_ns()
    if finished_ns < started_ns:
        raise ValueError("run monotonic clock moved backwards")
    elapsed = Decimal(finished_ns - started_ns) / Decimal(1_000_000_000)
    source = {
        "schema_version": "run-lifecycle/v1",
        "run_id": manifest.run_id,
        "clock_id": start["clock_id"],
        "started_at": start["started_at"],
        "finished_at": _utc_now(),
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": finished_ns,
        "elapsed_sec": str(elapsed),
        "exit_code": exit_code,
    }
    destination = runtime_dir / "source/run-lifecycle.json"
    _exclusive_json(destination, source)
    facts = validate_run_lifecycle(destination, manifest.run_id, start_path=start_path)
    return destination, facts


def validate_run_lifecycle(
    path: Path, run_id: str, *, start_path: Path | None = None
) -> tuple[Fact, ...]:
    value = load_json_strict(path.read_bytes())
    expected = {
        "schema_version",
        "run_id",
        "clock_id",
        "started_at",
        "finished_at",
        "started_monotonic_ns",
        "finished_monotonic_ns",
        "elapsed_sec",
        "exit_code",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("run lifecycle is not closed")
    if value["schema_version"] != "run-lifecycle/v1" or value["run_id"] != run_id:
        raise ValueError("run lifecycle identity differs")
    _nonempty(value["clock_id"], "run clock ID")
    started_at = _rfc3339(value["started_at"], "run start")
    finished_at = _rfc3339(value["finished_at"], "run finish")
    if finished_at < started_at:
        raise ValueError("run wall timestamps are reversed")
    started_ns = _uint(value["started_monotonic_ns"], "run start ns")
    finished_ns = _uint(value["finished_monotonic_ns"], "run finish ns")
    if finished_ns < started_ns:
        raise ValueError("run monotonic timestamps are reversed")
    elapsed = _decimal(value["elapsed_sec"], "run elapsed")
    if elapsed != Decimal(finished_ns - started_ns) / Decimal(1_000_000_000):
        raise ValueError("run elapsed does not match monotonic timestamps")
    exit_code = _int(value["exit_code"], "run exit code")
    if start_path is not None:
        start = load_json_strict(start_path.read_bytes())
        if not isinstance(start, dict) or set(start) != {
            "schema_version",
            "run_id",
            "clock_id",
            "started_at",
            "started_monotonic_ns",
        }:
            raise ValueError("run lifecycle start is not closed")
        if start["schema_version"] != "run-lifecycle-start/v1":
            raise ValueError("run lifecycle start schema differs")
        for name in ("run_id", "clock_id", "started_at", "started_monotonic_ns"):
            if start[name] != value[name]:
                raise ValueError("run lifecycle start binding differs")
    return (
        Fact("run.started_at", value["started_at"]),
        Fact("run.finished_at", value["finished_at"]),
        Fact("run.elapsed_sec", elapsed),
        Fact("run.exit_code", exit_code),
    )


def create_scheduler_accounting(
    manifest_path: Path,
    runtime_dir: Path,
    *,
    job_id: str,
    stdout_path: Path,
    stderr_path: Path,
    exit_code: int,
) -> tuple[Path, tuple[Fact, ...]]:
    manifest = load_manifest(manifest_path)
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise ValueError("accounting job ID is invalid")
    row = _parse_accounting_stdout(stdout_path.read_bytes())
    argv = [
        "pjstat",
        "--history",
        "-s",
        "--choose",
        ",".join(_ACCOUNTING_COLUMNS),
        "--data",
        "--delimiter",
        "\t",
        job_id,
    ]
    value = {
        "schema_version": "scheduler-accounting/v1",
        "run_id": manifest.run_id,
        "argv": argv,
        "outcome": {"kind": "exit", "code": exit_code},
        "stdout": _member(stdout_path, "source/scheduler/accounting.txt"),
        "stderr": _member(stderr_path, "source/scheduler/accounting.stderr"),
        "columns": list(_ACCOUNTING_COLUMNS),
        "row": row,
    }
    destination = runtime_dir / "source/scheduler/accounting.json"
    _exclusive_json(destination, value)
    facts = validate_scheduler_accounting(destination, runtime_dir, manifest.run_id)
    return destination, facts


def validate_scheduler_accounting(
    path: Path, runtime_dir: Path, run_id: str
) -> tuple[Fact, ...]:
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "run_id",
        "argv",
        "outcome",
        "stdout",
        "stderr",
        "columns",
        "row",
    }:
        raise ValueError("scheduler accounting is not closed")
    if (
        value["schema_version"] != "scheduler-accounting/v1"
        or value["run_id"] != run_id
    ):
        raise ValueError("scheduler accounting identity differs")
    if value["columns"] != list(_ACCOUNTING_COLUMNS):
        raise ValueError("scheduler accounting columns differ")
    argv = value["argv"]
    if not isinstance(argv, list) or argv[:-1] != [
        "pjstat",
        "--history",
        "-s",
        "--choose",
        ",".join(_ACCOUNTING_COLUMNS),
        "--data",
        "--delimiter",
        "\t",
    ]:
        raise ValueError("scheduler accounting argv differs")
    outcome = value["outcome"]
    if outcome != {"kind": "exit", "code": 0}:
        raise ValueError("scheduler accounting command did not succeed")
    for name, expected_path in (
        ("stdout", "source/scheduler/accounting.txt"),
        ("stderr", "source/scheduler/accounting.stderr"),
    ):
        relative = _validate_member(value[name])
        if relative != expected_path:
            raise ValueError("scheduler accounting member path differs")
        _bind_member(runtime_dir / relative, value[name])
    row = value["row"]
    if not isinstance(row, dict) or set(row) != set(_ACCOUNTING_COLUMNS):
        raise ValueError("scheduler accounting row is not closed")
    reparsed = _parse_accounting_stdout(
        runtime_dir.joinpath("source/scheduler/accounting.txt").read_bytes()
    )
    if row != reparsed or row["jid"] != argv[-1]:
        raise ValueError("scheduler accounting row binding differs")
    return _accounting_facts(row)


def create_minimal_hardware(runtime_dir: Path, *, node_rank: int) -> Path:
    source_dir = runtime_dir / "source/hardware"
    source_dir.mkdir(parents=True, exist_ok=True)
    stderr = source_dir / f"node-{node_rank}.lscpu.stderr"
    try:
        completed = subprocess.run(
            ["lscpu", "--json"],
            check=False,
            capture_output=True,
            env={**os.environ, "LC_ALL": "C"},
        )
        stderr.write_bytes(completed.stderr)
        lscpu_json = (
            load_json_strict(completed.stdout) if completed.returncode == 0 else None
        )
        exit_code = completed.returncode
    except OSError:
        stderr.write_bytes(b"")
        lscpu_json = None
        exit_code = 127
    value = {
        "schema_version": "hardware-node/v1",
        "node_rank": node_rank,
        "hostname": socket.gethostname(),
        "commands": {
            "lscpu": {
                "argv": ["lscpu", "--json"],
                "exit_code": exit_code,
                "stderr_member": f"source/hardware/node-{node_rank}.lscpu.stderr",
            }
        },
        "lscpu_json": lscpu_json,
        "platform_settings": _platform_settings(),
    }
    destination = source_dir / f"node-{node_rank}.json"
    _exclusive_json(destination, value)
    validate_minimal_hardware(destination, runtime_dir, node_rank)
    return destination


def validate_minimal_hardware(
    path: Path, runtime_dir: Path, node_rank: int
) -> tuple[Fact, ...]:
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "node_rank",
        "hostname",
        "commands",
        "lscpu_json",
        "platform_settings",
    }:
        raise ValueError("minimal hardware source is not closed")
    if value["schema_version"] != "hardware-node/v1" or value["node_rank"] != node_rank:
        raise ValueError("minimal hardware identity differs")
    hostname = _nonempty(value["hostname"], "hardware hostname")
    commands = value["commands"]
    if not isinstance(commands, dict) or set(commands) != {"lscpu"}:
        raise ValueError("hardware commands are not closed")
    command = commands["lscpu"]
    if not isinstance(command, dict) or set(command) != {
        "argv",
        "exit_code",
        "stderr_member",
    }:
        raise ValueError("lscpu command is not closed")
    if command["argv"] != ["lscpu", "--json"]:
        raise ValueError("lscpu argv differs")
    _int(command["exit_code"], "lscpu exit code")
    stderr_member = _relative(command["stderr_member"], "lscpu stderr member")
    if not (runtime_dir / stderr_member).is_file():
        raise ValueError("lscpu stderr member is missing")
    if value["lscpu_json"] is not None and not isinstance(value["lscpu_json"], dict):
        raise TypeError("lscpu JSON must be an object or null")
    settings = _validate_platform_settings(value["platform_settings"])
    return (
        *_minimal_hardware_facts(node_rank, hostname, settings),
        *_lscpu_facts(node_rank, value["lscpu_json"]),
    )


def create_node_topology(
    manifest_path: Path,
    runtime_dir: Path,
    *,
    node_rank: int,
    global_rank: int,
    local_rank: int,
    size: int,
    node_wrapper_argv: list[str],
    singularity_argv: list[str],
    sglang_argv: list[str],
    singularity_launcher: Path,
    sif: Path,
) -> Path:
    manifest = load_manifest(manifest_path)
    topology_dir = runtime_dir / "source/topology"
    topology_dir.mkdir(parents=True, exist_ok=True)
    query = "--query-gpu=" + ",".join(_GPU_QUERY_FIELDS)
    query_argv = ["nvidia-smi", query, "--format=csv,noheader,nounits"]
    host_query = _capture_source_command(
        query_argv,
        topology_dir / f"node-{node_rank}.host-gpu.stdout",
        topology_dir / f"node-{node_rank}.host-gpu.stderr",
        runtime_dir,
    )
    container_query_argv = [
        str(singularity_launcher),
        "exec",
        "--nv",
        str(sif),
        *query_argv,
    ]
    container_query = _capture_source_command(
        container_query_argv,
        topology_dir / f"node-{node_rank}.container-gpu.stdout",
        topology_dir / f"node-{node_rank}.container-gpu.stderr",
        runtime_dir,
    )
    topo_argv = ["nvidia-smi", "topo", "-m"]
    host_topology = _capture_source_command(
        topo_argv,
        topology_dir / f"node-{node_rank}.host-topology.stdout",
        topology_dir / f"node-{node_rank}.host-topology.stderr",
        runtime_dir,
    )
    value = {
        "schema_version": "node-topology/v1",
        "run_id": manifest.run_id,
        "node_rank": node_rank,
        "outer_mpi": {
            "global_rank": global_rank,
            "local_rank": local_rank,
            "size": size,
            "hostname": socket.gethostname(),
        },
        "gpu_commands": {
            "host": host_query,
            "container": container_query,
            "host_topology": host_topology,
        },
        "node_wrapper_argv": node_wrapper_argv,
        "singularity_argv": singularity_argv,
        "sglang_argv": sglang_argv,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "numa_mems": _numa_mems(),
    }
    destination = topology_dir / f"node-{node_rank}.json"
    _exclusive_json(destination, value)
    validate_node_topology(destination, runtime_dir, manifest.run_id, node_rank)
    return destination


def validate_node_topology(
    path: Path,
    runtime_dir: Path,
    run_id: str,
    node_rank: int,
    *,
    expected_gpus: int | None = None,
) -> tuple[Fact, ...]:
    value = load_json_strict(path.read_bytes())
    expected = {
        "schema_version",
        "run_id",
        "node_rank",
        "outer_mpi",
        "gpu_commands",
        "node_wrapper_argv",
        "singularity_argv",
        "sglang_argv",
        "cpu_affinity",
        "numa_mems",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("node topology is not closed")
    if (
        value["schema_version"] != "node-topology/v1"
        or value["run_id"] != run_id
        or value["node_rank"] != node_rank
    ):
        raise ValueError("node topology identity differs")
    outer = value["outer_mpi"]
    if not isinstance(outer, dict) or set(outer) != {
        "global_rank",
        "local_rank",
        "size",
        "hostname",
    }:
        raise ValueError("outer MPI topology is not closed")
    for name in ("global_rank", "local_rank", "size"):
        _uint(outer[name], f"outer MPI {name}")
    if outer["global_rank"] != node_rank or outer["local_rank"] != 0:
        raise ValueError("outer MPI rank placement differs")
    _nonempty(outer["hostname"], "outer MPI hostname")
    commands = value["gpu_commands"]
    if not isinstance(commands, dict) or set(commands) != {
        "host",
        "container",
        "host_topology",
    }:
        raise ValueError("GPU command set is not closed")
    query = "--query-gpu=" + ",".join(_GPU_QUERY_FIELDS)
    expected_query = ["nvidia-smi", query, "--format=csv,noheader,nounits"]
    if commands["host"].get("argv") != expected_query:
        raise ValueError("host GPU query argv differs")
    container_argv = commands["container"].get("argv")
    if not isinstance(container_argv, list) or container_argv[-3:] != expected_query:
        raise ValueError("container GPU query argv differs")
    if commands["host_topology"].get("argv") != ["nvidia-smi", "topo", "-m"]:
        raise ValueError("host GPU topology argv differs")
    for record in commands.values():
        _validate_source_command(record, runtime_dir)
        if record["exit_code"] != 0:
            raise ValueError("topology source command did not succeed")
    for name in ("node_wrapper_argv", "singularity_argv", "sglang_argv"):
        _string_array(value[name], name)
    affinity = _uint_array(value["cpu_affinity"], "node CPU affinity")
    if affinity != sorted(set(affinity)):
        raise ValueError("node CPU affinity must be sorted and unique")
    numa_mems = value["numa_mems"]
    if numa_mems is not None:
        _nonempty(numa_mems, "node NUMA mems")
    facts = _topology_facts(value, runtime_dir, expected_gpus)
    return facts


def topology_gpu_bindings(path: Path, runtime_dir: Path) -> dict[str, int]:
    """Return validated container UUID-to-index bindings for profile sampling."""

    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError("node topology must be an object")
    commands = value.get("gpu_commands")
    if not isinstance(commands, dict) or not isinstance(
        commands.get("container"), dict
    ):
        raise TypeError("node topology has no container GPU command")
    command = commands["container"]
    _validate_source_command(command, runtime_dir)
    if command["exit_code"] != 0:
        raise ValueError("container GPU query failed")
    return {
        _nonempty(row["uuid"], "GPU UUID"): _uint(row["index"], "GPU index")
        for row in _parse_gpu_csv(runtime_dir / command["stdout_member"])
    }


def validate_node_lifecycle(path: Path, *, node_rank: int) -> tuple[Fact, ...]:
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "clock_id",
        "spawn_attempt_monotonic_ns",
        "child_started_monotonic_ns",
        "child_finished_monotonic_ns",
        "termination",
    }:
        raise ValueError("node lifecycle is not closed")
    if value["schema_version"] != "node-lifecycle/v1":
        raise ValueError("unsupported node lifecycle")
    _nonempty(value["clock_id"], "node lifecycle clock ID")
    attempt = _uint(value["spawn_attempt_monotonic_ns"], "spawn attempt")
    termination = value["termination"]
    if not isinstance(termination, dict) or termination.get("kind") not in {
        "waited",
        "spawn_error",
    }:
        raise ValueError("node termination is invalid")
    started = value["child_started_monotonic_ns"]
    finished = value["child_finished_monotonic_ns"]
    if termination["kind"] == "spawn_error":
        if set(termination) != {"kind"} or started is not None or finished is not None:
            raise ValueError("spawn error lifecycle fields differ")
    else:
        if set(termination) != {"kind", "raw_wait_status"}:
            raise ValueError("waited lifecycle fields differ")
        started_ns = _uint(started, "child start")
        finished_ns = _uint(finished, "child finish")
        raw_status = _uint(termination["raw_wait_status"], "raw wait status")
        if not (attempt <= started_ns <= finished_ns):
            raise ValueError("node lifecycle timestamps are reversed")
        if os.WIFEXITED(raw_status) == os.WIFSIGNALED(raw_status):
            raise ValueError("node raw wait status is not terminal")
    prefix = f"parallel.node.{node_rank}.process"
    facts = [
        Fact(f"{prefix}.clock_id", value["clock_id"]),
        Fact(f"{prefix}.spawn_attempt_ns", attempt),
        Fact(f"{prefix}.termination.kind", termination["kind"]),
    ]
    if started is not None:
        facts.append(Fact(f"{prefix}.child_started_ns", started))
    if finished is not None:
        facts.append(Fact(f"{prefix}.child_finished_ns", finished))
    if termination["kind"] == "waited":
        facts.append(
            Fact(
                f"{prefix}.termination.raw_wait_status",
                termination["raw_wait_status"],
            )
        )
    return tuple(facts)


def validate_lifecycle_rendezvous(
    path: Path, run_id: str, expected_port: int
) -> tuple[Fact, ...]:
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "run_id",
        "rank0_hostname",
        "address",
        "port",
        "created_at",
    }:
        raise ValueError("lifecycle rendezvous is not closed")
    if (
        value["schema_version"] != "lifecycle-rendezvous/v1"
        or value["run_id"] != run_id
        or value["port"] != expected_port
    ):
        raise ValueError("lifecycle rendezvous identity differs")
    _nonempty(value["rank0_hostname"], "rank zero hostname")
    _nonempty(value["address"], "rendezvous address")
    _rfc3339(value["created_at"], "rendezvous creation")
    return tuple(
        Fact(f"lifecycle.rendezvous.{name}", value[name])
        for name in ("rank0_hostname", "address", "port", "created_at")
    )


def validate_lifecycle_summary(
    path: Path, run_id: str, expected_nodes: int
) -> tuple[Fact, ...]:
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "run_id",
        "nodes",
    }:
        raise ValueError("lifecycle summary is not closed")
    if value["schema_version"] != "lifecycle-summary/v1" or value["run_id"] != run_id:
        raise ValueError("lifecycle summary identity differs")
    nodes = value["nodes"]
    if not isinstance(nodes, list) or len(nodes) != expected_nodes:
        raise ValueError("lifecycle summary node count differs")
    if [row.get("node_rank") for row in nodes if isinstance(row, dict)] != list(
        range(expected_nodes)
    ):
        raise ValueError("lifecycle summary node ranks differ")
    facts: list[Fact] = []
    for row in nodes:
        if not isinstance(row, dict) or set(row) != {"node_rank", "ready", "shutdown"}:
            raise ValueError("lifecycle node entry is not closed")
        node = row["node_rank"]
        ready = _validate_lifecycle_mark(row["ready"], "ready")
        facts.extend(_lifecycle_mark_facts(node, "ready", ready))
        shutdown = row["shutdown"]
        if shutdown is not None:
            shutdown_value = _validate_lifecycle_mark(
                shutdown, "shutdown", invocation=True
            )
            if shutdown_value["clock_id"] != ready["clock_id"]:
                raise ValueError("lifecycle node clocks differ")
            if ready["ack_monotonic_ns"] > shutdown_value["received_monotonic_ns"]:
                raise ValueError("shutdown precedes ready ACK")
            facts.extend(_lifecycle_mark_facts(node, "shutdown", shutdown_value))
    return tuple(facts)


def _parse_accounting_stdout(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("scheduler accounting is not UTF-8") from error
    lines = text.splitlines()
    if len(lines) != 2:
        raise ValueError("scheduler accounting must contain one header and data row")
    header = lines[0].split("\t")
    fields = lines[1].split("\t")
    if (
        len(header) != len(_ACCOUNTING_COLUMNS) + 1
        or header[0] != "H"
        or len(fields) != len(_ACCOUNTING_COLUMNS) + 1
        or fields[0] != ""
    ):
        raise ValueError("scheduler accounting column count differs")
    return dict(zip(_ACCOUNTING_COLUMNS, fields[1:], strict=True))


def _accounting_facts(row: dict[str, str]) -> tuple[Fact, ...]:
    facts = [
        Fact(f"scheduler.accounting.{name}", row[name]) for name in ("jid", "st", "sdt")
    ]
    elapsed = _duration_seconds(row["elp"])
    facts.append(Fact("scheduler.accounting.elp.lexeme", row["elp"]))
    if elapsed is not None:
        facts.append(Fact("scheduler.accounting.elp.seconds", elapsed))
    for name in ("ec", "sn", "pc", "nnumu"):
        facts.append(Fact(f"scheduler.accounting.{name}.lexeme", row[name]))
        parsed = _nullable_int_lexeme(row[name])
        if parsed is not None:
            facts.append(Fact(f"scheduler.accounting.{name}.value", parsed))
    facts.append(Fact("scheduler.accounting.mmszu.lexeme", row["mmszu"]))
    memory = _memory_bytes(row["mmszu"])
    if memory is not None:
        facts.append(Fact("scheduler.accounting.mmszu.bytes", memory))
    for name in ("uctmut", "sctmut"):
        facts.append(Fact(f"scheduler.accounting.{name}.lexeme", row[name]))
        seconds = _milliseconds_seconds(row[name])
        if seconds is not None:
            facts.append(Fact(f"scheduler.accounting.{name}.seconds", seconds))
    return tuple(facts)


def _platform_settings() -> dict[str, object]:
    cpufreq_root = Path("/sys/devices/system/cpu/cpufreq")
    policies = None
    try:
        if cpufreq_root.exists():
            policy_paths = sorted(
                (
                    path
                    for path in cpufreq_root.iterdir()
                    if re.fullmatch(r"policy[0-9]+", path.name)
                ),
                key=lambda path: int(path.name.removeprefix("policy")),
            )
            policies = [
                {
                    "policy_id": int(path.name.removeprefix("policy")),
                    "related_cpus": _whitespace_uints(path / "related_cpus"),
                    "driver": _read_optional(path / "scaling_driver"),
                    "governor": _read_optional(path / "scaling_governor"),
                    "min_hz": _khz_optional(path / "scaling_min_freq"),
                    "max_hz": _khz_optional(path / "scaling_max_freq"),
                }
                for path in policy_paths
            ]
    except OSError:
        policies = None
    boost = _boost_settings()
    meminfo = _meminfo()
    return {
        "kernel_release": _read_optional(Path("/proc/sys/kernel/osrelease")),
        "online_cpus": _cpu_list_optional(Path("/sys/devices/system/cpu/online")),
        "isolated_cpus": _cpu_list_optional(Path("/sys/devices/system/cpu/isolated")),
        "nohz_full_cpus": _cpu_list_optional(
            Path("/sys/devices/system/cpu/nohz_full"), null_literal=True
        ),
        "memory": {
            "total_bytes": _meminfo_bytes(meminfo, "MemTotal"),
            "hugepage_size_bytes": _meminfo_bytes(meminfo, "Hugepagesize"),
            "hugepages_total": _meminfo_uint(meminfo, "HugePages_Total"),
        },
        "numa_balancing_enabled": _bool_file(Path("/proc/sys/kernel/numa_balancing")),
        "boost": boost,
        "cpufreq_policies": policies,
    }


def _validate_platform_settings(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "kernel_release",
        "online_cpus",
        "isolated_cpus",
        "nohz_full_cpus",
        "memory",
        "numa_balancing_enabled",
        "boost",
        "cpufreq_policies",
    }:
        raise ValueError("platform settings are not closed")
    for name in ("kernel_release",):
        if value[name] is not None:
            _nonempty(value[name], name)
    for name in ("online_cpus", "isolated_cpus", "nohz_full_cpus"):
        if value[name] is not None:
            rows = _uint_array(value[name], name)
            if rows != sorted(set(rows)):
                raise ValueError(f"{name} is not sorted and unique")
    memory = value["memory"]
    if not isinstance(memory, dict) or set(memory) != {
        "total_bytes",
        "hugepage_size_bytes",
        "hugepages_total",
    }:
        raise ValueError("platform memory is not closed")
    for child in memory.values():
        if child is not None:
            _uint(child, "platform memory")
    if value["numa_balancing_enabled"] is not None and not isinstance(
        value["numa_balancing_enabled"], bool
    ):
        raise TypeError("NUMA balancing setting must be bool or null")
    boost = value["boost"]
    if not isinstance(boost, dict) or set(boost) != {"source_path", "raw", "enabled"}:
        raise ValueError("CPU boost setting is not closed")
    if boost["source_path"] is None:
        if boost != {"source_path": None, "raw": None, "enabled": None}:
            raise ValueError("absent CPU boost setting is inconsistent")
    else:
        _relative(boost["source_path"], "CPU boost source")
        _nonempty(boost["raw"], "CPU boost raw")
        if boost["enabled"] is not None and not isinstance(boost["enabled"], bool):
            raise TypeError("CPU boost enabled must be bool or null")
    policies = value["cpufreq_policies"]
    if policies is not None:
        if not isinstance(policies, list):
            raise TypeError("CPU frequency policies must be an array or null")
        ids = []
        for row in policies:
            if not isinstance(row, dict) or set(row) != {
                "policy_id",
                "related_cpus",
                "driver",
                "governor",
                "min_hz",
                "max_hz",
            }:
                raise ValueError("CPU frequency policy is not closed")
            ids.append(_uint(row["policy_id"], "CPU frequency policy ID"))
            cpus = _uint_array(row["related_cpus"], "related CPUs")
            if cpus != sorted(set(cpus)):
                raise ValueError("related CPUs are not sorted and unique")
            for name in ("driver", "governor"):
                if row[name] is not None:
                    _nonempty(row[name], name)
            for name in ("min_hz", "max_hz"):
                if row[name] is not None:
                    _uint(row[name], name)
        if ids != sorted(set(ids)):
            raise ValueError("CPU frequency policy IDs are not sorted and unique")
    return value


def _minimal_hardware_facts(
    node: int, hostname: str, settings: dict[str, object]
) -> tuple[Fact, ...]:
    prefix = f"hardware.node.{node}"
    facts = [Fact(f"{prefix}.hostname", hostname)]
    scalar_paths = (
        ("kernel.release", settings["kernel_release"]),
        ("cpu.online", settings["online_cpus"]),
        ("cpu.isolated", settings["isolated_cpus"]),
        ("cpu.nohz_full", settings["nohz_full_cpus"]),
        ("numa_balancing_enabled", settings["numa_balancing_enabled"]),
    )
    for suffix, child in scalar_paths:
        if child is not None:
            facts.append(Fact(f"{prefix}.{suffix}", child))
    memory = settings["memory"]
    for name, child in memory.items():
        if child is not None:
            facts.append(Fact(f"{prefix}.memory.{name}", child))
    boost = settings["boost"]
    for name, child in boost.items():
        if child is not None:
            facts.append(Fact(f"{prefix}.cpu_boost.{name}", child))
    policies = settings["cpufreq_policies"]
    if policies is not None:
        for row in policies:
            for name, child in row.items():
                if name != "policy_id" and child is not None:
                    facts.append(
                        Fact(
                            f"{prefix}.cpufreq_policy.{row['policy_id']}.{name}",
                            child,
                        )
                    )
    return tuple(facts)


def _lscpu_facts(node: int, value: object) -> tuple[Fact, ...]:
    if value is None:
        return ()
    rows = value.get("lscpu") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise TypeError("lscpu JSON has no lscpu array")
    labels: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"field", "data"}:
            raise ValueError("lscpu row is not closed")
        field = _nonempty(row["field"], "lscpu field")
        data = _nonempty(row["data"], "lscpu data")
        name = field.removesuffix(":")
        if name in labels:
            raise ValueError("lscpu label is duplicated")
        labels[name] = data
    prefix = f"hardware.node.{node}.cpu"
    facts = []
    for label, name in (
        ("Architecture", "architecture"),
        ("Model name", "model_name"),
        ("Vendor ID", "vendor_id"),
        ("CPU family", "cpu_family"),
        ("Model", "model"),
        ("Stepping", "stepping"),
    ):
        if label in labels:
            facts.append(Fact(f"{prefix}.{name}", labels[label]))
    for label, name in (("NUMA node(s)", "numa_nodes"),):
        if label in labels:
            facts.append(Fact(f"{prefix}.{name}", _uint_lexeme(labels[label], label)))
    if "Socket(s)" in labels and "Core(s) per socket" in labels:
        sockets = _uint_lexeme(labels["Socket(s)"], "Socket(s)")
        cores = _uint_lexeme(labels["Core(s) per socket"], "Core(s) per socket")
        facts.append(Fact(f"{prefix}.physical_cores", sockets * cores))
    return tuple(facts)


def _capture_source_command(
    argv: list[str], stdout: Path, stderr: Path, runtime_dir: Path
) -> dict[str, object]:
    try:
        with stdout.open("xb") as out, stderr.open("xb") as err:
            status = subprocess.run(
                argv, stdout=out, stderr=err, check=False
            ).returncode
    except OSError as error:
        stdout.write_bytes(b"")
        stderr.write_text(str(error), encoding="utf-8")
        status = 127
    return {
        "argv": argv,
        "exit_code": status,
        "stdout_member": stdout.relative_to(runtime_dir).as_posix(),
        "stdout_size_bytes": stdout.stat().st_size,
        "stdout_sha256": _sha256(stdout),
        "stderr_member": stderr.relative_to(runtime_dir).as_posix(),
        "stderr_size_bytes": stderr.stat().st_size,
        "stderr_sha256": _sha256(stderr),
    }


def _validate_source_command(value: object, runtime_dir: Path) -> None:
    if not isinstance(value, dict) or set(value) != {
        "argv",
        "exit_code",
        "stdout_member",
        "stdout_size_bytes",
        "stdout_sha256",
        "stderr_member",
        "stderr_size_bytes",
        "stderr_sha256",
    }:
        raise ValueError("source command is not closed")
    _string_array(value["argv"], "source command argv")
    _int(value["exit_code"], "source command exit code")
    for kind in ("stdout", "stderr"):
        relative = _relative(value[f"{kind}_member"], f"{kind} member")
        candidate = runtime_dir / relative
        if (
            not candidate.is_file()
            or candidate.stat().st_size != value[f"{kind}_size_bytes"]
            or _sha256(candidate) != value[f"{kind}_sha256"]
        ):
            raise ValueError(f"source command {kind} binding failed")


def _topology_facts(
    value: dict[str, object], runtime_dir: Path, expected_gpus: int | None
) -> tuple[Fact, ...]:
    node = value["node_rank"]
    prefix = f"parallel.node.{node}"
    outer = value["outer_mpi"]
    facts = [
        Fact(f"{prefix}.{name}", outer[name])
        for name in ("global_rank", "local_rank", "size", "hostname")
    ]
    facts.extend(
        Fact(f"{prefix}.{name}", value[name])
        for name in (
            "node_wrapper_argv",
            "singularity_argv",
            "sglang_argv",
            "cpu_affinity",
        )
    )
    if value["numa_mems"] is not None:
        facts.append(Fact(f"{prefix}.numa_mems", value["numa_mems"]))
    command = value["gpu_commands"]["container"]
    if command["exit_code"] != 0:
        raise ValueError("container GPU query failed")
    rows = _parse_gpu_csv(runtime_dir / command["stdout_member"])
    if expected_gpus is not None and len(rows) != expected_gpus:
        raise ValueError("container GPU cardinality differs")
    for gpu in rows:
        gpu_prefix = f"hardware.node.{node}.gpu.{gpu['index']}"
        facts.append(Fact(f"{gpu_prefix}.index", gpu["index"]))
        for name, child in gpu.items():
            if name != "index" and child is not None:
                facts.append(Fact(f"{gpu_prefix}.{name}", child))
        facts.append(Fact(f"{gpu_prefix}.scope", "container"))
    return tuple(facts)


def _parse_gpu_csv(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != len(_GPU_QUERY_FIELDS):
            raise ValueError("nvidia-smi CSV column count differs")
        index = _uint_lexeme(fields[0], "GPU index")
        memory_mib = _nullable_decimal(fields[4])
        memory_bytes = None
        if memory_mib is not None:
            converted = memory_mib * Decimal(1024 * 1024)
            if converted != converted.to_integral_value():
                raise ValueError("GPU memory does not convert to integral bytes")
            memory_bytes = int(converted)
        rows.append(
            {
                "index": index,
                "name": fields[1],
                "uuid": fields[2],
                "pci_bus_id": fields[3],
                "memory.total_bytes": memory_bytes,
                "driver_version": _nullable_text(fields[5]),
                "persistence_mode": _nullable_text(fields[6]),
                "compute_mode": _nullable_text(fields[7]),
                "power_limit_watt": _nullable_decimal(fields[8]),
                "clock.sm_max_mhz": _nullable_decimal(fields[9]),
                "clock.memory_max_mhz": _nullable_decimal(fields[10]),
                "mig_mode": _nullable_text(fields[11]),
                "ecc_mode": _nullable_text(fields[12]),
            }
        )
    indices = [row["index"] for row in rows]
    uuids = [row["uuid"] for row in rows]
    if indices != sorted(set(indices)) or len(uuids) != len(set(uuids)):
        raise ValueError("GPU indices/UUIDs are not ordered and unique")
    return rows


def _validate_lifecycle_mark(
    value: object, label: str, *, invocation: bool = False
) -> dict[str, object]:
    expected = {
        "sequence",
        "clock_id",
        "received_monotonic_ns",
        "ack_monotonic_ns",
    }
    if invocation:
        expected.add("invocation_id")
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"lifecycle {label} mark is not closed")
    _uint(value["sequence"], f"{label} sequence")
    _nonempty(value["clock_id"], f"{label} clock ID")
    received = _uint(value["received_monotonic_ns"], f"{label} received")
    ack = _uint(value["ack_monotonic_ns"], f"{label} ACK")
    if received > ack:
        raise ValueError(f"lifecycle {label} timestamps are reversed")
    if invocation:
        _nonempty(value["invocation_id"], "shutdown invocation ID")
    return value


def _lifecycle_mark_facts(node: int, kind: str, value: dict[str, object]) -> list[Fact]:
    mapping = {
        "sequence": "sequence",
        "clock_id": "clock_id",
        "received_monotonic_ns": "received_ns",
        "ack_monotonic_ns": "ack_ns",
        "invocation_id": "invocation_id",
    }
    return [
        Fact(f"lifecycle.node.{node}.{kind}.{mapping[name]}", child)
        for name, child in value.items()
    ]


def _duration_seconds(value: str) -> Decimal | None:
    if _is_null_lexeme(value):
        return None
    match = re.fullmatch(
        r"(?:(?P<days>[0-9]+)-)?(?P<hours>[0-9]+):(?P<minutes>[0-5][0-9]):(?P<seconds>[0-5][0-9](?:\.[0-9]+)?)(?: \((?P<total>[0-9]+)\))?",
        value,
    )
    if match is None:
        raise ValueError("invalid scheduler elapsed lexeme")
    elapsed = (
        Decimal(match.group("days") or 0) * 86400
        + Decimal(match.group("hours")) * 3600
        + Decimal(match.group("minutes")) * 60
        + Decimal(match.group("seconds"))
    )
    if match.group("total") is not None and elapsed != Decimal(match.group("total")):
        raise ValueError("scheduler elapsed display and total differ")
    return elapsed


def _nullable_int_lexeme(value: str) -> int | None:
    if _is_null_lexeme(value):
        return None
    if _INT.fullmatch(value) is None:
        raise ValueError("invalid scheduler integer lexeme")
    return int(value)


def _memory_bytes(value: str) -> int | None:
    if _is_null_lexeme(value):
        return None
    match = re.search(r"\(([0-9]+)(?:\s+bytes)?\)\s*$", value)
    if match is None:
        raise ValueError("invalid scheduler memory lexeme")
    return int(match.group(1))


def _milliseconds_seconds(value: str) -> Decimal | None:
    if _is_null_lexeme(value):
        return None
    numeric = value.removesuffix(" ms")
    if numeric == value:
        raise ValueError("invalid scheduler CPU-time lexeme")
    try:
        milliseconds = Decimal(numeric)
    except Exception as error:
        raise ValueError("invalid scheduler CPU-time lexeme") from error
    if not milliseconds.is_finite() or milliseconds < 0:
        raise ValueError("invalid scheduler CPU-time lexeme")
    return milliseconds / Decimal(1000)


def _is_null_lexeme(value: str) -> bool:
    return value in {"", "-", "N/A", "(null)"}


def _boost_settings() -> dict[str, object]:
    choices = (
        (Path("/sys/devices/system/cpu/cpufreq/boost"), False),
        (Path("/sys/devices/system/cpu/intel_pstate/no_turbo"), True),
    )
    for path, invert in choices:
        raw = _read_optional(path)
        if raw is None:
            continue
        if raw not in {"0", "1"}:
            return {
                "source_path": path.as_posix().lstrip("/"),
                "raw": raw,
                "enabled": None,
            }
        enabled = raw == "1"
        return {
            "source_path": path.as_posix().lstrip("/"),
            "raw": raw,
            "enabled": not enabled if invert else enabled,
        }
    return {"source_path": None, "raw": None, "enabled": None}


def _cpu_list_optional(path: Path, *, null_literal: bool = False) -> list[int] | None:
    raw = _read_optional(path, preserve_empty=True)
    if raw is None:
        return None
    if not raw or (null_literal and raw == "(null)"):
        return []
    values: set[int] = set()
    for part in raw.split(","):
        if re.fullmatch(r"[0-9]+", part):
            values.add(int(part))
            continue
        match = re.fullmatch(r"([0-9]+)-([0-9]+)", part)
        if match is None or int(match.group(1)) > int(match.group(2)):
            raise ValueError(f"invalid CPU list: {raw!r}")
        values.update(range(int(match.group(1)), int(match.group(2)) + 1))
    return sorted(values)


def _whitespace_uints(path: Path) -> list[int]:
    raw = path.read_text(encoding="ascii").strip()
    values = [_uint_lexeme(item, "related CPU") for item in raw.split()]
    if values != sorted(set(values)):
        raise ValueError("related CPUs are not sorted and unique")
    return values


def _khz_optional(path: Path) -> int | None:
    raw = _read_optional(path)
    return None if raw is None else _uint_lexeme(raw, "frequency") * 1000


def _meminfo() -> dict[str, str] | None:
    try:
        result = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, separator, value = line.partition(":")
            if separator:
                result[key] = value.strip()
        return result
    except OSError:
        return None


def _meminfo_bytes(values: dict[str, str] | None, name: str) -> int | None:
    if values is None or name not in values:
        return None
    match = re.fullmatch(r"([0-9]+)\s+kB", values[name])
    if match is None:
        return None
    return int(match.group(1)) * 1024


def _meminfo_uint(values: dict[str, str] | None, name: str) -> int | None:
    if values is None or name not in values or _UINT.fullmatch(values[name]) is None:
        return None
    return int(values[name])


def _bool_file(path: Path) -> bool | None:
    raw = _read_optional(path)
    return None if raw not in {"0", "1"} else raw == "1"


def _read_optional(path: Path, *, preserve_empty: bool = False) -> str | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    return value if value or preserve_empty else None


def _numa_mems() -> str | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("Mems_allowed_list:"):
                value = line.partition(":")[2].strip()
                return value or None
    except OSError:
        return None
    return None


def _member(path: Path, relative: str) -> dict[str, object]:
    _relative(relative, "member path")
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_member(value: object) -> str:
    if not isinstance(value, dict) or set(value) != {"path", "size_bytes", "sha256"}:
        raise ValueError("member is not closed")
    relative = _relative(value["path"], "member path")
    _uint(value["size_bytes"], "member size")
    if (
        not isinstance(value["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
    ):
        raise ValueError("member SHA-256 is invalid")
    return relative


def _bind_member(path: Path, value: object) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != value["size_bytes"]
        or _sha256(path) != value["sha256"]
    ):
        raise ValueError("member hash binding failed")


def _exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_bytes(path, jcs_bytes(value) + b"\n")


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    try:
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clock_id() -> str:
    boot_id = (
        Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    )
    return f"{boot_id}:CLOCK_MONOTONIC"


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _rfc3339(value: object, label: str) -> datetime:
    text = _nonempty(value, label)
    if not text.endswith("Z"):
        raise ValueError(f"{label} must be UTC RFC3339")
    try:
        return datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be UTC RFC3339") from error


def _escape_environment_name(value: bytes) -> str:
    return (
        value.decode("ascii")
        if _SAFE_KEY.fullmatch(value)
        else f"encoded.{value.hex()}"
    )


def _relative(value: object, label: str) -> str:
    text = _nonempty(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or str(path) != text:
        raise ValueError(f"{label} must be a safe relative POSIX path")
    return text


def _string_array(value: object, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{label} must be a nonempty string array")
    return value


def _uint_array(value: object, label: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return [_uint(item, label) for item in value]


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _uint(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _decimal(value: object, label: str) -> Decimal:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a decimal string")
    try:
        result = Decimal(value)
    except Exception as error:
        raise ValueError(f"{label} is not decimal") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _uint_lexeme(value: str, label: str) -> int:
    if _UINT.fullmatch(value) is None:
        raise ValueError(f"{label} must be an unsigned integer")
    return int(value)


def _nullable_text(value: str) -> str | None:
    return None if _is_null_lexeme(value) else value


def _nullable_decimal(value: str) -> Decimal | None:
    if _is_null_lexeme(value):
        return None
    try:
        result = Decimal(value)
    except Exception as error:
        raise ValueError("invalid decimal source value") from error
    if not result.is_finite() or result < 0:
        raise ValueError("invalid decimal source value")
    return result
