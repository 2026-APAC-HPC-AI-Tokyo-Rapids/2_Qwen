from __future__ import annotations

import base64
import csv
import json
import os
import platform
import re
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .core import Metric, flatten

ENV_NAME = re.compile(rb"^[A-Za-z][A-Za-z0-9_]*$")
STATIC_FIELDS = (
    "index",
    "name",
    "uuid",
    "pci.bus_id",
    "compute_cap",
    "memory.total",
    "power.limit",
    "mig.mode.current",
    "driver_version",
)


def _command(argv: list[str], timeout: float = 60) -> tuple[int, str]:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            argv,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="strict",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return 127, ""


def run_start(command: list[str]) -> list[Metric]:
    result = [Metric("run.started_at", datetime.now(UTC).isoformat())]
    if command:
        result.append(Metric("run.command", shlex.join(command)))
    return result


def run_end(exit_code: int, elapsed_sec: str | None) -> list[Metric]:
    result = [
        Metric("run.finished_at", datetime.now(UTC).isoformat()),
        Metric("run.outcome", "succeeded" if exit_code == 0 else "failed"),
        Metric("run.exit_code", str(exit_code)),
    ]
    if elapsed_sec is not None:
        result.append(Metric("run.elapsed_sec", elapsed_sec))
    if exit_code < 0:
        result.append(Metric("run.signal", str(abs(exit_code))))
    return result


def platform_metrics() -> list[Metric]:
    result: list[Metric] = []
    for name, value in os.environb.items():
        name_token = (
            name.decode("ascii")
            if ENV_NAME.fullmatch(name)
            else f"encoded.{name.hex()}"
        )
        try:
            decoded = value.decode("utf-8")
            key = f"env.{name_token}"
        except UnicodeDecodeError:
            decoded = base64.b64encode(value).decode("ascii")
            key = f"env.bytes.{name_token}"
        result.append(Metric(key, decoded))
    model_path = os.environ.get("MODEL_PATH", "")
    result.append(Metric("model.name", Path(model_path).name if model_path else ""))
    os_release = platform.freedesktop_os_release()
    result.extend(
        [
            Metric("system.cluster", "genkai"),
            Metric("system.os.id", os_release.get("ID", "")),
            Metric("system.os.version_id", os_release.get("VERSION_ID", "")),
            Metric("system.kernel.release", os.uname().release),
            Metric("system.architecture", os.uname().machine),
        ]
    )
    code, output = _command(["lscpu", "--json"])
    fields: dict[str, str] = {}
    if code == 0:
        try:
            fields = {
                item.get("field", "").removesuffix(":"): str(item.get("data", ""))
                for item in json.loads(output).get("lscpu", [])
            }
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    for source, key in {
        "Model name": "hardware.cpu.model",
        "Socket(s)": "hardware.cpu.socket_count",
        "Core(s) per socket": "hardware.cpu.cores_per_socket",
        "Thread(s) per core": "hardware.cpu.threads_per_core",
        "NUMA node(s)": "hardware.cpu.numa_node_count",
    }.items():
        result.append(Metric(key, fields.get(source, "")))
    try:
        affinity = str(len(os.sched_getaffinity(0)))
    except OSError:
        affinity = ""
    result.append(Metric("hardware.cpu.available_logical_count", affinity))
    memory = ""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory = str(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    result.append(Metric("hardware.memory.total_bytes", memory))
    return result


def static_gpu() -> list[Metric]:
    code, output = _command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(STATIC_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_static_gpu(output) if code == 0 else [Metric("hardware.gpu.count", "")]


def parse_static_gpu(output: str) -> list[Metric]:
    keys = {
        "index": "logical_index",
        "name": "name",
        "uuid": "uuid",
        "pci.bus_id": "pci_bus_id",
        "compute_cap": "compute_capability",
        "memory.total": "memory.total_bytes",
        "power.limit": "power.limit_watt",
        "mig.mode.current": "mig.mode.current",
        "driver_version": "driver_version",
    }
    rows = list(csv.reader(output.splitlines(), skipinitialspace=True))
    result = [Metric("hardware.gpu.count", str(len(rows)))]
    for logical, row in enumerate(rows):
        values = dict(zip(STATIC_FIELDS, (item.strip() for item in row)))
        for source, suffix in keys.items():
            value = values.get(source, "")
            if value in {"N/A", "[N/A]", "Not Supported"}:
                value = ""
            if source == "memory.total" and value:
                try:
                    value = str(int(float(value) * 1024 * 1024))
                except ValueError:
                    value = ""
            result.append(Metric(f"hardware.gpu.{logical}.{suffix}", value))
    return result


def source_artifacts(runtime_dir: Path) -> list[Metric]:
    manifest = runtime_dir / "metrics-artifacts.json"
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        payload = {}
    result = flatten("artifact", payload)
    commit = ""
    code, output = _command(
        [
            "singularity",
            "exec",
            str(runtime_dir / "sglang.sif"),
            "cat",
            "/opt/sglang/sglang-commit.txt",
        ]
    )
    if code == 0:
        commit = output.strip().splitlines()[0] if output.strip() else ""
    result.append(Metric("software.sglang.commit", commit))
    code, output = _command(
        [
            "singularity",
            "exec",
            str(runtime_dir / "sglang.sif"),
            "/opt/sglang/venv/bin/python",
            "-c",
            (
                "import json,platform,torch,sglang;"
                "print(json.dumps({'sglang':getattr(sglang,'__version__',''),"
                "'python':platform.python_version(),'torch':torch.__version__,"
                "'cuda_runtime':torch.version.cuda or ''}))"
            ),
        ]
    )
    versions: dict[str, object] = {}
    if code == 0:
        try:
            versions = json.loads(output)
        except json.JSONDecodeError:
            pass
    for source, key in (
        ("sglang", "software.sglang.version"),
        ("python", "software.python.version"),
        ("torch", "software.torch.version"),
        ("cuda_runtime", "software.cuda.runtime_version"),
    ):
        result.append(Metric(key, str(versions.get(source, ""))))
    code, output = _command(["singularity", "--version"])
    result.append(
        Metric(
            "software.singularity.version",
            output.strip().splitlines()[0] if code == 0 and output.strip() else "",
        )
    )
    return result


def result_artifacts(runtime_dir: Path) -> list[Metric]:
    result: list[Metric] = []
    for relative in (
        "results/environment.log",
        "results/throughput.log",
        "results/results.jsonl",
    ):
        path = runtime_dir / relative
        prefix = f"artifact.{relative.replace('/', '.')}"
        try:
            size = str(path.stat().st_size)
        except OSError:
            size = ""
        result.extend(
            [
                Metric(f"{prefix}.path", relative),
                Metric(f"{prefix}.size_bytes", size),
            ]
        )
    return result


def log_artifact(runtime_dir: Path, job_id: str) -> list[Metric]:
    relative = f"logs/{job_id}.log"
    path = runtime_dir / relative
    try:
        size = str(path.stat().st_size)
    except OSError:
        size = ""
    return [
        Metric("artifact.pjm_log.path", relative),
        Metric("artifact.pjm_log.size_bytes", size),
    ]


PJM_FIELDS = (
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


def scheduler(job_id: str) -> list[Metric]:
    code, output = _command(
        [
            "pjstat",
            "--history",
            "-S",
            "--choose",
            ",".join(PJM_FIELDS),
            "--data",
            "--delimiter",
            "\t",
            job_id,
        ]
    )
    source: dict[str, str] = {}
    lines = [line for line in output.splitlines() if line.strip()]
    if code == 0 and len(lines) >= 2:
        values = lines[-1].split("\t")
        if values and not values[0]:
            values = values[1:]
        source = dict(zip(PJM_FIELDS, values))

    def parenthetical(value: str) -> str:
        return (
            value.rsplit("(", 1)[1].split(")", 1)[0].strip()
            if "(" in value and ")" in value
            else ""
        )

    def milliseconds(value: str) -> str:
        try:
            return repr(float(value) / 1000)
        except ValueError:
            return ""

    started = ""
    try:
        started = datetime.strptime(
            f"{source.get('sdt', '').strip()} +0900", "%Y/%m/%d %H:%M:%S %z"
        ).isoformat()
    except ValueError:
        pass
    values = {
        "scheduler.state": source.get("st", ""),
        "scheduler.started_at": started,
        "scheduler.elapsed_sec": parenthetical(source.get("elp", "")),
        "scheduler.exit_code": source.get("ec", ""),
        "scheduler.signal": "" if source.get("sn", "") in {"", "0"} else source["sn"],
        "scheduler.pjm_code": source.get("pc", ""),
        "scheduler.node_count.allocated": source.get("nnumu", ""),
        "resource.job.memory.max_rss_bytes": parenthetical(source.get("mmszu", "")),
        "resource.job.cpu.user_time_sec": milliseconds(source.get("uctmut", "")),
        "resource.job.cpu.system_time_sec": milliseconds(source.get("sctmut", "")),
    }
    return [Metric(key, value) for key, value in values.items()]
