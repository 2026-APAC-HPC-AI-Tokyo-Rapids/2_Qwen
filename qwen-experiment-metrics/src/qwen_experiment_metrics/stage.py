from __future__ import annotations

import csv
import json
import math
import os
import signal
import socket
import statistics
import struct
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from .core import Metric, flatten, send, warning

RUNTIME_FIELDS = (
    "index",
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
    "pstate",
)
RESULT_KEYS = {
    "backend": "result.benchmark.backend",
    "successful_requests": "result.benchmark.successful_requests",
    "total_latency": "perf.benchmark.generation.wall_time_sec",
    "total_input_tokens": "result.token.input.total",
    "total_output_tokens": "result.token.output.total",
    "request_throughput": "perf.throughput.request_per_sec",
    "input_throughput": "perf.throughput.input_token_per_sec",
    "output_throughput": "perf.throughput.output_token_per_sec",
    "total_throughput": "perf.throughput.total_token_per_sec",
    "last_gen_throughput": "perf.throughput.last_generation_token_per_sec",
}


@dataclass(frozen=True, slots=True)
class Sample:
    timestamp: float
    values: dict[str, str]


class EventReceiver(threading.Thread):
    def __init__(self, path: Path, run: str):
        super().__init__(daemon=True)
        self.path = path
        self.run_name = run
        self.stop_requested = threading.Event()
        self.ready = threading.Event()
        self.received: set[str] = set()

    def run(self) -> None:
        self.path.unlink(missing_ok=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(self.path))
            server.listen(4)
            server.settimeout(0.2)
            self.ready.set()
            while not self.stop_requested.is_set():
                try:
                    connection, _ = server.accept()
                except TimeoutError:
                    continue
                except OSError as error:
                    warning(f"SGLang event accept failed: {error}")
                    continue
                with connection:
                    try:
                        header = _read_exact(connection, 4)
                        size = struct.unpack(">I", header)[0]
                        if size > 64 * 1024 * 1024:
                            raise ValueError("SGLang event exceeds 64 MiB")
                        event = json.loads(_read_exact(connection, size))
                        if isinstance(event, dict) and isinstance(
                            event.get("type"), str
                        ):
                            self.received.add(event["type"])
                        send(self.run_name, event_metrics(event))
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        warning(f"invalid SGLang event: {error}")
        self.path.unlink(missing_ok=True)

    def stop(self) -> None:
        self.stop_requested.set()


class GpuSampler(threading.Thread):
    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_requested = threading.Event()
        self.samples: dict[str, list[Sample]] = defaultdict(list)

    def run(self) -> None:
        deadline = time.monotonic()
        while not self.stop_requested.is_set():
            self._poll()
            deadline += self.interval
            self.stop_requested.wait(max(0.0, deadline - time.monotonic()))

    def _poll(self) -> None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={','.join(RUNTIME_FIELDS)}",
                    "--format=csv,noheader,nounits",
                ],
                env={**os.environ, "LC_ALL": "C"},
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=max(1.0, self.interval),
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            warning(f"nvidia-smi sampler failed: {error}")
            return
        if completed.returncode != 0:
            warning(f"nvidia-smi sampler returned {completed.returncode}")
            return
        timestamp = time.monotonic()
        for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
            values = dict(zip(RUNTIME_FIELDS, (value.strip() for value in row)))
            uuid = values.get("uuid", "")
            if uuid:
                self.samples[uuid].append(Sample(timestamp, values))

    def stop(self) -> None:
        self.stop_requested.set()


def _read_exact(stream: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        block = stream.recv(size - len(result))
        if not block:
            raise OSError("unexpected end of event stream")
        result.extend(block)
    return bytes(result)


def event_metrics(event: object) -> list[Metric]:
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise TypeError("event must contain a string type")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise TypeError("event payload must be an object")
    if event["type"] == "sglang.configuration":
        result: list[Metric] = []
        for source, prefix in (
            ("bench_args", "sglang.benchmark_arg"),
            ("server_args", "sglang.server_arg"),
            ("environment", "sglang.env"),
        ):
            if source in payload:
                result.extend(flatten(prefix, payload[source]))
        return result
    if event["type"] == "sglang.benchmark_result":
        return [
            Metric(target, str(payload.get(source, "")))
            for source, target in RESULT_KEYS.items()
        ]
    raise ValueError(f"unknown event type: {event['type']}")


def _number(value: str) -> float | None:
    if value in {"", "N/A", "[N/A]", "Not Supported"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def aggregate_gpu(samples: dict[str, list[Sample]], interval: float) -> list[Metric]:
    result: list[Metric] = []
    if not samples:
        return [Metric("resource.stage.benchmark.gpu.sample_count", "0")]
    for logical, (uuid, device_samples) in enumerate(samples.items()):
        prefix = f"resource.stage.benchmark.gpu.{logical}"
        result.extend(
            [
                Metric(f"{prefix}.uuid", uuid),
                Metric(f"{prefix}.sample_count", str(len(device_samples))),
                Metric(f"{prefix}.sample_interval_sec", repr(interval)),
                Metric(
                    f"{prefix}.sampled_duration_sec",
                    repr(device_samples[-1].timestamp - device_samples[0].timestamp)
                    if len(device_samples) > 1
                    else "0.0",
                ),
            ]
        )
        specifications = (
            (
                "utilization.gpu",
                "utilization.gpu_percent",
                ("mean", "max", "p50", "p95"),
            ),
            ("utilization.memory", "utilization.memory_percent", ("mean", "max")),
            ("memory.used", "memory.used_bytes", ("mean", "max")),
            ("power.draw", "power.draw_watt", ("mean", "max")),
            ("temperature.gpu", "temperature_celsius", ("max",)),
            ("clocks.sm", "clock.sm_mhz", ("mean", "min")),
            ("clocks.mem", "clock.memory_mhz", ("mean",)),
        )
        for source, target, operations in specifications:
            values = [
                value
                for sample in device_samples
                if (value := _number(sample.values.get(source, ""))) is not None
            ]
            if source == "memory.used":
                values = [value * 1024 * 1024 for value in values]
            for operation in operations:
                if not values:
                    value = ""
                elif operation == "mean":
                    value = repr(statistics.fmean(values))
                elif operation == "max":
                    value = repr(max(values))
                elif operation == "min":
                    value = repr(min(values))
                else:
                    percentile = _percentile(
                        values, 0.50 if operation == "p50" else 0.95
                    )
                    value = "" if percentile is None else repr(percentile)
                result.append(Metric(f"{prefix}.{target}.{operation}", value))
        energy = 0.0
        observed = False
        for left, right in pairwise(device_samples):
            left_power = _number(left.values.get("power.draw", ""))
            right_power = _number(right.values.get("power.draw", ""))
            if left_power is not None and right_power is not None:
                energy += (
                    (left_power + right_power) / 2 * (right.timestamp - left.timestamp)
                )
                observed = True
        result.append(
            Metric(f"{prefix}.energy_joule", repr(energy) if observed else "")
        )
        states: dict[str, int] = defaultdict(int)
        for sample in device_samples:
            state = sample.values.get("pstate", "")
            if state and state not in {"N/A", "[N/A]"}:
                states[state.lower()] += 1
        result.extend(
            Metric(f"{prefix}.pstate.{state}.sample_count", str(count))
            for state, count in sorted(states.items())
        )
    return result


def _node_gpu_metrics(
    node_rank: int,
    hostname: str,
    static_metrics: list[Metric],
    runtime_metrics: list[Metric],
) -> list[Metric]:
    result = [Metric(f"hardware.node.{node_rank}.hostname", hostname)]
    for metric in static_metrics:
        if metric.key == "hardware.gpu.count":
            key = f"hardware.node.{node_rank}.gpu.count"
        elif metric.key.startswith("hardware.gpu."):
            key = metric.key.replace(
                "hardware.gpu.", f"hardware.node.{node_rank}.gpu.", 1
            )
        else:
            continue
        result.append(Metric(key, metric.value))
    for metric in runtime_metrics:
        if metric.key.startswith("resource.stage.benchmark.gpu."):
            key = metric.key.replace(
                "resource.stage.benchmark.gpu.",
                f"resource.stage.benchmark.node.{node_rank}.gpu.",
                1,
            )
            result.append(Metric(key, metric.value))
    return result


def monitor_node_gpu(node_rank: int, output: Path, ready: Path) -> int:
    from .collectors import static_gpu

    stop_requested = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    old_handlers = {
        signum: signal.signal(signum, stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    sampler = GpuSampler()
    static_metrics = static_gpu()
    sampler.start()
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.touch()
    try:
        while not stop_requested.wait(0.2):
            pass
    finally:
        sampler.stop()
        sampler.join(timeout=5)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        metrics = _node_gpu_metrics(
            node_rank,
            socket.gethostname(),
            static_metrics,
            aggregate_gpu(sampler.samples, sampler.interval),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                [{"key": metric.key, "value": metric.value} for metric in metrics],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        temporary.replace(output)
        ready.unlink(missing_ok=True)
    return 0


def load_node_gpu_metrics(directory: Path) -> list[Metric]:
    result: list[Metric] = []
    total_gpu_count = 0
    observed_nodes = 0
    for path in sorted(directory.glob("node-rank-*-gpu-metrics.json")):
        try:
            payload = json.loads(path.read_text())
            metrics = [Metric(item["key"], item["value"]) for item in payload]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            warning(f"could not read node GPU metrics from {path}: {error}")
            continue
        observed_nodes += 1
        result.extend(metrics)
        for metric in metrics:
            if metric.key.endswith(".gpu.count"):
                try:
                    total_gpu_count += int(metric.value)
                except ValueError:
                    pass
        path.unlink(missing_ok=True)
    expected_nodes = 0
    try:
        expected_nodes = int(os.environ.get("PJM_NODE", "0"))
    except ValueError:
        pass
    complete = expected_nodes in {0, observed_nodes}
    result.extend(
        [
            Metric("hardware.node_count.observed", str(observed_nodes)),
            Metric("hardware.gpu.count.observed", str(total_gpu_count)),
            Metric("hardware.gpu.count", str(total_gpu_count) if complete else ""),
        ]
    )
    return result


def run_stage(
    run: str,
    name: str,
    socket_dir: Path,
    command: list[str],
    node_gpu_dir: Path | None = None,
) -> int:
    socket_dir.mkdir(parents=True, exist_ok=True)
    socket_path = socket_dir / "events.sock"
    receiver = EventReceiver(socket_path, run)
    sampler = GpuSampler() if node_gpu_dir is None else None
    receiver.start()
    if not receiver.ready.wait(5):
        warning("SGLang event receiver did not become ready")
    if sampler is not None:
        sampler.start()
    started = time.monotonic_ns()
    process: subprocess.Popen[bytes] | None = None

    def forward(signum: int, _frame: object) -> None:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signum)

    old_handlers = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        process = subprocess.Popen(command, start_new_session=True)
        returncode = process.wait()
    except OSError as error:
        warning(f"stage {name} could not start: {error}")
        returncode = 127
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if sampler is not None:
            sampler.stop()
            sampler.join(timeout=5)
        receiver.stop()
        # The receiver may still be completing the last BigQuery request.  Wait
        # for it so the main thread never uses the credentials concurrently.
        receiver.join()
    elapsed = (time.monotonic_ns() - started) / 1_000_000_000
    if node_gpu_dir is None:
        assert sampler is not None
        send(run, aggregate_gpu(sampler.samples, sampler.interval))
    else:
        send(run, load_node_gpu_metrics(node_gpu_dir))
    if "sglang.benchmark_result" not in receiver.received:
        send(run, fallback_result(Path("results/results.jsonl")))
    stage_metrics = [
        Metric(f"perf.stage.{name}.wall_time_sec", repr(elapsed)),
        Metric(f"run.stage.{name}.exit_code", str(returncode)),
        Metric(
            f"run.stage.{name}.result_present",
            str((Path("results") / "results.jsonl").is_file()).lower(),
        ),
    ]
    if returncode < 0:
        stage_metrics.append(Metric(f"run.stage.{name}.signal", str(abs(returncode))))
    send(run, stage_metrics)
    return 128 + abs(returncode) if returncode < 0 else returncode


def fallback_result(path: Path) -> list[Metric]:
    payload: dict[str, object] = {}
    try:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if lines:
            candidate = json.loads(lines[-1])
            if isinstance(candidate, dict):
                payload = candidate
    except (OSError, json.JSONDecodeError) as error:
        warning(f"could not read benchmark result fallback: {error}")
    return [
        Metric(target, str(payload.get(source, "")))
        for source, target in RESULT_KEYS.items()
    ]
