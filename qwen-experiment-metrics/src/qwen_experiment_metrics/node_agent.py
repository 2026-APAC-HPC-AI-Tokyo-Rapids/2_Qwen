from __future__ import annotations

import argparse
import os
import select
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from .canonical import jcs_bytes, load_json_strict
from .manifest import load_manifest
from .recording import Recorder
from .sources import ExecutionSettings, load_execution_settings
from .system_sources import (
    validate_lifecycle_rendezvous,
    validate_lifecycle_summary,
    validate_node_lifecycle,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qwen-node-agent")
    root.add_argument("--manifest", type=Path, required=True)
    root.add_argument("--runtime-dir", type=Path, required=True)
    root.add_argument("--node-rank", type=int, required=True)
    root.add_argument("--master-host", required=True)
    root.add_argument("--phase-socket", type=Path)
    root.add_argument("--registry", type=Path)
    root.add_argument("command", nargs=argparse.REMAINDER)
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    command = arguments.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("node child command is empty")
    manifest = load_manifest(arguments.manifest)
    settings = load_execution_settings(
        arguments.runtime_dir / "control/execution-settings.json"
    )
    if arguments.node_rank < 0 or arguments.node_rank >= settings.nodes:
        raise ValueError("node rank is outside the execution grid")
    if arguments.node_rank == 0:
        return _coordinator(arguments, command, manifest.run_id, settings)
    return _participant(arguments, command, settings)


def _coordinator(
    arguments: argparse.Namespace,
    command: list[str],
    run_id: str,
    settings: ExecutionSettings,
) -> int:
    port = settings.node_agent_port
    timeout = settings.rendezvous_timeout_sec
    connections: dict[int, socket.socket] = {}
    ready_marks = {0: _local_mark(0)}
    lifecycle_dir = arguments.runtime_dir / "control/lifecycle"
    lifecycle_dir.mkdir(parents=True, exist_ok=True)
    rendezvous = {
        "schema_version": "lifecycle-rendezvous/v1",
        "run_id": run_id,
        "rank0_hostname": socket.gethostname(),
        "address": arguments.master_host,
        "port": port,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        server.bind(("", port))
        _exclusive_json(lifecycle_dir / "rendezvous.json", rendezvous)
        server.listen(max(settings.nodes - 1, 1))
        server.settimeout(timeout)
        while len(connections) < settings.nodes - 1:
            connection, _ = server.accept()
            hello = _receive(connection, timeout)
            node_rank = _uint(hello.get("node_rank"), "node rank")
            if (
                node_rank == 0
                or node_rank >= settings.nodes
                or node_rank in connections
            ):
                connection.close()
                raise ValueError("node-agent ready rank is invalid or duplicated")
            _send(connection, {"type": "ready", "sequence": 0})
            response = _receive(connection, timeout)
            ready_marks[node_rank] = _mark_from_wire(response, sequence=0)
            connections[node_rank] = connection

    phase: _PhaseCoordinator | None = None
    if settings.run_purpose == "profile":
        if arguments.phase_socket is None:
            raise ValueError("profile coordinator requires a phase socket")
        phase = _PhaseCoordinator(
            arguments.phase_socket,
            arguments.runtime_dir,
            run_id,
            connections,
            settings,
        )
        phase.start()
    raw_status = _run_child(command, arguments.runtime_dir, 0)
    exit_code = _shell_status(raw_status)
    phase_error = phase.finish() if phase is not None else None
    shutdown_marks = {0: _local_mark(1, invocation_id="measured-generation")}
    for node_rank, connection in sorted(connections.items()):
        try:
            _send(
                connection,
                {
                    "type": "shutdown",
                    "sequence": 1,
                    "invocation_id": "measured-generation",
                    "exit_code": exit_code,
                },
            )
            response = _receive(connection, settings.ready_timeout_sec)
            shutdown_marks[node_rank] = _mark_from_wire(
                response, sequence=1, invocation_id="measured-generation"
            )
        finally:
            connection.close()
    summary = {
        "schema_version": "lifecycle-summary/v1",
        "run_id": run_id,
        "nodes": [
            {
                "node_rank": node,
                "ready": ready_marks[node],
                "shutdown": shutdown_marks.get(node),
            }
            for node in range(settings.nodes)
        ],
    }
    _exclusive_json(lifecycle_dir / "summary.json", summary)
    _record_node(arguments, 0)
    facts = (
        *validate_lifecycle_rendezvous(
            lifecycle_dir / "rendezvous.json", run_id, settings.node_agent_port
        ),
        *validate_lifecycle_summary(
            lifecycle_dir / "summary.json", run_id, settings.nodes
        ),
    )
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        "lifecycle-control",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record_many(facts)
    if phase_error is not None:
        raise phase_error
    return exit_code


def _participant(
    arguments: argparse.Namespace, command: list[str], settings: ExecutionSettings
) -> int:
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    deadline = time.monotonic() + settings.rendezvous_timeout_sec
    while True:
        try:
            connection.connect((arguments.master_host, settings.node_agent_port))
            break
        except (ConnectionRefusedError, TimeoutError):
            if time.monotonic() >= deadline:
                connection.close()
                raise TimeoutError("node-agent rendezvous timed out")
            time.sleep(0.1)
    _send(connection, {"node_rank": arguments.node_rank})
    ready = _receive(connection, settings.ready_timeout_sec)
    if ready != {"type": "ready", "sequence": 0}:
        raise ValueError("node-agent ready message differs")
    _send(connection, _local_mark(0))

    attempt = time.monotonic_ns()
    try:
        child_pid = os.fork()
    except OSError:
        _write_lifecycle(
            arguments.runtime_dir, arguments.node_rank, attempt, None, None, None
        )
        _record_node(arguments, arguments.node_rank)
        raise
    if child_pid == 0:
        os.setsid()
        os.execvpe(command[0], command, os.environ)
    started = time.monotonic_ns()
    raw_status: int | None = None
    shutdown_exit: int | None = None
    shutdown_deadline: float | None = None
    while shutdown_exit is None:
        if raw_status is None:
            waited, status = os.waitpid(child_pid, os.WNOHANG)
            if waited:
                raw_status = status
                shutdown_deadline = time.monotonic() + settings.ready_timeout_sec
        readable, _, _ = select.select([connection], [], [], 0.2)
        if readable:
            shutdown = _receive(connection, settings.ready_timeout_sec)
            if shutdown.get("type") in {"phase_start", "phase_end"}:
                _handle_phase_control(connection, shutdown)
                continue
            if (
                shutdown.get("type") != "shutdown"
                or shutdown.get("sequence") != 1
                or shutdown.get("invocation_id") != "measured-generation"
            ):
                raise ValueError("node-agent shutdown message differs")
            shutdown_exit = _uint(shutdown.get("exit_code"), "shutdown exit code")
            _send(
                connection,
                _local_mark(1, invocation_id="measured-generation"),
            )
            if raw_status is None:
                os.killpg(child_pid, signal.SIGTERM)
        if shutdown_deadline is not None and time.monotonic() >= shutdown_deadline:
            raise TimeoutError("node-agent shutdown timed out after child exit")
    connection.close()
    if raw_status is None:
        _, raw_status = os.waitpid(child_pid, 0)
    finished = time.monotonic_ns()
    _write_lifecycle(
        arguments.runtime_dir,
        arguments.node_rank,
        attempt,
        started,
        finished,
        raw_status,
    )
    _record_node(arguments, arguments.node_rank)
    return shutdown_exit


class _PhaseCoordinator:
    def __init__(
        self,
        socket_path: Path,
        runtime_dir: Path,
        run_id: str,
        connections: dict[int, socket.socket],
        settings: ExecutionSettings,
    ) -> None:
        if not socket_path.is_absolute():
            raise ValueError("phase socket must be absolute")
        self.socket_path = socket_path
        self.runtime_dir = runtime_dir
        self.run_id = run_id
        self.connections = connections
        self.settings = settings
        self.stop_requested = threading.Event()
        self.error: BaseException | None = None
        self.complete = False
        self._active_connection: socket.socket | None = None
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise FileExistsError(self.socket_path)
        self.server.bind(str(self.socket_path))
        self.server.listen(1)
        self.server.settimeout(0.2)
        self.thread.start()

    def finish(self) -> BaseException | None:
        self.thread.join(timeout=1)
        if self.thread.is_alive():
            self.stop_requested.set()
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as wake:
                    wake.connect(str(self.socket_path))
            except OSError:
                pass
            self.thread.join(timeout=2)
        try:
            self.server.close()
        finally:
            self.socket_path.unlink(missing_ok=True)
        if self.thread.is_alive():
            return RuntimeError("phase coordinator did not stop")
        if self.error is not None:
            return self.error
        if not self.complete:
            return RuntimeError("profile phase summary was not completed")
        return None

    def _run(self) -> None:
        try:
            start = self._request("phase_start")
            self._closed_request(start, "phase_start", sequence=0)
            invocation_id = str(start["invocation_id"])
            kind = str(start["kind"])
            start_broadcast = time.monotonic_ns()
            start_marks = {0: _local_mark(0)}
            for node_rank, connection in sorted(self.connections.items()):
                _send(connection, start)
                start_marks[node_rank] = _mark_from_wire(
                    _receive(connection, self.settings.ready_timeout_sec), sequence=0
                )
            all_start_acks = time.monotonic_ns()
            self._reply(
                {
                    "type": "phase_start_ack",
                    "invocation_id": invocation_id,
                    "kind": kind,
                }
            )

            end = self._request("phase_end")
            self._closed_request(end, "phase_end", sequence=1)
            if end["invocation_id"] != invocation_id or end["kind"] != kind:
                raise ValueError("phase invocation identity changed")
            generation_start = _uint(end.get("generation_start_ns"), "generation start")
            generation_end = _uint(end.get("generation_end_ns"), "generation end")
            if not all_start_acks <= generation_start <= generation_end:
                raise ValueError("phase generation timestamps are reversed")
            end_broadcast = time.monotonic_ns()
            end_marks = {0: _local_mark(1)}
            remote_end = {
                "type": "phase_end",
                "sequence": 1,
                "invocation_id": invocation_id,
                "kind": kind,
            }
            for node_rank, connection in sorted(self.connections.items()):
                _send(connection, remote_end)
                end_marks[node_rank] = _mark_from_wire(
                    _receive(connection, self.settings.ready_timeout_sec), sequence=1
                )
            all_end_acks = time.monotonic_ns()
            summary = {
                "schema_version": "phase-summary/v1",
                "run_id": self.run_id,
                "invocation_id": invocation_id,
                "kind": kind,
                "rank0": {
                    "clock_id": _clock_id(),
                    "start_broadcast_ns": start_broadcast,
                    "all_start_acks_received_ns": all_start_acks,
                    "generation_start_ns": generation_start,
                    "generation_end_ns": generation_end,
                    "end_broadcast_ns": end_broadcast,
                    "all_end_acks_received_ns": all_end_acks,
                },
                "nodes": [
                    {
                        "node_rank": node,
                        "clock_id": start_marks[node]["clock_id"],
                        "start_sequence": start_marks[node]["sequence"],
                        "end_sequence": end_marks[node]["sequence"],
                        "start_received_ns": start_marks[node]["received_monotonic_ns"],
                        "start_ack_ns": start_marks[node]["ack_monotonic_ns"],
                        "end_received_ns": end_marks[node]["received_monotonic_ns"],
                        "end_ack_ns": end_marks[node]["ack_monotonic_ns"],
                    }
                    for node in range(self.settings.nodes)
                ],
            }
            destination = self.runtime_dir / "control/phase/summary.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            _exclusive_json(destination, summary)
            self._reply(
                {
                    "type": "phase_end_ack",
                    "invocation_id": invocation_id,
                    "kind": kind,
                }
            )
            self.complete = True
        except Exception as error:  # noqa: BLE001 - propagated across thread boundary
            if not self.stop_requested.is_set():
                self.error = error
        finally:
            if self._active_connection is not None:
                self._active_connection.close()
                self._active_connection = None
            self.server.close()
            self.socket_path.unlink(missing_ok=True)

    def _request(self, expected_type: str) -> dict[str, object]:
        while not self.stop_requested.is_set():
            try:
                connection, _ = self.server.accept()
            except TimeoutError:
                continue
            try:
                self._active_connection = connection
                value = _receive(connection, self.settings.ready_timeout_sec)
                if value.get("type") != expected_type:
                    raise ValueError(f"expected {expected_type} phase request")
                return value
            except Exception:
                connection.close()
                self._active_connection = None
                raise
        raise RuntimeError("phase coordinator stopped")

    def _reply(self, value: dict[str, object]) -> None:
        if self._active_connection is None:
            raise RuntimeError("phase reply has no active request")
        connection = self._active_connection
        self._active_connection = None
        try:
            _send(connection, value)
        finally:
            connection.close()

    @staticmethod
    def _closed_request(
        value: dict[str, object], expected_type: str, *, sequence: int
    ) -> None:
        expected = {"type", "sequence", "invocation_id", "kind"}
        if expected_type == "phase_end":
            expected.update({"generation_start_ns", "generation_end_ns"})
        if (
            set(value) != expected
            or value["type"] != expected_type
            or value["sequence"] != sequence
            or value["invocation_id"] != "measured-generation"
            or value["kind"] != "measured"
        ):
            raise ValueError("phase request is not the fixed measured invocation")


def _handle_phase_control(connection: socket.socket, value: dict[str, object]) -> None:
    if set(value) != {"type", "sequence", "invocation_id", "kind"} or (
        value["type"],
        value["sequence"],
        value["invocation_id"],
        value["kind"],
    ) not in {
        ("phase_start", 0, "measured-generation", "measured"),
        ("phase_end", 1, "measured-generation", "measured"),
    }:
        raise ValueError("participant phase control differs")
    _send(connection, _local_mark(int(value["sequence"])))


def _run_child(command: list[str], runtime_dir: Path, node_rank: int) -> int:
    attempt = time.monotonic_ns()
    try:
        child_pid = os.fork()
    except OSError:
        _write_lifecycle(runtime_dir, node_rank, attempt, None, None, None)
        raise
    if child_pid == 0:
        os.setsid()
        os.execvpe(command[0], command, os.environ)
    started = time.monotonic_ns()
    _, raw_status = os.waitpid(child_pid, 0)
    finished = time.monotonic_ns()
    _write_lifecycle(runtime_dir, node_rank, attempt, started, finished, raw_status)
    return raw_status


def _write_lifecycle(
    runtime_dir: Path,
    node_rank: int,
    attempt: int,
    started: int | None,
    finished: int | None,
    raw_status: int | None,
) -> None:
    value = {
        "schema_version": "node-lifecycle/v1",
        "clock_id": _clock_id(),
        "spawn_attempt_monotonic_ns": attempt,
        "child_started_monotonic_ns": started,
        "child_finished_monotonic_ns": finished,
        "termination": (
            {"kind": "spawn_error"}
            if raw_status is None
            else {"kind": "waited", "raw_wait_status": raw_status}
        ),
    }
    destination = runtime_dir / f"source/lifecycle/node-{node_rank}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_json(destination, value)


def _record_node(arguments: argparse.Namespace, node_rank: int) -> None:
    path = arguments.runtime_dir / f"source/lifecycle/node-{node_rank}.json"
    facts = validate_node_lifecycle(path, node_rank=node_rank)
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        f"node-lifecycle:node-{node_rank}",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record_many(facts)


def _local_mark(
    sequence: int, *, invocation_id: str | None = None
) -> dict[str, object]:
    received = time.monotonic_ns()
    mark: dict[str, object] = {
        "sequence": sequence,
        "clock_id": _clock_id(),
        "received_monotonic_ns": received,
        "ack_monotonic_ns": time.monotonic_ns(),
    }
    if invocation_id is not None:
        mark["invocation_id"] = invocation_id
    return mark


def _mark_from_wire(
    value: dict[str, object], *, sequence: int, invocation_id: str | None = None
) -> dict[str, object]:
    expected = {"sequence", "clock_id", "received_monotonic_ns", "ack_monotonic_ns"}
    if invocation_id is not None:
        expected.add("invocation_id")
    if set(value) != expected or value.get("sequence") != sequence:
        raise ValueError("node-agent lifecycle mark differs")
    if invocation_id is not None and value.get("invocation_id") != invocation_id:
        raise ValueError("node-agent lifecycle invocation differs")
    return value


def _send(connection: socket.socket, value: object) -> None:
    connection.sendall(jcs_bytes(value) + b"\n")


def _receive(connection: socket.socket, timeout: int) -> dict[str, object]:
    connection.settimeout(timeout)
    payload = bytearray()
    while not payload.endswith(b"\n"):
        block = connection.recv(4096)
        if not block:
            raise OSError("node-agent connection closed")
        payload.extend(block)
        if len(payload) > 64 * 1024:
            raise ValueError("node-agent message is too large")
    value = load_json_strict(bytes(payload[:-1]))
    if not isinstance(value, dict):
        raise TypeError("node-agent message must be an object")
    return value


def _exclusive_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(jcs_bytes(value) + b"\n")
            stream.flush()
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _clock_id() -> str:
    boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    return f"{boot}:CLOCK_MONOTONIC"


def _shell_status(raw_status: int) -> int:
    if os.WIFEXITED(raw_status):
        return os.WEXITSTATUS(raw_status)
    if os.WIFSIGNALED(raw_status):
        return 128 + os.WTERMSIG(raw_status)
    raise ValueError("child wait status is not terminal")


def _uint(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{label} must be a nonnegative integer")
    return value


if __name__ == "__main__":
    sys.exit(main())
