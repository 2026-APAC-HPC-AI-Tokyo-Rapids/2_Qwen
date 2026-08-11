from __future__ import annotations

import socket
import struct
import threading
from pathlib import Path

from .canonical import jcs_bytes, load_json_strict
from .events import ParsedEvent, facts_from_event
from .manifest import load_manifest
from .recording import Recorder

MAX_EVENT_BYTES = 64 * 1024 * 1024


class LocalEventRecordingError(ValueError):
    pass


class LocalEventFragmentRecorder:
    """Pure-event adapter routing node-local events to completed fact fragments."""

    def __init__(
        self,
        manifest_path: Path,
        runtime_dir: Path,
        registry_path: Path | None = None,
        *,
        node_rank: int | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.runtime_dir = runtime_dir
        self.registry_path = registry_path
        self.manifest = load_manifest(manifest_path)
        if node_rank is not None and node_rank < 0:
            raise ValueError("event archive node rank must be nonnegative")
        self.node_rank = node_rank
        self._recorders: dict[str, Recorder] = {}
        self._next_sequence: dict[str, int] = {}
        self._closed = False
        self._archive_path = (
            None
            if node_rank is None
            else runtime_dir / f"source/events/node-{node_rank}.jsonl"
        )
        self._archive_partial = (
            None
            if self._archive_path is None
            else self._archive_path.with_name(f".{self._archive_path.name}.partial")
        )
        self._archive_stream = None

    def record_bytes(self, payload: bytes) -> ParsedEvent:
        if self._closed:
            raise LocalEventRecordingError("event recorder is closed")
        if len(payload) > MAX_EVENT_BYTES:
            raise LocalEventRecordingError("domain event exceeds 64 MiB")
        value = load_json_strict(payload)
        event = facts_from_event(value, expected_run_id=self.manifest.run_id)
        if self.node_rank is not None:
            raw_node_rank = value.get("node_rank") if isinstance(value, dict) else None
            if raw_node_rank != self.node_rank:
                raise LocalEventRecordingError("domain event belongs to another node")
        expected_sequence = self._next_sequence.get(event.envelope_producer_id, 0)
        if event.sequence != expected_sequence:
            raise LocalEventRecordingError(
                f"producer {event.envelope_producer_id!r} sequence "
                f"{event.sequence} != {expected_sequence}"
            )
        recorder = self._recorders.get(event.producer_instance)
        if recorder is None:
            recorder = Recorder.open(
                self.manifest_path,
                self.registry_path,
                event.producer_instance,
                self.runtime_dir,
            )
            self._recorders[event.producer_instance] = recorder
        if self._archive_partial is not None:
            if self._archive_stream is None:
                self._archive_partial.parent.mkdir(parents=True, exist_ok=True)
                self._archive_stream = self._archive_partial.open("xb")
            self._archive_stream.write(jcs_bytes(value) + b"\n")
            self._archive_stream.flush()
        recorder.record_many(event.facts)
        self._next_sequence[event.envelope_producer_id] = expected_sequence + 1
        return event

    def close(self) -> tuple[Path, ...]:
        if self._closed:
            return tuple(
                recorder.completed_path for recorder in self._recorders.values()
            )
        paths = tuple(recorder.close() for recorder in self._recorders.values())
        if self._archive_stream is not None and self._archive_partial is not None:
            self._archive_stream.close()
            assert self._archive_path is not None
            self._archive_partial.replace(self._archive_path)
        self._closed = True
        return paths

    def abort(self) -> tuple[Path, ...]:
        paths = tuple(recorder.abort() for recorder in self._recorders.values())
        if self._archive_stream is not None:
            self._archive_stream.close()
        if self._archive_partial is not None:
            self._archive_partial.unlink(missing_ok=True)
        self._closed = True
        return paths


class UnixEventServer:
    """Length-prefixed AF_UNIX server; it never opens an IP socket."""

    def __init__(self, socket_path: Path, recorder: LocalEventFragmentRecorder) -> None:
        self.socket_path = socket_path
        self.recorder = recorder
        self.stop_requested = threading.Event()
        self.ready = threading.Event()
        self.errors: list[Exception] = []

    def serve_forever(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(self.socket_path))
                server.listen(8)
                server.settimeout(0.2)
                self.ready.set()
                # A producer's sendall can complete while its connection is still
                # queued in the listen backlog.  After shutdown is requested,
                # keep accepting until one empty interval proves the queue was
                # drained so completed workload events are not discarded.
                while True:
                    try:
                        connection, _ = server.accept()
                    except TimeoutError:
                        if self.stop_requested.is_set():
                            break
                        continue
                    with connection:
                        try:
                            size = struct.unpack(">I", _read_exact(connection, 4))[0]
                            if size > MAX_EVENT_BYTES:
                                raise LocalEventRecordingError(
                                    "domain event exceeds 64 MiB"
                                )
                            self.recorder.record_bytes(_read_exact(connection, size))
                        except (OSError, ValueError) as error:
                            self.errors.append(error)
        finally:
            self.socket_path.unlink(missing_ok=True)

    def stop(self) -> None:
        """Request termination; close the recorder after serve_forever returns."""

        self.stop_requested.set()


def _read_exact(stream: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        block = stream.recv(size - len(result))
        if not block:
            raise OSError("unexpected end of domain event stream")
        result.extend(block)
    return bytes(result)
