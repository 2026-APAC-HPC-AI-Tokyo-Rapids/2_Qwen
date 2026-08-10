#!/usr/bin/env python3
"""Drain and count v2 events without retaining their payloads."""

import json
import os
import signal
import socket
import struct
import sys
from collections import Counter
from pathlib import Path

MAX_EVENT_BYTES = 64 * 1024 * 1024


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise EOFError("truncated event")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def remove_if_present(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_summary(path, value):
    temporary = path.with_name(".%s.tmp-%s" % (path.name, os.getpid()))
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        stream.write("\n")
        stream.flush()
    os.replace(str(temporary), str(path))


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 5:
        raise ValueError(
            "usage: event-drain SOCKET READY SUMMARY RUN_ID NODE_RANK"
        )
    socket_path = Path(arguments[0])
    ready_path = Path(arguments[1])
    summary_path = Path(arguments[2])
    run_id = arguments[3]
    node_rank = int(arguments[4])
    if node_rank < 0:
        raise ValueError("node rank must be nonnegative")

    state = {"running": True}

    def stop(_signum, _frame):
        state["running"] = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    remove_if_present(socket_path)
    remove_if_present(ready_path)
    event_types = Counter()
    producer_ids = set()
    sequences = {}
    payload_bytes = 0
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(os.fspath(socket_path))
            server.listen(16)
            server.settimeout(0.5)
            ready_path.touch(exist_ok=False)
            while state["running"]:
                try:
                    stream, _ = server.accept()
                except socket.timeout:
                    continue
                with stream:
                    size = struct.unpack(">I", read_exact(stream, 4))[0]
                    if size <= 0 or size > MAX_EVENT_BYTES:
                        raise ValueError("invalid event length")
                    payload = read_exact(stream, size)
                    value = json.loads(payload.decode("utf-8"))
                    if not isinstance(value, dict):
                        raise ValueError("event envelope is not an object")
                    expected = {
                        "schema_version",
                        "run_id",
                        "producer_id",
                        "node_rank",
                        "sequence",
                        "event_type",
                        "payload",
                    }
                    if "global_rank" in value:
                        expected.add("global_rank")
                    if (
                        set(value) != expected
                        or value.get("schema_version") != "qwen-domain-event/v1"
                        or value.get("run_id") != run_id
                        or value.get("node_rank") != node_rank
                        or not isinstance(value.get("producer_id"), str)
                        or not isinstance(value.get("event_type"), str)
                        or isinstance(value.get("sequence"), bool)
                        or not isinstance(value.get("sequence"), int)
                        or value.get("sequence") < 0
                        or not isinstance(value.get("payload"), dict)
                    ):
                        raise ValueError("event envelope differs")
                    producer_id = value["producer_id"]
                    expected_sequence = sequences.get(producer_id, 0)
                    if value["sequence"] != expected_sequence:
                        raise ValueError("event producer sequence differs")
                    sequences[producer_id] = expected_sequence + 1
                    event_types[value["event_type"]] += 1
                    producer_ids.add(producer_id)
                    payload_bytes += size
    finally:
        remove_if_present(socket_path)
        remove_if_present(ready_path)
    write_summary(
        summary_path,
        {
            "schema_version": "overhead-event-drain/v1",
            "run_id": run_id,
            "node_rank": node_rank,
            "total_events": sum(event_types.values()),
            "total_payload_bytes": payload_bytes,
            "event_type_counts": dict(sorted(event_types.items())),
            "producer_ids": sorted(producer_ids),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
