#!/usr/bin/env python3
import os
import signal
import socket
import struct
import sys
from pathlib import Path

socket_path = Path(sys.argv[1])
ready_path = Path(sys.argv[2])
running = True


def stop(_signum: int, _frame: object) -> None:
    global running
    running = False


def read_exact(stream: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise EOFError("truncated event")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
socket_path.unlink(missing_ok=True)
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
    server.bind(os.fspath(socket_path))
    server.listen()
    server.settimeout(0.5)
    ready_path.touch(exist_ok=False)
    while running:
        try:
            stream, _ = server.accept()
        except TimeoutError:
            continue
        with stream:
            size = struct.unpack(">I", read_exact(stream, 4))[0]
            if size <= 0 or size > 16 * 1024 * 1024:
                raise ValueError("invalid event length")
            read_exact(stream, size)
socket_path.unlink(missing_ok=True)
