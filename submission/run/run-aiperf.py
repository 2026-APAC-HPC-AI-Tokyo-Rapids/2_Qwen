#!/usr/bin/env python3
"""Preserve AIPerf output and stop its own process group on startup failure."""

from __future__ import annotations

import os
import signal
import subprocess
import sys


def stop(child: subprocess.Popen[str]) -> None:
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # AIPerf children ignore SIGTERM. Also reap them if the controller exited.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def interrupted(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run-aiperf.py AIPERF [ARGS...]")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    child = subprocess.Popen(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
    try:
        assert child.stdout is not None
        for line in child.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if "ERROR" in line and "Failed to register service " in line:
                print("aiperf_startup_failed: service registration failed; stopping client", flush=True)
                return 1
        status = child.wait()
        return status if status >= 0 else 128 - status
    finally:
        stop(child)


if __name__ == "__main__":
    raise SystemExit(main())
