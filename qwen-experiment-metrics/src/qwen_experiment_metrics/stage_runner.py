from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from .domain import Fact
from .recording import Recorder

_STAGE_NAME = __import__("re").compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")


@dataclass(frozen=True, slots=True)
class Termination:
    kind: str
    code: int | None = None
    signal: int | None = None

    def to_object(self) -> dict[str, object]:
        if self.kind == "exit":
            return {"kind": "exit", "code": self.code}
        if self.kind == "signal":
            return {"kind": "signal", "signal": self.signal}
        if self.kind == "spawn_error":
            return {"kind": "spawn_error"}
        raise ValueError(f"unknown termination kind: {self.kind}")

    def shell_returncode(self) -> int:
        if self.kind == "exit":
            assert self.code is not None
            return self.code
        if self.kind == "signal":
            assert self.signal is not None
            return 128 + self.signal
        return 127


class StageRunner:
    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder

    def run(
        self,
        stage_name: str,
        argv: list[str],
        *,
        cwd: Path,
        stderr_mode: str = "inherit",
        env: dict[str, str] | None = None,
        stdout_path: Path | None = None,
    ) -> Termination:
        if _STAGE_NAME.fullmatch(stage_name) is None:
            raise ValueError("invalid stage name")
        if self.recorder.producer_id != f"stage:{stage_name}":
            raise ValueError("stage recorder producer_id does not match stage name")
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            raise ValueError("argv must be a non-empty array of non-empty strings")
        resolved_cwd = cwd.resolve(strict=True)
        if not resolved_cwd.is_dir():
            raise ValueError("cwd must be a directory")
        if stderr_mode not in {"inherit", "stdout"}:
            raise ValueError("stderr_mode must be inherit or stdout")

        prefix = f"run.stage.{stage_name}"
        self.recorder.record(
            Fact(
                f"{prefix}.launch",
                {
                    "argv": list(argv),
                    "cwd": os.fspath(resolved_cwd),
                    "stderr_mode": stderr_mode,
                },
            )
        )
        started_ns = time.monotonic_ns()
        stdout_stream = None
        try:
            if stdout_path is not None:
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                stdout_stream = stdout_path.open("xb")
            child = subprocess.Popen(
                argv,
                cwd=resolved_cwd,
                env=env,
                stdout=stdout_stream,
                stderr=None if stderr_mode == "inherit" else subprocess.STDOUT,
                shell=False,
            )
        except OSError:
            termination = Termination("spawn_error")
        else:
            previous_handlers = _install_forwarders(child)
            try:
                returncode = child.wait()
            finally:
                _restore_handlers(previous_handlers)
            termination = (
                Termination("signal", signal=-returncode)
                if returncode < 0
                else Termination("exit", code=returncode)
            )
        finally:
            if stdout_stream is not None:
                stdout_stream.flush()
                stdout_stream.close()
        elapsed_sec = (time.monotonic_ns() - started_ns) / 1_000_000_000
        self.recorder.record(Fact(f"{prefix}.termination", termination.to_object()))
        self.recorder.record(
            Fact(f"perf.stage.{stage_name}.wall_time_sec", elapsed_sec)
        )
        return termination


def _install_forwarders(
    child: subprocess.Popen[bytes],
) -> dict[signal.Signals, signal.Handlers]:
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous: dict[signal.Signals, signal.Handlers] = {}

    def forward(signum: int, _frame: FrameType | None) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, forward)
    return previous


def _restore_handlers(previous: dict[signal.Signals, signal.Handlers]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)
