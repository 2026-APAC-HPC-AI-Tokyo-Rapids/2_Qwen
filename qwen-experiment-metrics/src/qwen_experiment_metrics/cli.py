from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import collectors
from .core import capture_credentials, make_run, send, warning
from .stage import run_stage


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qwen-experiment-metrics")
    commands = root.add_subparsers(dest="action", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument(
        "collector",
        choices=(
            "run-start",
            "run-end",
            "platform",
            "static-gpu",
            "source",
            "artifact",
            "log-artifact",
            "scheduler",
        ),
    )
    collect.add_argument("--runtime-dir", default=".")
    collect.add_argument("--job-id")
    collect.add_argument("--exit-code", type=int, default=0)
    collect.add_argument("--elapsed-sec")
    collect.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    stage = commands.add_parser("stage")
    stage.add_argument("name")
    stage.add_argument("--socket-dir", required=True)
    stage.add_argument("command", nargs=argparse.REMAINDER)
    return root


def main(argv: list[str] | None = None) -> int:
    capture_credentials()
    arguments = parser().parse_args(argv)
    run = make_run(arguments.job_id if hasattr(arguments, "job_id") else None)
    if arguments.action == "stage":
        command = _without_separator(arguments.command)
        if not command:
            warning("stage command is empty")
            return 127
        return run_stage(run, arguments.name, Path(arguments.socket_dir), command)
    runtime_dir = Path(arguments.runtime_dir).resolve()
    try:
        match arguments.collector:
            case "run-start":
                batch = collectors.run_start(_without_separator(arguments.command))
            case "run-end":
                batch = collectors.run_end(arguments.exit_code, arguments.elapsed_sec)
            case "platform":
                batch = collectors.platform_metrics()
            case "static-gpu":
                batch = collectors.static_gpu()
            case "source":
                batch = collectors.source_artifacts(runtime_dir)
            case "artifact":
                batch = collectors.result_artifacts(runtime_dir)
            case "log-artifact":
                batch = collectors.log_artifact(runtime_dir, arguments.job_id or "")
            case "scheduler":
                batch = collectors.scheduler(arguments.job_id or "")
            case _:
                raise AssertionError(arguments.collector)
        send(run, batch)
    except Exception as error:  # noqa: BLE001
        warning(f"{arguments.collector} collector failed: {error}")
    return 0


def _without_separator(command: list[str]) -> list[str]:
    return command[1:] if command and command[0] == "--" else command


if __name__ == "__main__":
    sys.exit(main())
