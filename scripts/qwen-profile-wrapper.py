#!/usr/bin/env python3
"""Launch the fixed legacy SGLang profiler overlay without a shell."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROFILE_ID = "qwen-profile"
PROFILE_FLAGS = (
    "--profile",
    "--profile-activities",
    "CPU",
    "GPU",
    "--profile-steps",
    "5",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qwen-profile-wrapper")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    command = (
        arguments.command[1:]
        if arguments.command[:1] == ["--"]
        else arguments.command
    )
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise ValueError("profile wrapper requires a nonempty command")
    if not arguments.output_dir.is_absolute():
        raise ValueError("profiler output directory must be absolute")
    if any(flag in command for flag in PROFILE_FLAGS):
        raise ValueError("profile command already contains an overlay flag")
    environment = os.environ.copy()
    environment.update(
        {
            "SGLANG_TORCH_PROFILER_DIR": str(arguments.output_dir),
            "SGLANG_PROFILE_ID": PROFILE_ID,
            "SGLANG_PROFILE_WITH_STACK": "0",
            "SGLANG_PROFILE_RECORD_SHAPES": "0",
        }
    )
    os.execvpe(command[0], [*command, *PROFILE_FLAGS], environment)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
