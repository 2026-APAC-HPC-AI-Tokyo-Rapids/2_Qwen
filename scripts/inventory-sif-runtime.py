#!/usr/bin/env python3
"""Create the closed runtime-file inventory from inside the completed SIF."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def record(role: str, requested: str) -> dict[str, object]:
    resolved = Path(os.path.realpath(requested))
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "role": role,
        "realpath": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def first_existing(candidates: list[str]) -> str:
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(candidates)


runtime_files = [
    record("python_executable", "/opt/sglang/venv/bin/python"),
    record("sglang_package_entry", "/opt/sglang/source/python/sglang/__init__.py"),
    record(
        "cuda_runtime",
        first_existing(
            [
                "/usr/local/cuda/lib64/libcudart.so",
                "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so",
            ]
        ),
    ),
    record(
        "nccl_runtime",
        first_existing(
            [
                "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
                "/usr/local/cuda/lib64/libnccl.so.2",
            ]
        ),
    ),
]
nccl_binaries = [
    record("all_reduce_perf", "/opt/nccl-tests/bin/all_reduce_perf"),
    record("all_gather_perf", "/opt/nccl-tests/bin/all_gather_perf"),
    record("reduce_scatter_perf", "/opt/nccl-tests/bin/reduce_scatter_perf"),
]
Path("/metadata/runtime-inventory.json").write_text(
    json.dumps(
        {"runtime_files": runtime_files, "nccl_binaries": nccl_binaries},
        separators=(",", ":"),
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
