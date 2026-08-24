from __future__ import annotations

import signal
import threading
from collections.abc import Mapping
from pathlib import Path

from .gpu import GpuSampleWriter, NvidiaSmiSampler
from .manifest import load_manifest


def run_gpu_sampler(
    manifest_path: Path,
    output: Path,
    *,
    node_rank: int,
    clock_id: str,
    expected_gpus: Mapping[str, int],
    interval_sec: float,
    ready_file: Path | None = None,
    completion_file: Path,
) -> int:
    manifest = load_manifest(manifest_path)
    stop_requested = threading.Event()
    previous = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    for signum in previous:
        signal.signal(signum, stop)
    try:
        with GpuSampleWriter(
            output,
            run_id=manifest.run_id,
            purpose=manifest.run_purpose,
            node_rank=node_rank,
            clock_id=clock_id,
            expected_gpus=expected_gpus,
            flush_every_frames=10,
        ) as writer:
            sampler = NvidiaSmiSampler(writer, interval_sec=interval_sec)
            sampler.start()
            if ready_file is not None:
                ready_file.parent.mkdir(parents=True, exist_ok=True)
                ready_file.touch(exist_ok=False)
            while not stop_requested.wait(0.2):
                if not sampler.is_alive():
                    raise RuntimeError("GPU sampler thread terminated unexpectedly")
            sampler.stop()
            sampler.join(timeout=max(5.0, interval_sec * 2))
            if sampler.is_alive():
                raise RuntimeError("GPU sampler thread did not stop")
        completion_file.parent.mkdir(parents=True, exist_ok=True)
        completion_file.touch(exist_ok=False)
    finally:
        if ready_file is not None:
            ready_file.unlink(missing_ok=True)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0
