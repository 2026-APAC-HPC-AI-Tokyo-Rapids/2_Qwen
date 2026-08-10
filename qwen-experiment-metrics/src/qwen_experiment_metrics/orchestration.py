from __future__ import annotations

import signal
import threading
from collections.abc import Mapping
from pathlib import Path

from .canonical import load_json_strict
from .domain import Fact
from .gpu import GpuSampleWriter, NvidiaSmiSampler
from .manifest import load_manifest
from .network import (
    parse_imb_mpi1,
    parse_nccl_tests_json,
    project_mpi_network_results_v2,
)
from .recording import Recorder

_COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter")


def record_network_sources(
    manifest_path: Path,
    runtime_dir: Path,
    *,
    imb_stdout: Path,
    imb_version: str,
    nccl_results: Mapping[str, Path],
    nccl_tests_commit: str,
    expected_mpi_processes: int,
    expected_gpus_per_process: int,
    registry_path: Path | None = None,
) -> None:
    manifest = load_manifest(manifest_path)
    if set(nccl_results) != set(_COLLECTIVES):
        raise ValueError("all three NCCL collective results are required")
    _validate_network_capture(
        runtime_dir, imb_stdout, nccl_results, nccl_tests_commit
    )
    imb_source = _relative_member(runtime_dir, imb_stdout)
    imb = parse_imb_mpi1(imb_stdout.read_text(encoding="utf-8"))
    process_counts = {row.process_count for row in imb}
    if not imb or process_counts != {expected_mpi_processes}:
        raise ValueError("IMB source is incomplete or has the wrong process grid")
    mpi_rows = project_mpi_network_results_v2(
        manifest.run_id, imb, imb_source
    )
    if not mpi_rows:
        raise ValueError("IMB projection is empty")

    nccl_rows = []
    nccl_versions: set[int] = set()
    for collective in _COLLECTIVES:
        path = nccl_results[collective]
        data = path.read_bytes()
        source_member = _relative_member(runtime_dir, path)
        rows = parse_nccl_tests_json(
            data,
            run_id=manifest.run_id,
            collective=collective,
            source_member=source_member,
        )
        if not rows:
            raise ValueError(f"NCCL {collective} projection is empty")
        if {
            (row.mpi_process_count, row.gpu_count) for row in rows
        } != {
            (
                expected_mpi_processes,
                expected_mpi_processes * expected_gpus_per_process,
            )
        }:
            raise ValueError(f"NCCL {collective} has the wrong process/GPU grid")
        document = load_json_strict(data)
        if not isinstance(document, dict) or isinstance(
            document.get("nccl_version"), bool
        ):
            raise TypeError("NCCL runtime version is missing")
        version = document.get("nccl_version")
        if not isinstance(version, int) or version <= 0:
            raise ValueError("NCCL runtime version is invalid")
        nccl_versions.add(version)
        nccl_rows.extend(rows)
    if len(nccl_versions) != 1:
        raise ValueError("NCCL runtime versions differ across collectives")

    sizes = [row.message_bytes for row in imb]
    max_iterations = max(row.iterations for row in imb)
    total_gpus = expected_mpi_processes * expected_gpus_per_process
    facts = (
        Fact("network.imb_mpi1.process_count", expected_mpi_processes),
        Fact("network.imb_mpi1.message_size.min_bytes", min(sizes)),
        Fact("network.imb_mpi1.message_size.max_bytes", max(sizes)),
        Fact("network.imb_mpi1.iteration.max_per_sample", max_iterations),
        Fact("network.imb_mpi1.iteration.time_limit_sec", 1),
        Fact("network.imb_mpi1.iteration.policy", "dynamic"),
        Fact("software.imb_mpi1.version", imb_version),
        Fact("network.nccl_tests.mpi_process_count", expected_mpi_processes),
        Fact("network.nccl_tests.gpus_per_process", expected_gpus_per_process),
        Fact("network.nccl_tests.total_gpu_count", total_gpus),
        Fact("network.nccl_tests.message_size.min_bytes", 8),
        Fact("network.nccl_tests.message_size.max_bytes", 1 << 30),
        Fact("network.nccl_tests.message_size.factor", 2),
        Fact("network.nccl_tests.warmup_iterations", 5),
        Fact("network.nccl_tests.timed_iterations", 20),
        Fact("network.nccl_tests.check_iterations", 1),
        Fact("network.nccl_tests.rank_average_mode", 1),
        Fact("software.nccl_tests.commit", nccl_tests_commit),
        Fact("software.nccl.version", str(next(iter(nccl_versions)))),
    )
    with Recorder.open(
        manifest_path, registry_path, "network", runtime_dir
    ) as recorder:
        recorder.record_many(facts)


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


def _relative_member(runtime_dir: Path, path: Path) -> str:
    root = runtime_dir.resolve()
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("network source is outside runtime_dir")
    return resolved.relative_to(root).as_posix()


def _validate_network_capture(
    runtime_dir: Path,
    imb_stdout: Path,
    nccl_results: Mapping[str, Path],
    nccl_tests_commit: str,
) -> None:
    imb_dir = imb_stdout.parent
    if {path.name for path in imb_dir.iterdir() if path.is_file()} != {
        "argv.json",
        "stdout.txt",
        "stderr.txt",
    }:
        raise ValueError("IMB capture membership is not closed")
    imb_argv = load_json_strict((imb_dir / "argv.json").read_bytes())
    if not isinstance(imb_argv, list) or imb_argv[-3:] != [
        "PingPong",
        "Sendrecv",
        "Allreduce",
    ]:
        raise ValueError("IMB argv does not contain the fixed benchmark set")
    network_root = runtime_dir / "source/network/nccl-tests"
    sidecar_path = network_root / "sidecar.json"
    sidecar = load_json_strict(sidecar_path.read_bytes())
    if sidecar != {
        "schema_version": "nccl-tests-sidecar/v1",
        "commit": nccl_tests_commit,
    }:
        raise ValueError("NCCL sidecar commit binding failed")
    for collective, result_path in nccl_results.items():
        directory = result_path.parent
        if directory.parent != network_root or directory.name != collective:
            raise ValueError("NCCL result path does not match its collective")
        if {path.name for path in directory.iterdir() if path.is_file()} != {
            "argv.json",
            "result.json",
            "stdout.txt",
            "stderr.txt",
        }:
            raise ValueError(f"NCCL {collective} membership is not closed")
        argv = load_json_strict((directory / "argv.json").read_bytes())
        if (
            not isinstance(argv, list)
            or f"/opt/nccl-tests/bin/{collective}_perf" not in argv
            or argv[-2:] != ["-J", "/network/result.json"]
        ):
            raise ValueError(f"NCCL {collective} argv is inconsistent")
