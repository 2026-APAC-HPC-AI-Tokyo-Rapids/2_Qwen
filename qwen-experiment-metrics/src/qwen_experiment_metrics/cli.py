from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from .canonical import jcs_text
from .local_socket import LocalEventFragmentRecorder, UnixEventServer
from .manifest import (
    SourceTrack,
    create_run_manifest,
    load_manifest,
    manifest_facts,
)
from .merge import merge_run
from .recording import Recorder
from .stage_runner import StageRunner


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qwen-experiment-metrics")
    commands = root.add_subparsers(dest="action", required=True)

    initialize = commands.add_parser("init-run")
    initialize.add_argument("--runtime-dir", type=Path, required=True)
    initialize.add_argument("--purpose", choices=("bench", "profile"), required=True)
    initialize.add_argument(
        "--object-hash-algorithm", choices=("sha1", "sha256"), required=True
    )
    initialize.add_argument(
        "--base-role", choices=("score_reference", "profile_base"), required=True
    )
    initialize.add_argument("--base-commit", required=True)
    initialize.add_argument("--head-commit", required=True)
    initialize.add_argument("--control-plane-sha256", required=True)
    initialize.add_argument("--instrumentation-bundle-sha256", required=True)
    initialize.add_argument("--run-id")
    initialize.add_argument("--registry", type=Path)

    stage = commands.add_parser("run-stage")
    _common_runtime_arguments(stage)
    stage.add_argument("--name", required=True)
    stage.add_argument("--cwd", type=Path, required=True)
    stage.add_argument(
        "--stderr-mode", choices=("inherit", "stdout"), default="inherit"
    )
    stage.add_argument("--stdout-path", type=Path)
    stage.add_argument("command", nargs=argparse.REMAINDER)

    record = commands.add_parser("record-event")
    _common_runtime_arguments(record)

    serve = commands.add_parser("serve-events")
    _common_runtime_arguments(serve)
    serve.add_argument("--socket", type=Path, required=True)
    serve.add_argument("--ready-file", type=Path)

    gpu = commands.add_parser("sample-gpu")
    _common_runtime_arguments(gpu)
    gpu.add_argument("--node-rank", type=int, required=True)
    gpu.add_argument("--clock-id", required=True)
    gpu.add_argument("--output", type=Path, required=True)
    gpu.add_argument("--expected-gpu", action="append", default=[], required=True)
    gpu.add_argument("--interval-sec", type=float, default=1.0)
    gpu.add_argument("--ready-file", type=Path)
    gpu.add_argument("--completion-file", type=Path, required=True)

    network = commands.add_parser("record-network")
    _common_runtime_arguments(network)
    network.add_argument("--imb-stdout", type=Path, required=True)
    network.add_argument("--imb-version", required=True)
    network.add_argument("--nccl-all-reduce", type=Path, required=True)
    network.add_argument("--nccl-all-gather", type=Path, required=True)
    network.add_argument("--nccl-reduce-scatter", type=Path, required=True)
    network.add_argument("--nccl-tests-commit", required=True)
    network.add_argument("--expected-mpi-processes", type=int, required=True)
    network.add_argument("--expected-gpus-per-process", type=int, required=True)

    merge = commands.add_parser("merge")
    _common_runtime_arguments(merge)
    merge.add_argument("--output", type=Path)

    package = commands.add_parser("package")
    _common_runtime_arguments(package)
    package.add_argument(
        "--member",
        action="append",
        default=[],
        metavar="ROLE=RELATIVE_PATH",
    )
    package.add_argument("--profile-member", action="append", default=[])

    publish = commands.add_parser("publish")
    publish.add_argument("--metrics", type=Path, required=True)
    publish.add_argument("--artifact-manifest", type=Path, required=True)
    publish.add_argument("--pyproject", type=Path, required=True)
    publish.add_argument("--manifest", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.action == "init-run":
        return _init_run(arguments)
    if arguments.action == "run-stage":
        return _run_stage(arguments)
    if arguments.action == "record-event":
        return _record_event(arguments)
    if arguments.action == "serve-events":
        return _serve_events(arguments)
    if arguments.action == "sample-gpu":
        return _sample_gpu(arguments)
    if arguments.action == "record-network":
        return _record_network(arguments)
    if arguments.action == "merge":
        return _merge(arguments)
    if arguments.action == "package":
        return _package(arguments)
    if arguments.action == "publish":
        return _publish(arguments)
    raise AssertionError(arguments.action)


def _init_run(arguments: argparse.Namespace) -> int:
    runtime_dir = arguments.runtime_dir.resolve()
    manifest_path = create_run_manifest(
        runtime_dir,
        run_purpose=arguments.purpose,
        source_track=SourceTrack(
            object_hash_algorithm=arguments.object_hash_algorithm,
            base_role=arguments.base_role,
            base_commit=arguments.base_commit,
            head_commit=arguments.head_commit,
        ),
        observed_control_plane_sha256=arguments.control_plane_sha256,
        instrumentation_bundle_sha256=arguments.instrumentation_bundle_sha256,
        run_id=arguments.run_id,
    )
    manifest = load_manifest(manifest_path)
    with Recorder.open(
        manifest_path, arguments.registry, "runner", runtime_dir
    ) as recorder:
        recorder.record_many(manifest_facts(manifest))
    print(jcs_text({"manifest_path": str(manifest_path), "run_id": manifest.run_id}))
    return 0


def _run_stage(arguments: argparse.Namespace) -> int:
    command = _without_separator(arguments.command)
    if not command:
        raise ValueError("stage command is empty")
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        f"stage:{arguments.name}",
        arguments.runtime_dir,
    ) as recorder:
        termination = StageRunner(recorder).run(
            arguments.name,
            command,
            cwd=arguments.cwd,
            stderr_mode=arguments.stderr_mode,
            stdout_path=arguments.stdout_path,
        )
    return termination.shell_returncode()


def _record_event(arguments: argparse.Namespace) -> int:
    recorder = LocalEventFragmentRecorder(
        arguments.manifest, arguments.runtime_dir, arguments.registry
    )
    try:
        recorder.record_bytes(sys.stdin.buffer.read())
        recorder.close()
    except BaseException:
        recorder.abort()
        raise
    return 0


def _serve_events(arguments: argparse.Namespace) -> int:
    recorder = LocalEventFragmentRecorder(
        arguments.manifest, arguments.runtime_dir, arguments.registry
    )
    server = UnixEventServer(arguments.socket, recorder)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def stop(_signum: int, _frame: object) -> None:
        server.stop_requested.set()

    for signum in previous_handlers:
        signal.signal(signum, stop)
    try:
        if arguments.ready_file is not None:
            # The server sets readiness only after bind; a helper thread mirrors it to a file.
            _start_ready_file_writer(server, arguments.ready_file)
        server.serve_forever()
        recorder.close()
    except BaseException:
        recorder.abort()
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if arguments.ready_file is not None:
            arguments.ready_file.unlink(missing_ok=True)
    return 1 if server.errors else 0


def _merge(arguments: argparse.Namespace) -> int:
    outcome = merge_run(
        arguments.manifest,
        arguments.runtime_dir,
        registry_path=arguments.registry,
        output_path=arguments.output,
    )
    print(
        jcs_text(
            {
                "collection_complete": outcome.collection_complete,
                "conflicting_keys": list(outcome.conflicting_keys),
                "invalid_fragments": list(outcome.invalid_fragments),
                "missing_producers": list(outcome.missing_producers),
                "missing_required_facts": list(outcome.missing_required_facts),
                "missing_required_files": list(outcome.missing_required_files),
                "output_path": str(outcome.output_path),
            }
        )
    )
    return 0 if outcome.collection_complete else 1


def _sample_gpu(arguments: argparse.Namespace) -> int:
    from .orchestration import run_gpu_sampler

    expected: dict[str, int] = {}
    for binding in arguments.expected_gpu:
        uuid, separator, raw_index = binding.partition("=")
        if not separator or not uuid or uuid in expected:
            raise ValueError("expected GPU bindings must use unique UUID=INDEX values")
        expected[uuid] = int(raw_index)
    return run_gpu_sampler(
        arguments.manifest,
        arguments.output,
        node_rank=arguments.node_rank,
        clock_id=arguments.clock_id,
        expected_gpus=expected,
        interval_sec=arguments.interval_sec,
        ready_file=arguments.ready_file,
        completion_file=arguments.completion_file,
    )


def _record_network(arguments: argparse.Namespace) -> int:
    from .orchestration import record_network_sources

    record_network_sources(
        arguments.manifest,
        arguments.runtime_dir,
        imb_stdout=arguments.imb_stdout,
        imb_version=arguments.imb_version,
        nccl_results={
            "all_reduce": arguments.nccl_all_reduce,
            "all_gather": arguments.nccl_all_gather,
            "reduce_scatter": arguments.nccl_reduce_scatter,
        },
        nccl_tests_commit=arguments.nccl_tests_commit,
        expected_mpi_processes=arguments.expected_mpi_processes,
        expected_gpus_per_process=arguments.expected_gpus_per_process,
        registry_path=arguments.registry,
    )
    return 0


def _package(arguments: argparse.Namespace) -> int:
    from .artifact_store import RunPackager

    members: list[tuple[str, str]] = []
    for binding in arguments.member:
        role, separator, relative = binding.partition("=")
        if not separator or not role or not relative:
            raise ValueError("package members must use ROLE=RELATIVE_PATH")
        members.append((relative, role))
    result = RunPackager().package(
        arguments.manifest,
        arguments.runtime_dir,
        run_members=tuple(members),
        profile_members=tuple(arguments.profile_member),
    )
    print(
        jcs_text(
            {
                "artifact_manifest": str(result.manifest_path),
                "files": [str(path) for path in result.files],
            }
        )
    )
    return 0


def _publish(arguments: argparse.Namespace) -> int:
    from .publish import BigQueryConfig, publish_metrics

    expected_manifest = arguments.metrics.resolve().parent / "control/run-manifest.json"
    if arguments.manifest.resolve() != expected_manifest:
        raise ValueError("publish manifest must be the metrics runtime manifest")
    manifest = load_manifest(arguments.manifest)
    count = publish_metrics(
        arguments.metrics,
        BigQueryConfig.load(arguments.pyproject),
        artifact_manifest=arguments.artifact_manifest,
        expected_run=manifest.run_id,
    )
    print(jcs_text({"published_rows": count, "run_id": manifest.run_id}))
    return 0


def _common_runtime_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--runtime-dir", type=Path, required=True)
    command.add_argument("--registry", type=Path)


def _without_separator(command: list[str]) -> list[str]:
    return command[1:] if command and command[0] == "--" else command


def _start_ready_file_writer(server: UnixEventServer, path: Path) -> None:
    import threading

    def write_when_ready() -> None:
        server.ready.wait()
        if not server.stop_requested.is_set():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=False)

    threading.Thread(target=write_when_ready, daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
