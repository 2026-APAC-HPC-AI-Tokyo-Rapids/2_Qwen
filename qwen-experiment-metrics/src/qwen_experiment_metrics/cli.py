from __future__ import annotations

import argparse
import signal
from pathlib import Path

from .canonical import jcs_bytes, jcs_text, load_json_strict
from .domain import Fact
from .local_socket import LocalEventFragmentRecorder, UnixEventServer
from .manifest import (
    SourceTrack,
    create_run_manifest,
    load_manifest,
    manifest_facts,
)
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
    initialize.add_argument("--execution-settings", type=Path, required=True)
    initialize.add_argument("--repository", type=Path, required=True)
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

    serve = commands.add_parser("serve-events")
    _common_runtime_arguments(serve)
    serve.add_argument("--socket", type=Path, required=True)
    serve.add_argument("--ready-file", type=Path)
    serve.add_argument("--node-rank", type=int, required=True)

    gpu = commands.add_parser("sample-gpu")
    _common_runtime_arguments(gpu)
    gpu.add_argument("--node-rank", type=int, required=True)
    gpu.add_argument("--clock-id", required=True)
    gpu.add_argument("--output", type=Path, required=True)
    gpu.add_argument("--expected-gpu", action="append", default=[], required=True)
    gpu.add_argument("--interval-sec", type=float, default=1.0)
    gpu.add_argument("--ready-file", type=Path)
    gpu.add_argument("--completion-file", type=Path, required=True)

    provenance = commands.add_parser("record-provenance")
    _common_runtime_arguments(provenance)

    launchers = commands.add_parser("capture-runtime-launchers")
    _common_runtime_arguments(launchers)

    environment = commands.add_parser("record-environment")
    _common_runtime_arguments(environment)
    environment.add_argument("--scope", choices=("runtime",), required=True)

    run_start = commands.add_parser("start-run-lifecycle")
    _common_runtime_arguments(run_start)

    run_finish = commands.add_parser("finish-run-lifecycle")
    _common_runtime_arguments(run_finish)
    run_finish.add_argument("--exit-code", type=int, required=True)

    hardware = commands.add_parser("capture-hardware")
    _common_runtime_arguments(hardware)
    hardware.add_argument("--node-rank", type=int, required=True)

    profile_hardware = commands.add_parser("capture-profile-hardware")
    _common_runtime_arguments(profile_hardware)
    profile_hardware.add_argument("--node-rank", type=int, required=True)

    profile_results = commands.add_parser("record-profile-results")
    _common_runtime_arguments(profile_results)
    profile_results.add_argument("--expected-ranks", type=int, required=True)

    topology = commands.add_parser("capture-topology")
    _common_runtime_arguments(topology)
    topology.add_argument("--node-rank", type=int, required=True)
    topology.add_argument("--global-rank", type=int, required=True)
    topology.add_argument("--local-rank", type=int, required=True)
    topology.add_argument("--size", type=int, required=True)
    topology.add_argument("--node-wrapper-argv", type=Path, required=True)
    topology.add_argument("--container-argv", type=Path, required=True)
    topology.add_argument("--sglang-argv", type=Path, required=True)
    topology.add_argument("--container-launcher", type=Path, required=True)
    topology.add_argument("--container-target", type=str, required=True)

    workload = commands.add_parser("record-workload")
    _common_runtime_arguments(workload)
    workload.add_argument("--kind", choices=("warmup", "measured"), required=True)
    workload.add_argument("--source", type=Path, required=True)

    package = commands.add_parser("package")
    _common_runtime_arguments(package)
    package.add_argument(
        "--member",
        action="append",
        default=[],
        metavar="ROLE=RELATIVE_PATH",
    )
    package.add_argument("--profile-member", action="append", default=[])

    send_fragments = commands.add_parser("send-fragments")
    send_fragments.add_argument("--run", required=True)
    send_fragments.add_argument("--runtime-dir", type=Path, required=True)

    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    handlers = {
        "init-run": _init_run,
        "run-stage": _run_stage,
        "serve-events": _serve_events,
        "sample-gpu": _sample_gpu,
        "record-provenance": _record_provenance,
        "capture-runtime-launchers": _capture_runtime_launchers,
        "record-environment": _record_environment,
        "start-run-lifecycle": _start_run_lifecycle,
        "finish-run-lifecycle": _finish_run_lifecycle,
        "capture-hardware": _capture_hardware,
        "capture-profile-hardware": _capture_profile_hardware,
        "record-profile-results": _record_profile_results,
        "capture-topology": _capture_topology,
        "record-workload": _record_workload,
        "package": _package,
        "send-fragments": _send_fragments,
    }
    return handlers[arguments.action](arguments)


def _send_fragments(arguments: argparse.Namespace) -> int:
    from .publish import send_fragments

    return 0 if send_fragments(arguments.run, arguments.runtime_dir) else 1


def _init_run(arguments: argparse.Namespace) -> int:
    from .sources import (
        execution_facts,
        install_execution_settings,
        validate_source_track_repository,
    )

    runtime_dir = arguments.runtime_dir.resolve()
    execution_path = install_execution_settings(
        arguments.execution_settings, runtime_dir
    )
    from .sources import load_execution_settings

    settings = load_execution_settings(execution_path)
    if settings.run_purpose != arguments.purpose:
        raise ValueError(
            "execution settings purpose differs from requested run purpose"
        )
    source_track = SourceTrack(
        object_hash_algorithm=arguments.object_hash_algorithm,
        base_role=arguments.base_role,
        base_commit=arguments.base_commit,
        head_commit=arguments.head_commit,
    )
    validate_source_track_repository(arguments.repository, source_track)
    manifest_path = create_run_manifest(
        runtime_dir,
        run_purpose=arguments.purpose,
        source_track=source_track,
        observed_control_plane_sha256=arguments.control_plane_sha256,
        instrumentation_bundle_sha256=arguments.instrumentation_bundle_sha256,
        run_id=arguments.run_id,
    )
    manifest = load_manifest(manifest_path)
    with Recorder.open(
        manifest_path, arguments.registry, "runner", runtime_dir
    ) as recorder:
        recorder.record_many(manifest_facts(manifest))
        recorder.record_many(execution_facts(settings))
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


def _serve_events(arguments: argparse.Namespace) -> int:
    recorder = LocalEventFragmentRecorder(
        arguments.manifest,
        arguments.runtime_dir,
        arguments.registry,
        node_rank=arguments.node_rank,
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


def _sample_gpu(arguments: argparse.Namespace) -> int:
    from .orchestration import run_gpu_sampler

    expected: dict[str, int] = {}
    for binding in arguments.expected_gpu:
        uuid, separator, raw_index = binding.partition("=")
        if not separator or not uuid or uuid in expected:
            raise ValueError("expected GPU bindings must use unique UUID=INDEX values")
        expected[uuid] = int(raw_index)
    status = run_gpu_sampler(
        arguments.manifest,
        arguments.output,
        node_rank=arguments.node_rank,
        clock_id=arguments.clock_id,
        expected_gpus=expected,
        interval_sec=arguments.interval_sec,
        ready_file=arguments.ready_file,
        completion_file=arguments.completion_file,
    )
    if status == 0:
        rows = [
            load_json_strict(line)
            for line in arguments.output.read_bytes().splitlines()
        ]
        with Recorder.open(
            arguments.manifest,
            arguments.registry,
            f"gpu-raw:node-{arguments.node_rank}",
            arguments.runtime_dir,
        ) as recorder:
            for index, row in enumerate(rows):
                recorder.record(
                    Fact(
                        f"profile.gpu.raw.node.{arguments.node_rank}.sample.{index}",
                        row,
                    )
                )
    return status


def _record_provenance(arguments: argparse.Namespace) -> int:
    from .sources import provenance_facts

    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        "provenance",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record_many(provenance_facts(arguments.runtime_dir))
    return 0


def _capture_runtime_launchers(arguments: argparse.Namespace) -> int:
    from .sources import capture_runtime_launchers, runtime_launcher_facts

    source = capture_runtime_launchers(arguments.runtime_dir)
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        "runtime-launchers",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record_many(runtime_launcher_facts(source, arguments.runtime_dir))
    print(str(source))
    return 0


def _record_environment(arguments: argparse.Namespace) -> int:
    from .system_sources import (
        capture_environment_snapshot,
        environment_facts,
    )

    scope = arguments.scope
    producer = f"environment:{scope}"
    relative = f"source/environment/{scope}.snapshot"
    prefix = f"environment.{scope.replace('-', '_')}"
    destination = arguments.runtime_dir / relative
    capture_environment_snapshot(destination)
    with Recorder.open(
        arguments.manifest, arguments.registry, producer, arguments.runtime_dir
    ) as recorder:
        recorder.record_many(environment_facts(destination, key_prefix=prefix))
    return 0


def _start_run_lifecycle(arguments: argparse.Namespace) -> int:
    from .system_sources import start_run_lifecycle

    start_run_lifecycle(arguments.manifest, arguments.runtime_dir)
    return 0


def _finish_run_lifecycle(arguments: argparse.Namespace) -> int:
    from .system_sources import finish_run_lifecycle

    _, facts = finish_run_lifecycle(
        arguments.manifest, arguments.runtime_dir, arguments.exit_code
    )
    with Recorder.open(
        arguments.manifest, arguments.registry, "run-lifecycle", arguments.runtime_dir
    ) as recorder:
        recorder.record_many(facts)
    return 0


def _capture_hardware(arguments: argparse.Namespace) -> int:
    from .system_sources import create_minimal_hardware, validate_minimal_hardware

    path = create_minimal_hardware(arguments.runtime_dir, node_rank=arguments.node_rank)
    facts = validate_minimal_hardware(path, arguments.runtime_dir, arguments.node_rank)
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        f"hardware:node-{arguments.node_rank}",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record_many(facts)
    return 0


def _capture_profile_hardware(arguments: argparse.Namespace) -> int:
    from .profile_sources import capture_profile_hardware

    capture_profile_hardware(
        arguments.manifest,
        arguments.runtime_dir,
        node_rank=arguments.node_rank,
    )
    return 0


def _record_profile_results(arguments: argparse.Namespace) -> int:
    import hashlib

    from .profile_contract import (
        project_gpu_phase,
        project_phase_summary,
        project_profiler_events,
    )

    manifest = load_manifest(arguments.manifest)
    phase = project_phase_summary(
        arguments.runtime_dir,
        run_id=manifest.run_id,
        expected_nodes=len(
            list((arguments.runtime_dir / "profile/raw/gpu").glob("node-*.jsonl"))
        ),
    )[0]
    expected_gpus: dict[int, dict[str, int]] = {}
    for path in sorted(
        (arguments.runtime_dir / "profile/raw/gpu").glob("node-*.jsonl")
    ):
        node_rank = int(path.stem.removeprefix("node-"))
        bindings: dict[str, int] = {}
        for line in path.read_bytes().splitlines():
            row = load_json_strict(line)
            bindings[str(row["gpu_uuid"])] = int(row["logical_index"])
        expected_gpus[node_rank] = bindings
    gpu_rows = project_gpu_phase(
        arguments.runtime_dir,
        run_id=manifest.run_id,
        phase=phase,
        expected_gpus=expected_gpus,
    )
    profiler = project_profiler_events(
        arguments.runtime_dir,
        run_id=manifest.run_id,
        invocation_id="measured-generation",
        expected_ranks=arguments.expected_ranks,
    ).require_conflict_free()
    with Recorder.open(
        arguments.manifest, arguments.registry, "profile-reducer", arguments.runtime_dir
    ) as recorder:
        recorder.record(Fact("profile.phase.summary", phase.as_dict()))
        for row in gpu_rows:
            value = row.as_dict()
            recorder.record(
                Fact(
                    f"profile.gpu.aggregate.node.{value['node_rank']}.index.{value['logical_index']}",
                    value,
                )
            )
        for capture in profiler.captures:
            recorder.record(
                Fact(
                    f"profile.profiler.capture.rank.{capture.global_rank}",
                    capture.as_dict(),
                )
            )
        for row in profiler.profiler_rows:
            value = row.as_dict()
            identity = hashlib.sha256(
                jcs_bytes(
                    [row.global_rank, row.activity, row.event_category, row.event_name]
                )
            ).hexdigest()
            recorder.record(Fact(f"profile.profiler.event.{identity}", value))
        recorder.record(
            Fact(
                "profile.profiler.coverage",
                {
                    "expected_ranks": arguments.expected_ranks,
                    "missing_ranks": list(profiler.missing_ranks),
                    "partial_ranks": list(profiler.partial_ranks),
                },
            )
        )
    return 0


def _capture_topology(arguments: argparse.Namespace) -> int:
    from .canonical import load_json_strict
    from .system_sources import create_node_topology, validate_node_topology

    def argv(path: Path) -> list[str]:
        value = load_json_strict(path.read_bytes())
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise TypeError("argv source must be a JSON string array")
        return value

    path = create_node_topology(
        arguments.manifest,
        arguments.runtime_dir,
        node_rank=arguments.node_rank,
        global_rank=arguments.global_rank,
        local_rank=arguments.local_rank,
        size=arguments.size,
        node_wrapper_argv=argv(arguments.node_wrapper_argv),
        container_argv=argv(arguments.container_argv),
        sglang_argv=argv(arguments.sglang_argv),
        container_launcher=arguments.container_launcher,
        container_target=arguments.container_target,
    )
    facts = validate_node_topology(
        path,
        arguments.runtime_dir,
        load_manifest(arguments.manifest).run_id,
        arguments.node_rank,
    )
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        f"topology:node-{arguments.node_rank}",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record_many(facts)
    return 0


def _record_workload(arguments: argparse.Namespace) -> int:
    from .workload import validate_ordered_workload

    facts = validate_ordered_workload(arguments.source, kind=arguments.kind)
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        f"workload:{arguments.kind}",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record_many(facts)
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
    raise SystemExit(main())
