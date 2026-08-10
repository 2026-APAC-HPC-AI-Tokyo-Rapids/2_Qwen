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

    inputs = commands.add_parser("prepare-inputs")
    inputs.add_argument("--runtime-dir", type=Path, required=True)
    inputs.add_argument("--model-path", type=Path, required=True)
    inputs.add_argument("--model-repository", required=True)
    inputs.add_argument("--model-revision", required=True)
    inputs.add_argument("--dataset-path", type=Path, required=True)
    inputs.add_argument("--dataset-repository", required=True)
    inputs.add_argument("--dataset-revision", required=True)
    inputs.add_argument("--dataset-relative-path", required=True)

    provenance = commands.add_parser("record-provenance")
    _common_runtime_arguments(provenance)

    submission = commands.add_parser("prepare-submission")
    _common_runtime_arguments(submission)
    submission.add_argument("--pjm-script", type=Path, required=True)
    submission.add_argument("--repository-path", required=True)
    submission.add_argument("command", nargs=argparse.REMAINDER)

    submission_result = commands.add_parser("record-submission-result")
    _common_runtime_arguments(submission_result)
    submission_result.add_argument("--job-id", required=True)

    job_entry = commands.add_parser("record-job-entry")
    _common_runtime_arguments(job_entry)

    cache = commands.add_parser("record-cache")
    _common_runtime_arguments(cache)
    cache.add_argument("--node-rank", type=int, required=True)
    cache.add_argument("--cache-root", type=Path, required=True)

    launchers = commands.add_parser("capture-runtime-launchers")
    launchers.add_argument("--runtime-dir", type=Path, required=True)

    environment = commands.add_parser("record-environment")
    _common_runtime_arguments(environment)
    environment.add_argument(
        "--scope",
        choices=("job-entry", "host-node", "sglang-node", "worker-rank"),
        required=True,
    )
    environment.add_argument("--index", type=int)
    environment.add_argument("--source", type=Path)

    run_start = commands.add_parser("start-run-lifecycle")
    _common_runtime_arguments(run_start)

    run_finish = commands.add_parser("finish-run-lifecycle")
    _common_runtime_arguments(run_finish)
    run_finish.add_argument("--exit-code", type=int, required=True)

    accounting = commands.add_parser("record-accounting")
    _common_runtime_arguments(accounting)
    accounting.add_argument("--job-id", required=True)
    accounting.add_argument("--stdout", type=Path, required=True)
    accounting.add_argument("--stderr", type=Path, required=True)
    accounting.add_argument("--exit-code", type=int, required=True)

    hardware = commands.add_parser("capture-hardware")
    _common_runtime_arguments(hardware)
    hardware.add_argument("--node-rank", type=int, required=True)

    topology = commands.add_parser("capture-topology")
    _common_runtime_arguments(topology)
    topology.add_argument("--node-rank", type=int, required=True)
    topology.add_argument("--global-rank", type=int, required=True)
    topology.add_argument("--local-rank", type=int, required=True)
    topology.add_argument("--size", type=int, required=True)
    topology.add_argument("--node-wrapper-argv", type=Path, required=True)
    topology.add_argument("--singularity-argv", type=Path, required=True)
    topology.add_argument("--sglang-argv", type=Path, required=True)
    topology.add_argument("--singularity-launcher", type=Path, required=True)
    topology.add_argument("--sif", type=Path, required=True)

    workload = commands.add_parser("record-workload")
    _common_runtime_arguments(workload)
    workload.add_argument("--kind", choices=("warmup", "measured"), required=True)
    workload.add_argument("--source", type=Path, required=True)

    node_lifecycle = commands.add_parser("record-node-lifecycle")
    _common_runtime_arguments(node_lifecycle)
    node_lifecycle.add_argument("--node-rank", type=int, required=True)
    node_lifecycle.add_argument("--source", type=Path, required=True)

    lifecycle_control = commands.add_parser("record-lifecycle-control")
    _common_runtime_arguments(lifecycle_control)

    sif_sidecar = commands.add_parser("create-sif-sidecar")
    sif_sidecar.add_argument("--workspace", type=Path, required=True)
    sif_sidecar.add_argument("--sglang-source", type=Path, required=True)
    sif_sidecar.add_argument("--sif", type=Path, required=True)
    sif_sidecar.add_argument("--base-oci", type=Path, required=True)
    sif_sidecar.add_argument("--embedded-manifest", type=Path, required=True)
    sif_sidecar.add_argument("--runtime-inventory", type=Path, required=True)
    sif_sidecar.add_argument("--output", type=Path, required=True)

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
    if arguments.action == "prepare-inputs":
        return _prepare_inputs(arguments)
    if arguments.action == "record-provenance":
        return _record_provenance(arguments)
    if arguments.action == "prepare-submission":
        return _prepare_submission(arguments)
    if arguments.action == "record-submission-result":
        return _record_submission_result(arguments)
    if arguments.action == "record-job-entry":
        return _record_job_entry(arguments)
    if arguments.action == "record-cache":
        return _record_cache(arguments)
    if arguments.action == "capture-runtime-launchers":
        return _capture_runtime_launchers(arguments)
    if arguments.action == "record-environment":
        return _record_environment(arguments)
    if arguments.action == "start-run-lifecycle":
        return _start_run_lifecycle(arguments)
    if arguments.action == "finish-run-lifecycle":
        return _finish_run_lifecycle(arguments)
    if arguments.action == "record-accounting":
        return _record_accounting(arguments)
    if arguments.action == "capture-hardware":
        return _capture_hardware(arguments)
    if arguments.action == "capture-topology":
        return _capture_topology(arguments)
    if arguments.action == "record-workload":
        return _record_workload(arguments)
    if arguments.action == "record-node-lifecycle":
        return _record_node_lifecycle(arguments)
    if arguments.action == "record-lifecycle-control":
        return _record_lifecycle_control(arguments)
    if arguments.action == "create-sif-sidecar":
        return _create_sif_sidecar(arguments)
    if arguments.action == "merge":
        return _merge(arguments)
    if arguments.action == "package":
        return _package(arguments)
    if arguments.action == "publish":
        return _publish(arguments)
    raise AssertionError(arguments.action)


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
        raise ValueError("execution settings purpose differs from requested run purpose")
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


def _prepare_inputs(arguments: argparse.Namespace) -> int:
    from .sources import create_input_manifests

    model, dataset = create_input_manifests(
        arguments.runtime_dir,
        model_path=arguments.model_path,
        model_repository=arguments.model_repository,
        model_revision=arguments.model_revision,
        dataset_path=arguments.dataset_path,
        dataset_repository=arguments.dataset_repository,
        dataset_revision=arguments.dataset_revision,
        dataset_relative_path=arguments.dataset_relative_path,
    )
    print(jcs_text({"dataset_manifest": str(dataset), "model_manifest": str(model)}))
    return 0


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


def _prepare_submission(arguments: argparse.Namespace) -> int:
    from .sources import create_submission_request

    _, facts = create_submission_request(
        arguments.manifest,
        arguments.runtime_dir,
        pjm_script=arguments.pjm_script,
        repository_path=arguments.repository_path,
        argv=_without_separator(arguments.command),
    )
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        "submission-request",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record_many(facts)
    return 0


def _record_submission_result(arguments: argparse.Namespace) -> int:
    from .sources import create_submission_result

    _, fact = create_submission_result(
        arguments.manifest, arguments.runtime_dir, arguments.job_id
    )
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        "submission-result",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record(fact)
    return 0


def _record_job_entry(arguments: argparse.Namespace) -> int:
    from .sources import job_entry_fact

    manifest = load_manifest(arguments.manifest)
    fact = job_entry_fact(
        arguments.runtime_dir / "control/job-entry.json", manifest.run_id
    )
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        "job-entry",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record(fact)
    return 0


def _record_cache(arguments: argparse.Namespace) -> int:
    from .sources import create_cache_manifest

    producer = f"cache:node-{arguments.node_rank}"
    _, fact = create_cache_manifest(
        arguments.manifest,
        arguments.runtime_dir,
        node_rank=arguments.node_rank,
        cache_root=arguments.cache_root,
    )
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        producer,
        arguments.runtime_dir,
    ) as recorder:
        recorder.record(fact)
    return 0


def _capture_runtime_launchers(arguments: argparse.Namespace) -> int:
    from .sources import capture_runtime_launchers

    print(str(capture_runtime_launchers(arguments.runtime_dir)))
    return 0


def _record_environment(arguments: argparse.Namespace) -> int:
    from .system_sources import (
        capture_environment_snapshot,
        environment_facts,
        validate_environment_snapshot,
    )

    scope = arguments.scope
    if scope == "job-entry":
        if arguments.index is not None:
            raise ValueError("job-entry environment has no index")
        producer = "environment:job-entry"
        relative = "source/environment/job-entry.snapshot"
        prefix = "environment.job_entry"
    else:
        if arguments.index is None or arguments.index < 0:
            raise ValueError("indexed environment scope requires nonnegative --index")
        stem = {"host-node": "host", "sglang-node": "sglang", "worker-rank": "worker"}[scope]
        producer = f"environment:{scope}-{arguments.index}"
        relative = f"source/environment/{stem}-{arguments.index}.snapshot"
        prefix = {
            "host-node": "environment.host_exec_node",
            "sglang-node": "environment.sglang_launch_node",
            "worker-rank": "environment.worker_global_rank",
        }[scope] + f".{arguments.index}"
    destination = arguments.runtime_dir / relative
    if arguments.source is None:
        capture_environment_snapshot(destination)
    else:
        if arguments.source.resolve() != destination.resolve():
            raise ValueError("environment source must use its canonical runtime path")
        validate_environment_snapshot(destination)
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


def _record_accounting(arguments: argparse.Namespace) -> int:
    from .system_sources import create_scheduler_accounting

    _, facts = create_scheduler_accounting(
        arguments.manifest,
        arguments.runtime_dir,
        job_id=arguments.job_id,
        stdout_path=arguments.stdout,
        stderr_path=arguments.stderr,
        exit_code=arguments.exit_code,
    )
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        "scheduler-accounting",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record_many(facts)
    return 0


def _capture_hardware(arguments: argparse.Namespace) -> int:
    from .system_sources import create_minimal_hardware, validate_minimal_hardware

    path = create_minimal_hardware(
        arguments.runtime_dir, node_rank=arguments.node_rank
    )
    facts = validate_minimal_hardware(
        path, arguments.runtime_dir, arguments.node_rank
    )
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        f"hardware:node-{arguments.node_rank}",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record_many(facts)
    return 0


def _capture_topology(arguments: argparse.Namespace) -> int:
    from .canonical import load_json_strict
    from .system_sources import create_node_topology, validate_node_topology

    def argv(path: Path) -> list[str]:
        value = load_json_strict(path.read_bytes())
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
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
        singularity_argv=argv(arguments.singularity_argv),
        sglang_argv=argv(arguments.sglang_argv),
        singularity_launcher=arguments.singularity_launcher,
        sif=arguments.sif,
    )
    facts = validate_node_topology(
        path, arguments.runtime_dir, load_manifest(arguments.manifest).run_id,
        arguments.node_rank
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
    import hashlib

    from .workload import validate_ordered_workload

    dataset = arguments.runtime_dir / "source/dataset-manifest.json"
    identity = hashlib.sha256(dataset.read_bytes()).hexdigest()
    facts = validate_ordered_workload(
        arguments.source, kind=arguments.kind, dataset_manifest_sha256=identity
    )
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        f"workload:{arguments.kind}",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record_many(facts)
    return 0


def _record_node_lifecycle(arguments: argparse.Namespace) -> int:
    from .system_sources import validate_node_lifecycle

    facts = validate_node_lifecycle(arguments.source, node_rank=arguments.node_rank)
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        f"node-lifecycle:node-{arguments.node_rank}",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record_many(facts)
    return 0


def _record_lifecycle_control(arguments: argparse.Namespace) -> int:
    from .sources import load_execution_settings
    from .system_sources import (
        validate_lifecycle_rendezvous,
        validate_lifecycle_summary,
    )

    manifest = load_manifest(arguments.manifest)
    settings = load_execution_settings(
        arguments.runtime_dir / "control/execution-settings.json"
    )
    facts = (
        *validate_lifecycle_rendezvous(
            arguments.runtime_dir / "control/lifecycle/rendezvous.json",
            manifest.run_id,
            settings.node_agent_port,
        ),
        *validate_lifecycle_summary(
            arguments.runtime_dir / "control/lifecycle/summary.json",
            manifest.run_id,
            settings.nodes,
        ),
    )
    with Recorder.open(
        arguments.manifest,
        arguments.registry,
        "lifecycle-control",
        arguments.runtime_dir,
    ) as recorder:
        recorder.record_many(facts)
    return 0


def _create_sif_sidecar(arguments: argparse.Namespace) -> int:
    from .sources import create_sif_sidecar

    print(
        str(
            create_sif_sidecar(
                workspace=arguments.workspace,
                sglang_source=arguments.sglang_source,
                sif=arguments.sif,
                base_oci=arguments.base_oci,
                embedded_manifest=arguments.embedded_manifest,
                runtime_inventory=arguments.runtime_inventory,
                output=arguments.output,
            )
        )
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
    if manifest.run_purpose == "profile":
        from .profile_contract import project_profiler_events
        from .sources import load_execution_settings

        settings = load_execution_settings(
            arguments.metrics.parent / "control/execution-settings.json"
        )
        projection = project_profiler_events(
            arguments.metrics.parent,
            run_id=manifest.run_id,
            invocation_id="measured-generation",
            expected_ranks=settings.tensor_parallel_size,
        )
        if (
            projection.missing_ranks
            or projection.partial_ranks
            or projection.conflicts
        ):
            raise ValueError("profile raw artifacts are incomplete or conflicting")
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
