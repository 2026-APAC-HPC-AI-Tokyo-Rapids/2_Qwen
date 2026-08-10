from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .canonical import fact_row_bytes, load_json_strict, validate_encoded_value
from .domain import Fact
from .events import parse_primary_result
from .gpu import parse_gpu_samples_jsonl
from .manifest import load_manifest
from .profile_contract import project_profiler_events
from .recording import RecorderError, producer_from_fragment
from .registry import Registry, RegistryError, load_registry
from .sources import (
    ExecutionSettings,
    load_execution_settings,
    validate_cache_manifest,
    validate_dataset_manifest,
    validate_model_manifest,
    validate_runtime_launchers,
    validate_sif_sidecar,
    validate_submission_request,
)
from .system_sources import (
    topology_gpu_bindings,
    validate_environment_snapshot,
    validate_lifecycle_rendezvous,
    validate_lifecycle_summary,
    validate_minimal_hardware,
    validate_node_lifecycle,
    validate_node_topology,
    validate_run_lifecycle,
    validate_scheduler_accounting,
)
from .workload import validate_ordered_workload


@dataclass(frozen=True, slots=True)
class MergeOutcome:
    output_path: Path
    collection_complete: bool
    missing_producers: tuple[str, ...]
    missing_required_facts: tuple[str, ...]
    conflicting_keys: tuple[str, ...]
    invalid_fragments: tuple[str, ...]
    missing_required_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Observed:
    producer_id: str
    value: str
    fragment_path: Path


def merge_run(
    manifest_path: Path,
    runtime_dir: Path,
    *,
    registry_path: Path | None = None,
    output_path: Path | None = None,
) -> MergeOutcome:
    manifest = load_manifest(manifest_path)
    registry = load_registry(registry_path)
    settings = load_execution_settings(runtime_dir / "control/execution-settings.json")
    if settings.run_purpose != manifest.run_purpose:
        raise ValueError("execution settings purpose differs from run manifest")
    required_producers = registry.default_required_producers(manifest.run_purpose)
    network_required = _has_valid_primary_result(runtime_dir, manifest.run_id)
    required_producers.update(_execution_producers(settings, network_required))
    missing_required_files = _missing_execution_files(
        runtime_dir,
        manifest.run_purpose,
        manifest.run_id,
        settings,
        network_required,
    )

    fragment_dir = runtime_dir / "domain" / "fragments"
    observed_by_key: dict[str, list[_Observed]] = {}
    facts_by_producer: dict[str, list[str]] = {}
    values_by_producer: dict[str, dict[str, str]] = {}
    observed_producers: set[str] = set()
    invalid_fragments: list[str] = []
    completed_fragment_paths = sorted(fragment_dir.glob("*.jsonl"))

    for path in completed_fragment_paths:
        try:
            producer_id = producer_from_fragment(path)
            registry.producer(producer_id, manifest.run_purpose)
            rows = _read_fragment(
                path,
                manifest.run_id,
                producer_id,
                registry,
                manifest.run_purpose,
                manifest.instrumentation_bundle_sha256,
            )
        except (OSError, ValueError, RecorderError, RegistryError) as error:
            invalid_fragments.append(f"{path.name}: {error}")
            continue
        observed_producers.add(producer_id)
        producer_keys = facts_by_producer.setdefault(producer_id, [])
        producer_values = values_by_producer.setdefault(producer_id, {})
        for key, value in rows:
            producer_keys.append(key)
            producer_values[key] = value
            observed_by_key.setdefault(key, []).append(
                _Observed(producer_id, value, path)
            )

    invalid_fragments.extend(
        _configuration_axis_errors(
            facts_by_producer, observed_producers, settings.nodes
        )
    )
    invalid_fragments.extend(_runner_contract_errors(facts_by_producer, manifest.run_purpose))
    invalid_fragments.extend(
        _placement_axis_errors(values_by_producer, observed_producers, settings)
    )

    missing_producers = sorted(required_producers - observed_producers)
    missing_required_facts: list[str] = []
    for producer_id in sorted(observed_producers):
        producer = registry.producer(producer_id, manifest.run_purpose)
        producer_keys = facts_by_producer.get(producer_id, [])
        for required in registry.required_fact_specs(producer, manifest.run_purpose):
            if not any(required.matches(key) for key in producer_keys):
                missing_required_facts.append(f"{producer_id}:{required.id}")

    merged_rows: list[tuple[str, str]] = []
    conflicts: list[str] = []
    for key, observations in observed_by_key.items():
        distinct = {item.value for item in observations}
        if len(distinct) != 1:
            conflicts.append(key)
        merged_rows.extend((key, value) for value in distinct)

    collection_complete = not (
        missing_producers
        or missing_required_facts
        or conflicts
        or invalid_fragments
        or missing_required_files
    )
    collection_spec = registry.fact("collection_complete", manifest.run_purpose)
    assert collection_spec.owner == "merge"
    merged_rows.append(
        (
            "collection_complete",
            registry.encode(
                "merge",
                manifest.run_purpose,
                Fact("collection_complete", collection_complete),
            ),
        )
    )

    destination = output_path or runtime_dir / "metrics.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb", buffering=64 * 1024) as stream:
            for key, value in sorted(
                merged_rows,
                key=lambda item: (item[0].encode("utf-8"), item[1].encode("utf-8")),
            ):
                stream.write(fact_row_bytes(manifest.run_id, key, value) + b"\n")
            stream.flush()
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    conflicting_paths = {
        item.fragment_path
        for key in conflicts
        for item in observed_by_key.get(key, ())
    }
    invalid_names = {item.split(":", 1)[0] for item in invalid_fragments}
    for fragment_path in completed_fragment_paths:
        if fragment_path.name not in invalid_names and fragment_path not in conflicting_paths:
            fragment_path.unlink()
    return MergeOutcome(
        destination,
        collection_complete,
        tuple(missing_producers),
        tuple(missing_required_facts),
        tuple(sorted(conflicts)),
        tuple(invalid_fragments),
        tuple(missing_required_files),
    )


def _read_fragment(
    path: Path,
    run_id: str,
    producer_id: str,
    registry: Registry,
    purpose: str,
    instrumentation_bundle_sha256: str,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"line {line_number} is not newline terminated")
            document = load_json_strict(line)
            if not isinstance(document, dict) or set(document) != {
                "run",
                "key",
                "value",
            }:
                raise ValueError(f"line {line_number} is not a three-column fact")
            row_run, key, value = document["run"], document["key"], document["value"]
            if row_run != run_id:
                raise ValueError(f"line {line_number} has the wrong run ID")
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(f"line {line_number} key/value must be strings")
            spec = registry.fact(key, purpose)
            producer = registry.producer(producer_id, purpose)
            if spec.owner != producer.id:
                raise ValueError(f"line {line_number} violates fact ownership")
            _validate_instance_axis(producer_id, key, line_number)
            validate_encoded_value(spec.logical_type, value)
            manifest_values = {
                "run.identity.run_id": run_id,
                "run.identity.run_purpose": purpose,
                "run.identity.repository_id": "apac2026-qwen",
                "run.identity.instrumentation_bundle_sha256": (
                    instrumentation_bundle_sha256
                ),
            }
            if key in manifest_values and value != manifest_values[key]:
                raise ValueError(f"line {line_number} conflicts with run manifest")
            rows.append((key, value))
    return rows


def _execution_producers(
    settings: ExecutionSettings, network_required: bool
) -> set[str]:
    result = {
        "stage:benchmark",
        "result",
        *(f"configuration:node-{node}" for node in range(settings.nodes)),
        *(f"cache:node-{node}" for node in range(settings.nodes)),
        "submission-request",
        "submission-result",
        "job-entry",
        "environment:job-entry",
        "run-lifecycle",
        "scheduler-accounting",
        "workload:warmup",
        "workload:measured",
        "lifecycle-control",
        *(f"environment:host-node-{node}" for node in range(settings.nodes)),
        *(f"environment:sglang-node-{node}" for node in range(settings.nodes)),
        *(f"hardware:node-{node}" for node in range(settings.nodes)),
        *(f"topology:node-{node}" for node in range(settings.nodes)),
        *(f"node-lifecycle:node-{node}" for node in range(settings.nodes)),
        *(
            f"environment:worker-rank-{rank}"
            for rank in range(settings.tensor_parallel_size)
        ),
        *(f"worker:global-rank-{rank}" for rank in range(settings.tensor_parallel_size)),
    }
    if network_required:
        result.update(
            {
                "stage:network_imb_mpi1",
                "stage:network_nccl_all_reduce",
                "stage:network_nccl_all_gather",
                "stage:network_nccl_reduce_scatter",
                "network",
            }
        )
    return result


def _missing_execution_files(
    runtime_dir: Path,
    purpose: str,
    run_id: str,
    settings: ExecutionSettings,
    network_required: bool,
) -> list[str]:
    required = [
        "control/job-entry.json",
        "control/execution-settings.json",
        "control/submission-request.json",
        "control/submission-result.json",
        "source/sif-sidecar.json",
        "source/model-manifest.json",
        "source/dataset-manifest.json",
        "source/runtime-launchers.json",
        "source/scheduler/accounting.txt",
        "source/scheduler/accounting.stderr",
        "source/scheduler/accounting.json",
        "source/run-lifecycle.json",
        "control/lifecycle/run-start.json",
        "control/lifecycle/rendezvous.json",
        "control/lifecycle/summary.json",
        "source/environment/job-entry.snapshot",
        "source/workload-warmup.json",
        "source/workload-measured.json",
        "source/primary-result.json",
        *(f"source/topology/node-{node}.json" for node in range(settings.nodes)),
        *(f"source/hardware/node-{node}.json" for node in range(settings.nodes)),
        *(
            f"source/hardware/node-{node}.lscpu.stderr"
            for node in range(settings.nodes)
        ),
        *(f"source/lifecycle/node-{node}.json" for node in range(settings.nodes)),
        *(
            f"source/environment/host-{node}.snapshot"
            for node in range(settings.nodes)
        ),
        *(
            f"source/environment/sglang-{node}.snapshot"
            for node in range(settings.nodes)
        ),
        *(
            f"source/environment/worker-{rank}.snapshot"
            for rank in range(settings.tensor_parallel_size)
        ),
        *(
            f"source/cache/node-{node}-input-manifest.json"
            for node in range(settings.nodes)
        ),
        "logs/benchmark.log",
    ]
    primary = runtime_dir / "source/primary-result.json"
    if primary.is_file() and not network_required:
        required.append("source/primary-result.json:invalid")
    if network_required:
        required.extend(
            [
                "source/network/imb-mpi1/argv.json",
                "source/network/imb-mpi1/stdout.txt",
                "source/network/imb-mpi1/stderr.txt",
                "source/network/nccl-tests/sidecar.json",
                *[
                    f"source/network/nccl-tests/{collective}/{name}"
                    for collective in ("all_reduce", "all_gather", "reduce_scatter")
                    for name in ("argv.json", "result.json", "stdout.txt", "stderr.txt")
                ],
            ]
        )
    if purpose == "profile":
        for rank in range(settings.tensor_parallel_size):
            required.extend(
                [
                    f"profile/raw/torch/qwen-profile-TP-{rank}.trace.json.gz",
                    f"profile/raw/torch/qwen-profile-TP-{rank}.capture.json",
                ]
            )
    missing = [relative for relative in required if not (runtime_dir / relative).is_file()]
    job_entry = runtime_dir / "control/job-entry.json"
    if job_entry.is_file():
        try:
            _validate_job_entry(job_entry, run_id)
        except (OSError, ValueError, TypeError):
            missing.append("control/job-entry.json:invalid")
    submission_request = runtime_dir / "control/submission-request.json"
    if submission_request.is_file():
        try:
            validate_submission_request(
                submission_request,
                runtime_dir / "control/run-manifest.json",
                run_id,
            )
        except (OSError, ValueError, TypeError):
            missing.append("control/submission-request.json:invalid")
    submission_result = runtime_dir / "control/submission-result.json"
    if submission_result.is_file():
        try:
            _validate_submission_result(submission_result, run_id)
        except (OSError, ValueError, TypeError):
            missing.append("control/submission-result.json:invalid")
    validators = (
        (
            "source/sif-sidecar.json",
            lambda path: validate_sif_sidecar(
                path, member_root=runtime_dir / "source"
            ),
        ),
        ("source/model-manifest.json", validate_model_manifest),
        ("source/dataset-manifest.json", validate_dataset_manifest),
    )
    for relative, validator in validators:
        path = runtime_dir / relative
        if path.is_file():
            try:
                validator(path)
            except (OSError, ValueError, TypeError):
                missing.append(f"{relative}:invalid")
    launcher_path = runtime_dir / "source/runtime-launchers.json"
    if launcher_path.is_file():
        try:
            validate_runtime_launchers(launcher_path, runtime_dir)
        except (OSError, ValueError, TypeError):
            missing.append("source/runtime-launchers.json:invalid")
    for node_rank in range(settings.nodes):
        relative = f"source/cache/node-{node_rank}-input-manifest.json"
        path = runtime_dir / relative
        if path.is_file():
            try:
                validate_cache_manifest(path, run_id, node_rank)
            except (OSError, ValueError, TypeError):
                missing.append(f"{relative}:invalid")
    expected_gpus: dict[int, dict[str, int]] = {}
    for node_rank in range(settings.nodes):
        relative = f"source/topology/node-{node_rank}.json"
        path = runtime_dir / relative
        if path.is_file():
            try:
                validate_node_topology(
                    path,
                    runtime_dir,
                    run_id,
                    node_rank,
                    expected_gpus=settings.gpus_per_node,
                )
                expected_gpus[node_rank] = topology_gpu_bindings(path, runtime_dir)
            except (OSError, ValueError, TypeError):
                missing.append(f"{relative}:invalid")
    environment_members = [
        "source/environment/job-entry.snapshot",
        *(f"source/environment/host-{node}.snapshot" for node in range(settings.nodes)),
        *(f"source/environment/sglang-{node}.snapshot" for node in range(settings.nodes)),
        *(
            f"source/environment/worker-{rank}.snapshot"
            for rank in range(settings.tensor_parallel_size)
        ),
    ]
    for relative in environment_members:
        path = runtime_dir / relative
        if path.is_file():
            try:
                validate_environment_snapshot(path)
            except (OSError, ValueError, TypeError):
                missing.append(f"{relative}:invalid")
    for node_rank in range(settings.nodes):
        for relative, validator in (
            (
                f"source/hardware/node-{node_rank}.json",
                lambda path, node=node_rank: validate_minimal_hardware(
                    path, runtime_dir, node
                ),
            ),
            (
                f"source/lifecycle/node-{node_rank}.json",
                lambda path, node=node_rank: validate_node_lifecycle(
                    path, node_rank=node
                ),
            ),
        ):
            path = runtime_dir / relative
            if path.is_file():
                try:
                    validator(path)
                except (OSError, ValueError, TypeError):
                    missing.append(f"{relative}:invalid")
    lifecycle = runtime_dir / "source/run-lifecycle.json"
    if lifecycle.is_file():
        try:
            validate_run_lifecycle(
                lifecycle,
                run_id,
                start_path=runtime_dir / "control/lifecycle/run-start.json",
            )
        except (OSError, ValueError, TypeError):
            missing.append("source/run-lifecycle.json:invalid")
    accounting = runtime_dir / "source/scheduler/accounting.json"
    if accounting.is_file():
        try:
            validate_scheduler_accounting(accounting, runtime_dir, run_id)
            _validate_scheduler_job_identity(runtime_dir)
        except (OSError, ValueError, TypeError):
            missing.append("source/scheduler/accounting.json:invalid")
    try:
        dataset_identity = hashlib.sha256(
            (runtime_dir / "source/dataset-manifest.json").read_bytes()
        ).hexdigest()
    except OSError:
        dataset_identity = ""
    for kind in ("warmup", "measured"):
        relative = f"source/workload-{kind}.json"
        path = runtime_dir / relative
        if path.is_file():
            try:
                validate_ordered_workload(
                    path, kind=kind, dataset_manifest_sha256=dataset_identity
                )
            except (OSError, ValueError, TypeError):
                missing.append(f"{relative}:invalid")
    rendezvous = runtime_dir / "control/lifecycle/rendezvous.json"
    summary = runtime_dir / "control/lifecycle/summary.json"
    if rendezvous.is_file():
        try:
            validate_lifecycle_rendezvous(
                rendezvous, run_id, settings.node_agent_port
            )
        except (OSError, ValueError, TypeError):
            missing.append("control/lifecycle/rendezvous.json:invalid")
    if summary.is_file():
        try:
            validate_lifecycle_summary(summary, run_id, settings.nodes)
        except (OSError, ValueError, TypeError):
            missing.append("control/lifecycle/summary.json:invalid")
    if purpose == "profile":
        for node_rank in range(settings.nodes):
            completion = f"control/gpu/node-{node_rank}.complete"
            if not (runtime_dir / completion).is_file():
                missing.append(completion)
            relative = f"profile/raw/gpu/node-{node_rank}.jsonl"
            path = runtime_dir / relative
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(relative)
                continue
            try:
                first = load_json_strict(path.read_bytes().splitlines()[0])
                if not isinstance(first, dict) or not isinstance(
                    first.get("clock_id"), str
                ):
                    raise TypeError("GPU raw has no clock identity")
                parsed = parse_gpu_samples_jsonl(
                    path.read_bytes(),
                    run_id=run_id,
                    node_rank=node_rank,
                    clock_id=first["clock_id"],
                    expected_gpus=expected_gpus[node_rank],
                )
                if not parsed:
                    raise ValueError("GPU raw contains no complete frame")
            except (KeyError, OSError, ValueError, TypeError):
                missing.append(f"{relative}:invalid")
        try:
            profiler = project_profiler_events(
                runtime_dir,
                run_id=run_id,
                invocation_id="measured-generation",
                expected_ranks=settings.tensor_parallel_size,
            )
            if (
                profiler.missing_ranks
                or profiler.partial_ranks
                or profiler.conflicts
            ):
                missing.append("profile/raw/torch:invalid")
        except (OSError, ValueError, TypeError):
            missing.append("profile/raw/torch:invalid")
    return missing


def _validate_job_entry(path: Path, run_id: str) -> None:
    document = load_json_strict(path.read_bytes())
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "run_id",
        "pj_job_id",
        "entered_at",
    }:
        raise ValueError("job entry is not closed")
    job_id = document["pj_job_id"]
    entered_at = document["entered_at"]
    if (
        document["schema_version"] != "job-entry/v1"
        or document["run_id"] != run_id
        or not isinstance(job_id, str)
        or re.fullmatch(r"[0-9]+", job_id) is None
        or not isinstance(entered_at, str)
        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", entered_at)
        is None
    ):
        raise ValueError("job entry identity is invalid")
    datetime.fromisoformat(entered_at.removesuffix("Z") + "+00:00")


def _validate_submission_result(path: Path, run_id: str) -> None:
    document = load_json_strict(path.read_bytes())
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "run_id",
        "job_id",
    }:
        raise ValueError("submission result is not closed")
    if (
        document["schema_version"] != "submission-result/v1"
        or document["run_id"] != run_id
        or not isinstance(document["job_id"], str)
        or re.fullmatch(r"[0-9]+", document["job_id"]) is None
    ):
        raise ValueError("submission result identity is invalid")


def _has_valid_primary_result(runtime_dir: Path, run_id: str) -> bool:
    path = runtime_dir / "source/primary-result.json"
    if not path.is_file():
        return False
    try:
        parse_primary_result(load_json_strict(path.read_bytes()), expected_run_id=run_id)
    except (OSError, ValueError, TypeError):
        return False
    return True


def _validate_instance_axis(producer_id: str, key: str, line_number: int) -> None:
    if producer_id.startswith("stage:"):
        name = producer_id.removeprefix("stage:")
        if not (
            key.startswith(f"run.stage.{name}.")
            or key == f"perf.stage.{name}.wall_time_sec"
        ):
            raise ValueError(f"line {line_number} stage instance/key mismatch")
    if producer_id.startswith("configuration:node-"):
        node = producer_id.removeprefix("configuration:node-")
        if not key.startswith(f"sglang.node.{node}."):
            raise ValueError(f"line {line_number} configuration node/key mismatch")
    if producer_id.startswith("cache:node-"):
        node = producer_id.removeprefix("cache:node-")
        if key != f"cache.node.{node}.input_manifest_sha256":
            raise ValueError(f"line {line_number} cache node/key mismatch")
    instance_axes = (
        ("hardware:node-", "hardware.node."),
        ("node-lifecycle:node-", "parallel.node."),
        ("worker:global-rank-", "parallel.worker."),
    )
    for producer_prefix, key_prefix in instance_axes:
        if producer_id.startswith(producer_prefix):
            index = producer_id.removeprefix(producer_prefix)
            if not key.startswith(f"{key_prefix}{index}."):
                raise ValueError(f"line {line_number} producer/key axis mismatch")
    if producer_id.startswith("topology:node-"):
        index = producer_id.removeprefix("topology:node-")
        if not key.startswith(
            (f"parallel.node.{index}.", f"hardware.node.{index}.gpu.")
        ):
            raise ValueError(f"line {line_number} topology node/key mismatch")
    if producer_id.startswith("workload:"):
        kind = producer_id.removeprefix("workload:")
        if key != f"workload.{kind}.manifest_sha256":
            raise ValueError(f"line {line_number} workload kind/key mismatch")


def _validate_scheduler_job_identity(runtime_dir: Path) -> None:
    accounting = load_json_strict(
        (runtime_dir / "source/scheduler/accounting.json").read_bytes()
    )
    job_entry = load_json_strict((runtime_dir / "control/job-entry.json").read_bytes())
    submission = load_json_strict(
        (runtime_dir / "control/submission-result.json").read_bytes()
    )
    if not all(isinstance(item, dict) for item in (accounting, job_entry, submission)):
        raise TypeError("scheduler identity sources must be objects")
    job_ids = {accounting["row"]["jid"], job_entry["pj_job_id"], submission["job_id"]}
    if len(job_ids) != 1:
        raise ValueError("submission, job-entry, and accounting job IDs differ")


def _configuration_axis_errors(
    facts_by_producer: dict[str, list[str]],
    observed_producers: set[str],
    nodes: int,
) -> list[str]:
    producers = tuple(f"configuration:node-{node}" for node in range(nodes))
    if not all(item in observed_producers for item in producers):
        return []
    normalized = []
    for producer in producers:
        node = producer.rsplit("-", 1)[1]
        normalized.append(
            {
                key.removeprefix(f"sglang.node.{node}.")
                for key in facts_by_producer.get(producer, ())
            }
        )
    return [] if normalized and normalized[0] and all(
        item == normalized[0] for item in normalized[1:]
    ) else ["configuration snapshots have different or empty leaf sets"]


def _runner_contract_errors(
    facts_by_producer: dict[str, list[str]], purpose: str
) -> list[str]:
    observed = set(facts_by_producer.get("runner", ()))
    expected = {
        "run.identity.run_id",
        "run.identity.run_purpose",
        "run.identity.repository_id",
        "run.identity.instrumentation_bundle_sha256",
        "source.track.head_commit",
        "source.track.manifest_sha256",
        "source.track.source_sha256",
        *{
            f"scheduler.request.{name}"
            for name in (
                "nodes",
                "mpi_processes",
                "gpus_per_node",
                "tensor_parallel_size",
                "resource_group",
                "transport_policy",
                "advertise_source",
                "sglang_port",
                "node_agent_port",
                "rendezvous_timeout_sec",
                "ready_timeout_sec",
            )
        },
        (
            "source.track.score_reference_commit"
            if purpose == "bench"
            else "source.track.profile_base_commit"
        ),
    }
    return [] if observed == expected else [
        "runner fragment does not contain the exact manifest/settings fact set"
    ]


def _placement_axis_errors(
    values: dict[str, dict[str, str]],
    observed: set[str],
    settings: ExecutionSettings,
) -> list[str]:
    topology_producers = {
        f"topology:node-{node}" for node in range(settings.nodes)
    }
    worker_producers = {
        f"worker:global-rank-{rank}"
        for rank in range(settings.tensor_parallel_size)
    }
    configuration_producers = {
        f"configuration:node-{node}" for node in range(settings.nodes)
    }
    if not (
        topology_producers | worker_producers | configuration_producers
    ).issubset(observed):
        return []
    errors: list[str] = []
    topology_bindings: dict[int, dict[int, str]] = {}
    hostnames: list[str] = []
    for node in range(settings.nodes):
        rows = values[f"topology:node-{node}"]
        prefix = f"parallel.node.{node}"
        if (
            rows.get(f"{prefix}.global_rank") != str(node)
            or rows.get(f"{prefix}.local_rank") != "0"
            or rows.get(f"{prefix}.size") != str(settings.mpi_processes)
        ):
            errors.append(f"placement node {node} outer MPI grid differs")
        hostname = rows.get(f"{prefix}.hostname")
        if hostname is not None:
            hostnames.append(hostname)
        bindings: dict[int, str] = {}
        for key, value in rows.items():
            match = re.fullmatch(
                rf"hardware\.node\.{node}\.gpu\.([0-9]+)\.uuid", key
            )
            if match is not None:
                bindings[int(match.group(1))] = value
        if set(bindings) != set(range(settings.gpus_per_node)):
            errors.append(f"placement node {node} container GPU grid differs")
        topology_bindings[node] = bindings
    if len(hostnames) != settings.nodes or len(set(hostnames)) != settings.nodes:
        errors.append("placement node hostnames are missing or duplicated")
    worker_uuids: set[str] = set()
    workers_per_node = {node: 0 for node in range(settings.nodes)}
    for rank in range(settings.tensor_parallel_size):
        rows = values[f"worker:global-rank-{rank}"]
        prefix = f"parallel.worker.{rank}"
        try:
            node = int(rows[f"{prefix}.node_rank"])
            tp_rank = int(rows[f"{prefix}.tp_rank"])
            gpu_id = int(rows[f"{prefix}.gpu_id"])
            gpu_uuid = rows[f"{prefix}.gpu_uuid"]
        except (KeyError, ValueError):
            errors.append(f"placement worker {rank} identity is incomplete")
            continue
        if node not in workers_per_node or tp_rank != rank:
            errors.append(f"placement worker {rank} rank grid differs")
            continue
        workers_per_node[node] += 1
        if topology_bindings[node].get(gpu_id) != gpu_uuid:
            errors.append(f"placement worker {rank} GPU binding differs")
        if gpu_uuid in worker_uuids:
            errors.append(f"placement worker {rank} GPU UUID is duplicated")
        worker_uuids.add(gpu_uuid)
    if any(count != settings.gpus_per_node for count in workers_per_node.values()):
        errors.append("placement worker count per node differs")
    for node in range(settings.nodes):
        rows = values[f"configuration:node-{node}"]
        prefix = f"sglang.node.{node}.server_args_effective"
        expected = {
            "tp_size": settings.tensor_parallel_size,
            "nnodes": settings.nodes,
            "node_rank": node,
        }
        for name, wanted in expected.items():
            encoded = rows.get(f"{prefix}.{name}")
            if encoded is None or load_json_strict(encoded) != wanted:
                errors.append(f"placement ServerArgs node {node} {name} differs")
        address = rows.get(f"{prefix}.dist_init_addr")
        if (
            address is None
            or not hostnames
            or load_json_strict(address) != f"{hostnames[0]}:{settings.sglang_port}"
        ):
            errors.append(f"placement ServerArgs node {node} address differs")
    return errors
