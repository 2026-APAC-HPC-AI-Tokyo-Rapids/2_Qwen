from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .canonical import fact_row_bytes, load_json_strict, validate_encoded_value
from .domain import Fact
from .events import parse_primary_result
from .gpu import parse_gpu_samples_jsonl
from .manifest import load_manifest
from .recording import RecorderError, producer_from_fragment
from .registry import Registry, RegistryError, load_registry


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
    required_producers = registry.default_required_producers(manifest.run_purpose)
    network_required = _has_valid_primary_result(runtime_dir, manifest.run_id)
    required_producers.update(_execution_producers(network_required))
    missing_required_files = _missing_execution_files(
        runtime_dir, manifest.run_purpose, manifest.run_id, network_required
    )

    fragment_dir = runtime_dir / "domain" / "fragments"
    observed_by_key: dict[str, list[_Observed]] = {}
    facts_by_producer: dict[str, list[str]] = {}
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
        for key, value in rows:
            producer_keys.append(key)
            observed_by_key.setdefault(key, []).append(
                _Observed(producer_id, value, path)
            )

    invalid_fragments.extend(
        _configuration_axis_errors(facts_by_producer, observed_producers)
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


def _execution_producers(network_required: bool) -> set[str]:
    result = {
        "stage:benchmark",
        "configuration:node-0",
        "configuration:node-1",
        "result",
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
    runtime_dir: Path, purpose: str, run_id: str, network_required: bool
) -> list[str]:
    required = [
        "source/primary-result.json",
        "source/topology/node-0.json",
        "source/topology/node-1.json",
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
    missing = [relative for relative in required if not (runtime_dir / relative).is_file()]
    expected_gpus: dict[int, dict[str, int]] = {}
    for node_rank in range(2):
        relative = f"source/topology/node-{node_rank}.json"
        path = runtime_dir / relative
        if path.is_file():
            try:
                expected_gpus[node_rank] = _validate_topology(path, run_id, node_rank)
            except (OSError, ValueError, TypeError):
                missing.append(f"{relative}:invalid")
    if purpose == "profile":
        for node_rank in range(2):
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
    return missing


def _has_valid_primary_result(runtime_dir: Path, run_id: str) -> bool:
    path = runtime_dir / "source/primary-result.json"
    if not path.is_file():
        return False
    try:
        parse_primary_result(load_json_strict(path.read_bytes()), expected_run_id=run_id)
    except (OSError, ValueError, TypeError):
        return False
    return True


def _validate_topology(path: Path, run_id: str, node_rank: int) -> dict[str, int]:
    document = load_json_strict(path.read_bytes())
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "run_id",
        "node_rank",
        "hostname",
        "gpus",
    }:
        raise ValueError("topology is not closed")
    if (
        document["schema_version"] != "topology/v1"
        or document["run_id"] != run_id
        or document["node_rank"] != node_rank
        or not isinstance(document["hostname"], str)
        or not document["hostname"]
    ):
        raise ValueError("topology identity mismatch")
    gpus = document["gpus"]
    if not isinstance(gpus, list) or len(gpus) != 4:
        raise ValueError("topology must contain four GPUs")
    result: dict[str, int] = {}
    for row in gpus:
        if not isinstance(row, dict) or set(row) != {
            "logical_index",
            "uuid",
            "pci_bus_id",
            "model",
            "memory_total_mib",
        }:
            raise ValueError("topology GPU is not closed")
        index = row["logical_index"]
        uuid = row["uuid"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not isinstance(uuid, str)
            or not uuid
            or uuid in result
        ):
            raise ValueError("invalid topology GPU binding")
        result[uuid] = index
    if set(result.values()) != set(range(4)):
        raise ValueError("topology logical indices are not contiguous")
    return result


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


def _configuration_axis_errors(
    facts_by_producer: dict[str, list[str]], observed_producers: set[str]
) -> list[str]:
    producers = ("configuration:node-0", "configuration:node-1")
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
    return [] if normalized[0] == normalized[1] and normalized[0] else [
        "configuration snapshots have different or empty leaf sets"
    ]
