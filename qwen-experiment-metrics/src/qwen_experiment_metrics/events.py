from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from .canonical import encode_value, jcs_bytes
from .domain import Fact
from .manifest import RUN_ID_PATTERN, SHA256_PATTERN

_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")


class DomainEventError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DomainEvent:
    run_id: str
    producer_id: str
    node_rank: int
    sequence: int
    event_type: str
    payload: dict[str, object]
    global_rank: int | None = None
    schema_version: str = "qwen-domain-event/v1"


@dataclass(frozen=True, slots=True)
class ParsedEvent:
    producer_instance: str
    envelope_producer_id: str
    sequence: int
    facts: tuple[Fact, ...]


def parse_domain_event(value: object, *, expected_run_id: str) -> DomainEvent:
    if not isinstance(value, dict):
        raise DomainEventError("domain event must be an object")
    required = {
        "schema_version",
        "run_id",
        "producer_id",
        "node_rank",
        "sequence",
        "event_type",
        "payload",
    }
    optional = {"global_rank"}
    _closed_fields(value, required, optional, "domain event")
    if value["schema_version"] != "qwen-domain-event/v1":
        raise DomainEventError("unsupported domain event schema")
    run_id = _string(value["run_id"], "run_id")
    if run_id != expected_run_id or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise DomainEventError("domain event run_id does not match manifest")
    producer_id = _string(value["producer_id"], "producer_id")
    node_rank = _uint(value["node_rank"], "node_rank")
    sequence = _uint(value["sequence"], "sequence")
    event_type = _string(value["event_type"], "event_type")
    payload = value["payload"]
    if not isinstance(payload, dict):
        raise DomainEventError("payload must be an object")
    global_rank_value = value.get("global_rank")
    global_rank = (
        None if global_rank_value is None else _uint(global_rank_value, "global_rank")
    )
    return DomainEvent(
        run_id=run_id,
        producer_id=producer_id,
        node_rank=node_rank,
        global_rank=global_rank,
        sequence=sequence,
        event_type=event_type,
        payload=payload,
    )


def facts_from_event(value: object, *, expected_run_id: str) -> ParsedEvent:
    event = parse_domain_event(value, expected_run_id=expected_run_id)
    if event.event_type in {"bench_args_effective", "server_args_effective"}:
        facts = parse_configuration_payload(
            event.payload,
            node_rank=event.node_rank,
            event_type=event.event_type,
        )
        return ParsedEvent(
            producer_instance=f"configuration:node-{event.node_rank}",
            envelope_producer_id=event.producer_id,
            sequence=event.sequence,
            facts=facts,
        )
    if event.event_type == "primary_result":
        if event.node_rank != 0 or event.global_rank not in {None, 0}:
            raise DomainEventError("primary result must be emitted by rank 0")
        facts = parse_primary_result(event.payload, expected_run_id=expected_run_id)
        return ParsedEvent(
            producer_instance="result",
            envelope_producer_id=event.producer_id,
            sequence=event.sequence,
            facts=facts,
        )
    if event.event_type == "worker_ready":
        if event.global_rank is None:
            raise DomainEventError("worker_ready requires global_rank")
        facts = parse_worker_ready_payload(
            event.payload,
            node_rank=event.node_rank,
            global_rank=event.global_rank,
        )
        return ParsedEvent(
            producer_instance=f"worker:global-rank-{event.global_rank}",
            envelope_producer_id=event.producer_id,
            sequence=event.sequence,
            facts=facts,
        )
    raise DomainEventError(f"unsupported domain event type: {event.event_type}")


def parse_worker_ready_payload(
    value: object, *, node_rank: int, global_rank: int
) -> tuple[Fact, ...]:
    """Validate the closed worker-ready payload without inferring placement."""

    if not isinstance(value, dict):
        raise DomainEventError("worker_ready payload must be an object")
    expected = {
        "tp_rank",
        "gpu_id",
        "gpu_uuid",
        "cpu_affinity",
        "numa_mems",
        "prefill_attention_backend",
        "decode_attention_backend",
        "torch_distributed_backend",
        "transport_backend",
    }
    _closed_fields(value, expected, set(), "worker_ready payload")
    tp_rank = _uint(value["tp_rank"], "tp_rank")
    gpu_id = _uint(value["gpu_id"], "gpu_id")
    if tp_rank != global_rank:
        raise DomainEventError("worker tp_rank must equal envelope global_rank")
    affinity = value["cpu_affinity"]
    if (
        not isinstance(affinity, list)
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in affinity)
        or affinity != sorted(set(affinity))
    ):
        raise DomainEventError("worker cpu_affinity must be sorted unique uints")
    strings = {
        name: _string(value[name], name)
        for name in (
            "gpu_uuid",
            "numa_mems",
            "prefill_attention_backend",
            "decode_attention_backend",
            "torch_distributed_backend",
            "transport_backend",
        )
    }
    if any(not child for child in strings.values()):
        raise DomainEventError("worker_ready string fields must be nonempty")
    prefix = f"parallel.worker.{global_rank}"
    return (
        Fact(f"{prefix}.node_rank", node_rank),
        Fact(f"{prefix}.tp_rank", tp_rank),
        Fact(f"{prefix}.gpu_id", gpu_id),
        Fact(f"{prefix}.gpu_uuid", strings["gpu_uuid"]),
        Fact(f"{prefix}.cpu_affinity", affinity),
        Fact(f"{prefix}.numa_mems", strings["numa_mems"]),
        Fact(
            f"{prefix}.attention.prefill_backend",
            strings["prefill_attention_backend"],
        ),
        Fact(
            f"{prefix}.attention.decode_backend",
            strings["decode_attention_backend"],
        ),
        Fact(
            f"{prefix}.torch_distributed_backend",
            strings["torch_distributed_backend"],
        ),
        Fact(f"{prefix}.transport_backend", strings["transport_backend"]),
    )


def parse_configuration_payload(
    value: object, *, node_rank: int, event_type: str
) -> tuple[Fact, ...]:
    if not isinstance(value, dict):
        raise DomainEventError("configuration payload must be an object")
    _closed_fields(
        value,
        {"schema_version", "config_type", "definition", "fields"},
        set(),
        "configuration payload",
    )
    if value["schema_version"] != "config-snapshot/v1":
        raise DomainEventError("unsupported configuration schema")
    expected_config_type = {
        "bench_args_effective": "BenchArgs",
        "server_args_effective": "ServerArgs",
    }.get(event_type)
    if expected_config_type is None or value["config_type"] != expected_config_type:
        raise DomainEventError("configuration type does not match event type")
    definition = value["definition"]
    if not isinstance(definition, dict):
        raise DomainEventError("configuration definition must be an object")
    _closed_fields(
        definition,
        {"source_path", "class_name", "source_blob_oid"},
        set(),
        "configuration definition",
    )
    for name in ("source_path", "class_name"):
        if not _string(definition[name], name):
            raise DomainEventError(f"{name} cannot be empty")
    source_path = _string(definition["source_path"], "source_path")
    if source_path.startswith("/") or ".." in source_path.split("/"):
        raise DomainEventError("source_path must be repository-relative")
    if (
        _OID.fullmatch(_string(definition["source_blob_oid"], "source_blob_oid"))
        is None
    ):
        raise DomainEventError("source_blob_oid must be a full Git object ID")
    rows = value["fields"]
    if not isinstance(rows, list):
        raise DomainEventError("configuration fields must be an array")
    names: list[str] = []
    facts: list[Fact] = []
    for row in rows:
        if not isinstance(row, dict):
            raise DomainEventError("configuration field must be an object")
        _closed_fields(row, {"name", "declared_type", "value"}, set(), "field")
        name = _string(row["name"], "field name")
        declared_type = _string(row["declared_type"], "declared_type")
        if not name or not declared_type:
            raise DomainEventError("field name and declared_type cannot be empty")
        # Canonicalization also proves that the dynamic value is valid JCS.
        jcs_bytes(row["value"])
        names.append(name)
        facts.append(
            Fact(
                f"sglang.node.{node_rank}.{event_type}.{escape_key_component(name)}",
                row["value"],
            )
        )
    if names != sorted(names, key=lambda item: item.encode("utf-8")):
        raise DomainEventError("configuration fields must be name-byte sorted")
    if len(names) != len(set(names)):
        raise DomainEventError("configuration field names must be unique")
    return tuple(facts)


def parse_primary_result(value: object, *, expected_run_id: str) -> tuple[Fact, ...]:
    if not isinstance(value, dict):
        raise DomainEventError("primary result must be an object")
    expected = {
        "schema_version",
        "run_id",
        "invocation_id",
        "generation_wall_sec",
        "returned_count",
        "error_count",
        "observed_input_tokens",
        "observed_output_tokens",
        "ordered_output_token_ids_sha256",
    }
    _closed_fields(value, expected, set(), "primary result")
    if value["schema_version"] != "primary-result/v1":
        raise DomainEventError("unsupported primary result schema")
    if value["run_id"] != expected_run_id:
        raise DomainEventError("primary result run_id does not match manifest")
    if not _string(value["invocation_id"], "invocation_id"):
        raise DomainEventError("invocation_id cannot be empty")
    wall = value["generation_wall_sec"]
    if isinstance(wall, bool) or not isinstance(wall, (int, float, str)):
        raise DomainEventError("generation_wall_sec must be a decimal")
    try:
        encoded_wall = encode_value("decimal", wall)
        decimal_wall = Decimal(encoded_wall)
        if not decimal_wall.is_finite() or decimal_wall <= 0:
            raise DomainEventError("generation_wall_sec must be positive")
    except (ValueError, OverflowError) as error:
        raise DomainEventError("generation_wall_sec must be finite decimal") from error
    facts = [
        Fact("perf.benchmark.generation.wall_time_sec", wall),
        Fact(
            "result.request.returned_count",
            _uint(value["returned_count"], "returned_count"),
        ),
        Fact("result.request.error_count", _uint(value["error_count"], "error_count")),
    ]
    for source, key in (
        ("observed_input_tokens", "result.token.input.total"),
        ("observed_output_tokens", "result.token.output.total"),
    ):
        child = value[source]
        if child is not None:
            facts.append(Fact(key, _uint(child, source)))
    identity = value["ordered_output_token_ids_sha256"]
    if identity is not None:
        if not isinstance(identity, str) or SHA256_PATTERN.fullmatch(identity) is None:
            raise DomainEventError("ordered output identity must be lowercase SHA-256")
        facts.append(Fact("result.token.output.ordered_token_ids_sha256", identity))
    return tuple(facts)


def escape_key_component(value: str) -> str:
    if _SAFE_COMPONENT.fullmatch(value):
        return value
    return f"encoded.{value.encode('utf-8').hex()}"


def _closed_fields(
    value: dict[str, object], required: set[str], optional: set[str], label: str
) -> None:
    actual = set(value)
    if not required.issubset(actual) or not actual.issubset(required | optional):
        raise DomainEventError(
            f"{label} fields differ: missing={sorted(required - actual)}, "
            f"extra={sorted(actual - required - optional)}"
        )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise DomainEventError(f"{label} must be a string")
    return value


def _uint(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DomainEventError(f"{label} must be a nonnegative integer")
    return value
