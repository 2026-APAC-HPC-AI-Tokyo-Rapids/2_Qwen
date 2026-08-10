from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path

from .canonical import encode_value
from .domain import Fact

_PURPOSES = frozenset({"bench", "profile"})
_LOGICAL_TYPES = frozenset(
    {"string", "sha256", "bool", "int", "uint", "decimal", "jcs"}
)


class RegistryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProducerSpec:
    id: str
    instance_pattern: str
    applicability: tuple[str, ...]
    required: bool

    def matches(self, producer_id: str) -> bool:
        return re.fullmatch(self.instance_pattern, producer_id) is not None


@dataclass(frozen=True, slots=True)
class FactSpec:
    id: str
    key_pattern: str
    owner: str
    logical_type: str
    unit: str
    cardinality: str
    authoritative_source: str
    capture_moment: str
    raw_destination: str
    validation: str
    applicability: tuple[str, ...]
    required: bool

    def matches(self, key: str) -> bool:
        return re.fullmatch(self.key_pattern, key) is not None


@dataclass(frozen=True, slots=True)
class Registry:
    schema_version: str
    producers: tuple[ProducerSpec, ...]
    facts: tuple[FactSpec, ...]

    def producer(self, producer_id: str, purpose: str) -> ProducerSpec:
        matches = [item for item in self.producers if item.matches(producer_id)]
        if len(matches) != 1:
            raise RegistryError(
                f"producer {producer_id!r} matched {len(matches)} registry entries"
            )
        producer = matches[0]
        if purpose not in producer.applicability:
            raise RegistryError(
                f"producer {producer_id!r} is not applicable to {purpose}"
            )
        return producer

    def fact(self, key: str, purpose: str) -> FactSpec:
        matches = [item for item in self.facts if item.matches(key)]
        if len(matches) != 1:
            raise RegistryError(f"fact {key!r} matched {len(matches)} registry entries")
        fact = matches[0]
        if purpose not in fact.applicability:
            raise RegistryError(f"fact {key!r} is not applicable to {purpose}")
        return fact

    def encode(self, producer_id: str, purpose: str, fact: Fact) -> str:
        producer = self.producer(producer_id, purpose)
        spec = self.fact(fact.key, purpose)
        if spec.owner != producer.id:
            raise RegistryError(
                f"producer {producer_id!r} cannot own {fact.key!r}; owner={spec.owner!r}"
            )
        return encode_value(spec.logical_type, fact.logical_value)

    def default_required_producers(self, purpose: str) -> set[str]:
        result: set[str] = set()
        for producer in self.producers:
            if not producer.required or purpose not in producer.applicability:
                continue
            exact = _exact_regex_value(producer.instance_pattern)
            if exact is not None:
                result.add(exact)
        return result

    def required_fact_specs(
        self, producer: ProducerSpec, purpose: str
    ) -> tuple[FactSpec, ...]:
        return tuple(
            fact
            for fact in self.facts
            if fact.owner == producer.id
            and fact.required
            and purpose in fact.applicability
        )


def load_registry(path: Path | None = None) -> Registry:
    if path is not None:
        with path.open("rb") as stream:
            return _parse_registry(tomllib.load(stream))
    resource = files("qwen_experiment_metrics.contracts").joinpath("qwen-v2.toml")
    with as_file(resource) as registry_path, registry_path.open("rb") as stream:
        return _parse_registry(tomllib.load(stream))


def _parse_registry(document: object) -> Registry:
    if not isinstance(document, dict):
        raise RegistryError("registry must be a TOML table")
    _require_exact_keys(document, {"schema_version", "producers", "facts"}, "registry")
    if document["schema_version"] != "qwen-metrics-registry/v1":
        raise RegistryError("unsupported registry schema")
    producer_rows = document["producers"]
    fact_rows = document["facts"]
    if not isinstance(producer_rows, list) or not isinstance(fact_rows, list):
        raise RegistryError("producers and facts must be arrays of tables")
    producers = tuple(_parse_producer(row) for row in producer_rows)
    facts = tuple(_parse_fact(row) for row in fact_rows)
    _require_unique((item.id for item in producers), "producer id")
    _require_unique((item.id for item in facts), "fact id")
    producer_ids = {item.id for item in producers}
    for fact in facts:
        if fact.owner not in producer_ids:
            raise RegistryError(f"unknown owner {fact.owner!r} for {fact.id!r}")
    return Registry("qwen-metrics-registry/v1", producers, facts)


def _parse_producer(row: object) -> ProducerSpec:
    if not isinstance(row, dict):
        raise RegistryError("producer entry must be a table")
    expected = {"id", "instance_pattern", "applicability", "required"}
    _require_exact_keys(row, expected, "producer")
    applicability = _parse_applicability(row["applicability"])
    if not isinstance(row["id"], str) or not row["id"]:
        raise RegistryError("producer id must be non-empty")
    if not isinstance(row["instance_pattern"], str):
        raise RegistryError("producer instance_pattern must be a string")
    _compile_full_regex(row["instance_pattern"])
    if not isinstance(row["required"], bool):
        raise RegistryError("producer required must be bool")
    return ProducerSpec(
        row["id"], row["instance_pattern"], applicability, row["required"]
    )


def _parse_fact(row: object) -> FactSpec:
    if not isinstance(row, dict):
        raise RegistryError("fact entry must be a table")
    expected = {
        "id",
        "key_pattern",
        "owner",
        "logical_type",
        "unit",
        "cardinality",
        "authoritative_source",
        "capture_moment",
        "raw_destination",
        "validation",
        "applicability",
        "required",
    }
    _require_exact_keys(row, expected, "fact")
    for name in expected - {"applicability", "required"}:
        if not isinstance(row[name], str) or not row[name]:
            raise RegistryError(f"fact {name} must be a non-empty string")
    if row["logical_type"] not in _LOGICAL_TYPES:
        raise RegistryError(f"unknown logical type: {row['logical_type']!r}")
    if not isinstance(row["required"], bool):
        raise RegistryError("fact required must be bool")
    _compile_full_regex(row["key_pattern"])
    return FactSpec(
        id=row["id"],
        key_pattern=row["key_pattern"],
        owner=row["owner"],
        logical_type=row["logical_type"],
        unit=row["unit"],
        cardinality=row["cardinality"],
        authoritative_source=row["authoritative_source"],
        capture_moment=row["capture_moment"],
        raw_destination=row["raw_destination"],
        validation=row["validation"],
        applicability=_parse_applicability(row["applicability"]),
        required=row["required"],
    )


def _parse_applicability(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RegistryError("applicability must be a non-empty array")
    if any(not isinstance(item, str) or item not in _PURPOSES for item in value):
        raise RegistryError("applicability contains an unknown purpose")
    if len(set(value)) != len(value):
        raise RegistryError("applicability contains duplicates")
    return tuple(value)


def _compile_full_regex(pattern: str) -> None:
    try:
        re.compile(pattern)
    except re.error as error:
        raise RegistryError(f"invalid regex {pattern!r}: {error}") from error


def _require_exact_keys(
    row: dict[object, object], expected: set[str], name: str
) -> None:
    keys = set(row)
    if keys != expected:
        raise RegistryError(
            f"{name} fields differ: missing={sorted(expected - keys)}, "
            f"extra={sorted(keys - expected)}"
        )


def _require_unique(values: object, label: str) -> None:
    seen: set[str] = set()
    for value in values:  # type: ignore[union-attr]
        if value in seen:
            raise RegistryError(f"duplicate {label}: {value}")
        seen.add(value)


def _exact_regex_value(pattern: str) -> str | None:
    if pattern.startswith("^") and pattern.endswith("$"):
        middle = pattern[1:-1]
        if re.fullmatch(r"[A-Za-z0-9_.:-]+", middle):
            return middle
    return None
