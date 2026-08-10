from __future__ import annotations

from decimal import Decimal

import pytest

from qwen_experiment_metrics.canonical import (
    CanonicalizationError,
    encode_value,
    jcs_text,
    load_json_strict,
)
from qwen_experiment_metrics.domain import Fact
from qwen_experiment_metrics.registry import RegistryError, load_registry


def test_jcs_is_rfc8785_and_rejects_nonfinite_values() -> None:
    assert jcs_text({"z": 1e-7, "a": [True, "雪"]}) == '{"a":[true,"雪"],"z":1e-7}'
    with pytest.raises(CanonicalizationError):
        jcs_text({"value": float("inf")})
    with pytest.raises(CanonicalizationError, match="duplicate"):
        load_json_strict('{"a":1,"a":2}')


def test_decimal_contract_is_distinct_from_jcs_float() -> None:
    assert encode_value("decimal", Decimal("12.3400")) == "12.34"
    assert encode_value("decimal", "0.0001") == "0.0001"
    assert encode_value("decimal", 1e-7) == "1e-7"
    with pytest.raises(CanonicalizationError, match="non-canonical"):
        encode_value("decimal", "01.0")


def test_machine_registry_has_complete_observation_contract() -> None:
    registry = load_registry()
    assert registry.schema_version == "qwen-metrics-registry/v1"
    assert registry.default_required_producers("bench") == {"runner", "result"}
    for fact in registry.facts:
        assert fact.authoritative_source
        assert fact.capture_moment
        assert fact.raw_destination
        assert fact.validation
        assert fact.cardinality
        assert fact.applicability
    assert registry.encode("runner", "bench", Fact("run.identity.run_id", "x")) == "x"
    with pytest.raises(RegistryError, match="cannot own"):
        registry.encode("runner", "bench", Fact("result.request.returned_count", 1))
