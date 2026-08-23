from __future__ import annotations

from pathlib import Path

from .canonical import jcs_bytes, load_json_strict
from .domain import Fact


def validate_ordered_workload(path: Path, *, kind: str) -> tuple[Fact, ...]:
    value = load_json_strict(path.read_bytes())
    expected = {"schema_version", "kind", "requests"}
    if kind == "warmup":
        expected.add("mode")
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("ordered workload is not closed")
    if value["schema_version"] != "ordered-workload/v1" or value["kind"] != kind:
        raise ValueError("ordered workload identity differs")
    if kind == "warmup" and value["mode"] not in {"random", "dataset"}:
        raise ValueError("warmup mode is invalid")
    requests = value["requests"]
    if not isinstance(requests, list) or not requests:
        raise ValueError("ordered workload requests must be a nonempty array")
    for request in requests:
        _validate_request(request)
    # The ordered array is the single raw owner.  Reducers expose only its
    # semantic identity; count/totals remain query-time derivations.
    return (Fact(f"workload.{kind}.manifest_sha256", _sha256_jcs(value)),)


def _validate_request(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "source",
        "prompt_payload",
        "requested_output_token_count",
        "sampling_params",
    }:
        raise ValueError("ordered workload request is not closed")
    source = value["source"]
    if not isinstance(source, dict):
        raise TypeError("workload source must be an object")
    if source.get("kind") == "dataset_row":
        if set(source) != {"kind", "array_index"}:
            raise ValueError("dataset-row source is not closed")
        _uint(source["array_index"], "dataset array index")
    elif source.get("kind") == "generated":
        if set(source) != {"kind", "generator_id", "seed", "generator_ordinal"}:
            raise ValueError("generated source is not closed")
        _nonempty(source["generator_id"], "generator ID")
        _uint(source["seed"], "generator seed")
        _uint(source["generator_ordinal"], "generator ordinal")
    else:
        raise ValueError("workload source kind is invalid")
    prompt = value["prompt_payload"]
    if not isinstance(prompt, dict):
        raise TypeError("prompt payload must be an object")
    if prompt.get("kind") == "text":
        if set(prompt) != {"kind", "text", "input_token_count"}:
            raise ValueError("text prompt payload is not closed")
        if not isinstance(prompt["text"], str):
            raise TypeError("prompt text must be a string")
        _uint(prompt["input_token_count"], "input token count")
    elif prompt.get("kind") == "token_ids":
        if set(prompt) != {"kind", "token_ids"}:
            raise ValueError("token-id prompt payload is not closed")
        token_ids = prompt["token_ids"]
        if not isinstance(token_ids, list):
            raise TypeError("prompt token_ids must be an array")
        for token in token_ids:
            if isinstance(token, bool) or not isinstance(token, int):
                raise TypeError("prompt token IDs must be integers")
    else:
        raise ValueError("prompt payload kind is invalid")
    _uint(value["requested_output_token_count"], "requested output tokens")
    parameters = value["sampling_params"]
    if not isinstance(parameters, list):
        raise TypeError("sampling params must be an array")
    names: list[str] = []
    for parameter in parameters:
        if not isinstance(parameter, dict) or set(parameter) != {
            "name",
            "declared_type",
            "value",
        }:
            raise ValueError("sampling parameter is not closed")
        names.append(_nonempty(parameter["name"], "sampling parameter name"))
        _nonempty(parameter["declared_type"], "sampling parameter declared type")
        jcs_bytes(parameter["value"])
    if names != sorted(set(names), key=lambda item: item.encode()):
        raise ValueError("sampling parameter names must be byte-sorted and unique")


def _sha256_jcs(value: object) -> str:
    import hashlib

    return hashlib.sha256(jcs_bytes(value)).hexdigest()


def _uint(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{label} must be a nonnegative integer")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a nonempty string")
    return value
