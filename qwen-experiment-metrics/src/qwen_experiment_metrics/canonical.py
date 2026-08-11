from __future__ import annotations

import base64
import math
import re
from decimal import Decimal
from typing import Any

import rfc8785

_CANONICAL_DECIMAL = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CanonicalizationError(ValueError):
    pass


def jcs_bytes(value: object) -> bytes:
    """Return strict RFC 8785 JSON canonical bytes."""

    _reject_non_json_or_nonfinite(value)
    try:
        return rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, UnicodeError) as error:
        raise CanonicalizationError(str(error)) from error


def jcs_text(value: object) -> str:
    return jcs_bytes(value).decode("utf-8")


def fact_row_bytes(run: str, key: str, value: str) -> bytes:
    """Encode the external fact row with its required run,key,value member order."""

    import json

    if not all(isinstance(item, str) for item in (run, key, value)):
        raise CanonicalizationError("fact row fields must be strings")
    try:
        return json.dumps(
            {"run": run, "key": key, "value": value},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (UnicodeEncodeError, ValueError) as error:
        raise CanonicalizationError(str(error)) from error


def encode_value(logical_type: str, value: object) -> str:
    if logical_type == "string":
        if not isinstance(value, str):
            raise CanonicalizationError("string value required")
        return value
    if logical_type == "sha256":
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise CanonicalizationError("lowercase SHA-256 value required")
        return value
    if logical_type == "bool":
        if not isinstance(value, bool):
            raise CanonicalizationError("bool value required")
        return "true" if value else "false"
    if logical_type in {"int", "uint"}:
        if isinstance(value, bool) or not isinstance(value, int):
            raise CanonicalizationError(f"{logical_type} value required")
        if logical_type == "uint" and value < 0:
            raise CanonicalizationError("uint cannot be negative")
        if value < -(1 << 63) or value > (1 << 63) - 1:
            raise CanonicalizationError(f"{logical_type} exceeds signed 64-bit range")
        return str(value)
    if logical_type == "decimal":
        return _encode_decimal(value)
    if logical_type == "jcs":
        return jcs_text(value)
    raise CanonicalizationError(f"unknown logical type: {logical_type}")


def encode_logical_value(value: object) -> str:
    """Encode a producer value without owner/type registry coupling."""

    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        return _encode_decimal(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return jcs_text(value)


def validate_encoded_value(logical_type: str, value: str) -> None:
    """Validate bytes read from a fragment without trusting its writer."""

    if logical_type == "string":
        return
    if logical_type == "sha256":
        encode_value(logical_type, value)
        return
    if logical_type == "bool":
        if value not in {"true", "false"}:
            raise CanonicalizationError("encoded bool is not canonical")
        return
    if logical_type in {"int", "uint"}:
        pattern = (
            r"^-?(?:0|[1-9][0-9]*)$"
            if logical_type == "int"
            else r"^(?:0|[1-9][0-9]*)$"
        )
        if re.fullmatch(pattern, value) is None or value == "-0":
            raise CanonicalizationError(f"encoded {logical_type} is not canonical")
        number = int(value)
        if number < -(1 << 63) or number > (1 << 63) - 1:
            raise CanonicalizationError(f"encoded {logical_type} exceeds 64-bit range")
        return
    if logical_type == "decimal":
        if _CANONICAL_DECIMAL.fullmatch(value) is not None:
            return
        parsed = load_json_strict(value)
        if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
            raise CanonicalizationError("encoded decimal is not numeric")
        if encode_value("decimal", parsed) != value:
            raise CanonicalizationError("encoded decimal is not canonical")
        return
    if logical_type == "jcs":
        parsed = load_json_strict(value)
        if jcs_text(parsed) != value:
            raise CanonicalizationError("encoded JCS value is not canonical")
        return
    raise CanonicalizationError(f"unknown logical type: {logical_type}")


def _encode_decimal(value: object) -> str:
    if isinstance(value, bool):
        raise CanonicalizationError("decimal value required")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise CanonicalizationError("decimal must be finite")
        text = format(value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        if text in {"-0", ""}:
            text = "0"
    elif isinstance(value, str):
        text = value
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise CanonicalizationError("decimal must be finite")
        return jcs_text(value)
    else:
        raise CanonicalizationError("decimal value required")
    if _CANONICAL_DECIMAL.fullmatch(text) is None:
        raise CanonicalizationError("non-canonical decimal string")
    return text


def _reject_non_json_or_nonfinite(value: object) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("JCS floats must be finite")
        return
    if isinstance(value, list):
        for child in value:
            _reject_non_json_or_nonfinite(child)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("JCS object keys must be strings")
            _reject_non_json_or_nonfinite(child)
        return
    raise CanonicalizationError(f"value is not a JCS value: {type(value).__name__}")


def load_json_strict(data: bytes | str) -> Any:
    import json

    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalizationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=closed_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise CanonicalizationError(str(error)) from error
    _reject_non_json_or_nonfinite(value)
    return value
