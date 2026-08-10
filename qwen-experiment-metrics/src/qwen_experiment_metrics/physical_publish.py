"""Physical BigQuery publisher for closed v2 projection rows.

This adapter is deliberately separate from the existing raw-fact publisher.
It has no registry/CLI side effects: callers pass a conflict-checked
``ProjectionBatch`` and an explicit configuration.  The default sender uses
the same lazy credential boundary as ``publish.py``; tests and dry-runs never
need Google credentials or network access.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .canonical import jcs_bytes
from .projections import RELATION_REGISTRY, ProjectionBatch, ProjectionRow
from .publish import _access_token, _urlopen

INSERT_CHUNK_SIZE = 500
TABLE_ALLOWLIST = {relation: relation for relation in RELATION_REGISTRY}


class PhysicalPublishError(RuntimeError):
    """BigQuery rejected a projection batch or the request was malformed."""


@dataclass(frozen=True, slots=True)
class PhysicalPublisherConfig:
    project: str
    dataset: str
    location: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (
            self.project, self.dataset, self.location
        )):
            raise ValueError("BigQuery physical configuration is not closed")


@dataclass(frozen=True, slots=True)
class PhysicalPublishReport:
    relation_rows: int
    request_count: int
    dry_run: bool


RequestSender = Callable[[urllib.request.Request], bytes]


def physical_table_name(relation_id: str) -> str:
    """Resolve only registry-owned relation IDs to physical table names."""

    try:
        return TABLE_ALLOWLIST[relation_id]
    except KeyError as error:
        raise PhysicalPublishError(f"relation is not in the table allowlist: {relation_id}") from error


def projection_json_row(row: ProjectionRow) -> dict[str, object]:
    """Return the exact closed projection object sent as BigQuery JSON."""

    relation = RELATION_REGISTRY[row.relation_id]
    value = row.to_projection_object()
    expected = {"schema_version", "run_id", *relation.fields}
    if relation.lineage:
        expected.update({"source_member", "source_sha256"})
    if set(value) != expected:
        raise PhysicalPublishError(f"projection row is not closed for {row.relation_id}")
    return value


def deterministic_insert_id(row: ProjectionRow) -> str:
    """Derive a replay-stable insertId from table identity and row bytes."""

    identity = jcs_bytes([physical_table_name(row.relation_id), projection_json_row(row)])
    return hashlib.sha256(identity).hexdigest()


def build_insert_body(rows: Iterable[ProjectionRow]) -> bytes:
    values = tuple(rows)
    if not values:
        raise ValueError("cannot build an empty BigQuery insert body")
    relation = values[0].relation_id
    if any(row.relation_id != relation for row in values):
        raise ValueError("one insertAll body cannot contain multiple physical tables")
    body = {
        "skipInvalidRows": False,
        "ignoreUnknownValues": False,
        "rows": [
            {"insertId": deterministic_insert_id(row), "json": projection_json_row(row)}
            for row in values
        ],
    }
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def publish_projection_batch(
    batch: ProjectionBatch,
    config: PhysicalPublisherConfig,
    *,
    sender: RequestSender | None = None,
    dry_run: bool = False,
) -> PhysicalPublishReport:
    """Insert a conflict-free batch into allowlisted tables in 500-row chunks.

    ``dry_run`` renders and validates every request body but never resolves a
    credential and never opens a network connection.  Non-dry runs resolve the
    raw credential only inside the existing publisher boundary.
    """

    rows = batch.require_conflict_free()
    grouped: dict[str, list[ProjectionRow]] = {}
    for row in rows:
        physical_table_name(row.relation_id)
        projection_json_row(row)
        grouped.setdefault(row.relation_id, []).append(row)
    chunks = tuple(
        chunk
        for relation_id in sorted(grouped, key=lambda item: item.encode())
        for chunk in _chunks(grouped[relation_id], INSERT_CHUNK_SIZE)
    )
    if dry_run:
        for chunk in chunks:
            build_insert_body(chunk)
        return PhysicalPublishReport(len(rows), len(chunks), True)
    request_fn = sender or _urlopen
    token = None if sender is not None else _access_token()
    for chunk in chunks:
        relation_id = chunk[0].relation_id
        endpoint = _endpoint(config, physical_table_name(relation_id))
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        response = request_fn(urllib.request.Request(
            endpoint, data=build_insert_body(chunk), headers=headers, method="POST"
        ))
        payload = _response_json(response)
        errors = payload.get("insertErrors")
        if errors:
            raise PhysicalPublishError(
                "BigQuery rejected projection rows: "
                + json.dumps(errors, ensure_ascii=False, sort_keys=True)
            )
    return PhysicalPublishReport(len(rows), len(chunks), False)


def _endpoint(config: PhysicalPublisherConfig, table: str) -> str:
    return (
        "https://bigquery.googleapis.com/bigquery/v2/projects/"
        f"{urllib.parse.quote(config.project, safe='')}/datasets/"
        f"{urllib.parse.quote(config.dataset, safe='')}/tables/"
        f"{urllib.parse.quote(table, safe='')}/insertAll"
    )


def _response_json(response: bytes | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(response, Mapping):
        payload = dict(response)
    else:
        try:
            decoded = json.loads(response or b"{}")
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PhysicalPublishError("BigQuery response is not JSON") from error
        if not isinstance(decoded, dict):
            raise PhysicalPublishError("BigQuery response is not an object")
        payload = decoded
    return payload


def _chunks(rows: Iterable[ProjectionRow], size: int) -> Iterable[tuple[ProjectionRow, ...]]:
    chunk: list[ProjectionRow] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) == size:
            yield tuple(chunk)
            chunk.clear()
    if chunk:
        yield tuple(chunk)
