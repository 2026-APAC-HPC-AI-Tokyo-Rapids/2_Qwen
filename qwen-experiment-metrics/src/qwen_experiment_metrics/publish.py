from __future__ import annotations

import hashlib
import json
import os
import tomllib
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .artifact_store import ROLE_FILENAMES
from .canonical import load_json_strict

GOOGLE_CREDENTIALS_ENV = "EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_JSON"
BIGQUERY_INSERT_SCOPE = "https://www.googleapis.com/auth/bigquery.insertdata"


@dataclass(frozen=True, slots=True)
class BigQueryConfig:
    project: str
    dataset: str
    table: str
    location: str

    @classmethod
    def load(cls, pyproject: Path) -> BigQueryConfig:
        with pyproject.open("rb") as stream:
            document = tomllib.load(stream)
        section = document["tool"]["qwen-experiment-metrics"]["bigquery"]
        values = tuple(
            section[name] for name in ("project", "dataset", "table", "location")
        )
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError("BigQuery configuration values must be non-empty strings")
        return cls(*values)


@dataclass(frozen=True, slots=True)
class RawRow:
    run: str
    key: str
    value: str

    def insert_row(self) -> dict[str, object]:
        identity = b"\0".join(
            (self.run.encode(), self.key.encode(), self.value.encode())
        )
        return {
            "insertId": hashlib.sha256(identity).hexdigest(),
            "json": {"run": self.run, "key": self.key, "value": self.value},
        }


def read_metrics(path: Path, *, expected_run: str | None = None) -> tuple[RawRow, ...]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("metrics.jsonl must end with LF")
    rows: list[RawRow] = []
    for number, encoded in enumerate(raw.splitlines(), start=1):
        try:
            item = load_json_strict(encoded)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"invalid metrics row {number}") from error
        if not isinstance(item, dict) or list(item) != ["run", "key", "value"]:
            raise ValueError(f"metrics row {number} must contain ordered run/key/value")
        if not all(isinstance(item[name], str) for name in ("run", "key", "value")):
            raise ValueError(f"metrics row {number} values must be strings")
        if not item["run"] or not item["key"]:
            raise ValueError(f"metrics row {number} has an empty identity")
        if expected_run is not None and item["run"] != expected_run:
            raise ValueError(f"metrics row {number} belongs to a foreign run")
        rows.append(RawRow(item["run"], item["key"], item["value"]))
    if not rows:
        raise ValueError("metrics.jsonl is empty")
    if len({row.run for row in rows}) != 1:
        raise ValueError("metrics.jsonl contains multiple runs")
    return tuple(rows)


def publish_metrics(
    path: Path,
    config: BigQueryConfig,
    *,
    artifact_manifest: Path,
    expected_run: str | None = None,
    request: Callable[[urllib.request.Request], bytes] | None = None,
) -> int:
    """Publish an immutable sidecar once; replay policy remains runner-owned."""

    rows = read_metrics(path, expected_run=expected_run)
    run_id = rows[0].run
    _verify_artifact_binding(path, artifact_manifest, run_id)
    token = None if request is not None else _access_token()
    endpoint = (
        "https://bigquery.googleapis.com/bigquery/v2/projects/"
        f"{urllib.parse.quote(config.project, safe='')}/datasets/"
        f"{urllib.parse.quote(config.dataset, safe='')}/tables/"
        f"{urllib.parse.quote(config.table, safe='')}/insertAll"
    )
    sender = request or _urlopen
    published = 0
    for chunk in _chunks(rows, 500):
        body = json.dumps(
            {
                "skipInvalidRows": False,
                "ignoreUnknownValues": False,
                "rows": [row.insert_row() for row in chunk],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        response = sender(
            urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        )
        payload = json.loads(response or b"{}")
        if payload.get("insertErrors"):
            raise RuntimeError(
                "BigQuery rejected rows: "
                + json.dumps(payload["insertErrors"], ensure_ascii=False)
            )
        published += len(chunk)
    return published


def _verify_artifact_binding(
    metrics_path: Path, artifact_manifest: Path, run_id: str
) -> None:
    if artifact_manifest.is_symlink() or not artifact_manifest.is_file():
        raise ValueError("artifact manifest must be a regular file")
    document = load_json_strict(artifact_manifest.read_bytes())
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "run_id",
        "logical_artifact_name",
        "run_purpose",
        "files",
    }:
        raise ValueError("artifact manifest is not a closed v1 object")
    if (
        document["schema_version"] != "artifact-manifest/v1"
        or document["run_id"] != run_id
        or document["logical_artifact_name"] != run_id
    ):
        raise ValueError("artifact manifest run binding failed")
    files = document["files"]
    if not isinstance(files, list):
        raise TypeError("artifact manifest files must be an array")
    run_purpose = document["run_purpose"]
    expected_roles = {
        "bench": ["run_bundle", "metrics"],
        "profile": ["run_bundle", "metrics", "profile_raw"],
    }.get(run_purpose)
    if expected_roles is None:
        raise ValueError("artifact manifest run purpose is invalid")
    if [row.get("role") if isinstance(row, dict) else None for row in files] != (
        expected_roles
    ):
        raise ValueError("artifact manifest role order does not match run purpose")
    by_role: dict[str, dict[str, object]] = {}
    for row in files:
        if not isinstance(row, dict) or set(row) != {
            "role",
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("artifact manifest file is not closed")
        role = row["role"]
        relative = row["path"]
        if not isinstance(role, str) or not isinstance(relative, str) or role in by_role:
            raise ValueError("artifact manifest roles must be unique strings")
        if relative != ROLE_FILENAMES.get(role):
            raise ValueError(f"artifact role {role} has the wrong fixed filename")
        candidate = artifact_manifest.parent / relative
        if (
            candidate.parent != artifact_manifest.parent
            or candidate.is_symlink()
            or not candidate.is_file()
        ):
            raise ValueError("artifact manifest path escapes its artifact set")
        if (
            isinstance(row["size_bytes"], bool)
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] != candidate.stat().st_size
            or row["sha256"] != _sha256_file(candidate)
        ):
            raise ValueError(f"artifact role {role} failed size/hash verification")
        by_role[role] = row
    if metrics_path.is_symlink() or metrics_path.resolve() != (
        artifact_manifest.parent / ROLE_FILENAMES["metrics"]
    ).resolve():
        raise ValueError("metrics path is not the artifact manifest metrics role")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _access_token() -> str:
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account

    raw = os.environ.pop(GOOGLE_CREDENTIALS_ENV, None)
    if raw is not None:
        info = json.loads(raw)
        if not isinstance(info, dict):
            raise ValueError("Google credentials JSON must be an object")
        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=[BIGQUERY_INSERT_SCOPE]
        )
    else:
        credentials, _ = google.auth.default(scopes=[BIGQUERY_INSERT_SCOPE])
    credentials.refresh(google.auth.transport.requests.Request())
    if not credentials.token:
        raise RuntimeError("Google credentials returned no access token")
    return credentials.token


def _urlopen(request: urllib.request.Request) -> bytes:
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _chunks(rows: Iterable[RawRow], size: int) -> Iterator[tuple[RawRow, ...]]:
    chunk: list[RawRow] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) == size:
            yield tuple(chunk)
            chunk.clear()
    if chunk:
        yield tuple(chunk)
