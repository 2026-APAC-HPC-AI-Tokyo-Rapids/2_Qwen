from __future__ import annotations

import base64
import dataclasses
import enum
import hashlib
import json
import math
import os
import random
import re
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import google.auth
from google.oauth2 import service_account

JSON_ENV = "EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_JSON"
BASE64_ENV = "EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_BASE64"
INSERT_SCOPE = "https://www.googleapis.com/auth/bigquery.insertdata"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
SAFE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class Metric:
    key: str
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise TypeError("metric key must be a non-empty string")
        if not isinstance(self.value, str):
            raise TypeError("metric value must be a scalar string")


@dataclass(frozen=True, slots=True)
class BigQueryConfig:
    project: str
    dataset: str
    table: str
    location: str


@dataclass(frozen=True, slots=True)
class CapturedCredentials:
    info: dict[str, object] | None
    error: Exception | None


_captured: CapturedCredentials | None = None


def warning(message: str) -> None:
    import sys

    print(f"qwen-experiment-metrics: warning: {message}", file=sys.stderr)


def scalar(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, enum.Enum):
        return scalar(value.value)
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else ""
    return str(value)


def _key_part(value: object) -> str:
    text = str(value)
    if SAFE_KEY.fullmatch(text):
        return text
    return f"encoded.{text.encode('utf-8').hex()}"


def _json_value(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    if isinstance(value, enum.Enum):
        return _json_value(value.value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def flatten(prefix: str, value: object) -> list[Metric]:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        if not value:
            return [Metric(prefix, "")]
        result: list[Metric] = []
        for key, child in value.items():
            result.extend(flatten(f"{prefix}.{_key_part(key)}", child))
        return result
    if isinstance(value, (list, tuple)):
        encoded = json.dumps(
            _json_value(value),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return [Metric(prefix, encoded)]
    return [Metric(prefix, scalar(value))]


def capture_credentials() -> None:
    global _captured
    if _captured is not None:
        return
    raw_json = os.environ.pop(JSON_ENV, None)
    raw_base64 = os.environ.pop(BASE64_ENV, None)
    try:
        if raw_json is None and raw_base64 is not None:
            raw_json = base64.b64decode(raw_base64, validate=True).decode()
        if raw_json is None:
            _captured = CapturedCredentials(None, None)
            return
        info = json.loads(raw_json)
        if not isinstance(info, dict):
            raise TypeError("Google credentials JSON must be an object")
        _captured = CapturedCredentials(info, None)
    except Exception as error:  # noqa: BLE001
        _captured = CapturedCredentials(None, error)


def _credentials() -> tuple[object, str | None]:
    capture_credentials()
    assert _captured is not None
    if _captured.error:
        raise ValueError(
            "invalid Google credentials environment variable"
        ) from _captured.error
    if _captured.info is not None:
        credentials = service_account.Credentials.from_service_account_info(
            _captured.info, scopes=[INSERT_SCOPE]
        )
        project = _captured.info.get("project_id")
        return credentials, project if isinstance(project, str) else None
    return google.auth.default(scopes=[INSERT_SCOPE])


def load_config(path: Path | None = None) -> BigQueryConfig:
    pyproject = path
    if pyproject is None:
        pyproject = next(
            candidate
            for parent in Path(__file__).resolve().parents
            if (candidate := parent / "pyproject.toml").is_file()
        )
    with pyproject.open("rb") as stream:
        section = tomllib.load(stream)["tool"]["qwen-experiment-metrics"]["bigquery"]
    values = {
        name: section[name] for name in ("project", "dataset", "table", "location")
    }
    if not all(isinstance(value, str) and value for value in values.values()):
        raise ValueError("BigQuery configuration values must be non-empty strings")
    return BigQueryConfig(**values)


def make_run(job_id: str | None = None) -> str:
    explicit = os.environ.get("EXPERIMENT_METRICS_RUN")
    if explicit:
        return explicit
    job = job_id or os.environ.get("PJM_JOBID") or os.environ.get("PJM_JOB_ID")
    github_run = os.environ.get("GITHUB_RUN_ID")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    if github_run and attempt and job:
        return f"qwen-{github_run}-{attempt}-{job}"
    if job:
        return f"qwen-manual-{job}"
    return f"qwen-local-{os.getpid()}"


def _rows(run: str, batch: Iterable[Metric]) -> list[dict[str, object]]:
    result = []
    for metric in batch:
        payload = b"\0".join((run.encode(), metric.key.encode(), metric.value.encode()))
        result.append(
            {
                "insertId": hashlib.sha256(payload).hexdigest(),
                "json": {"run": run, "key": metric.key, "value": metric.value},
            }
        )
    return result


def send(run: str, batch: list[Metric]) -> bool:
    if not batch:
        return True
    if os.environ.get("EXPERIMENT_METRICS_SINK") == "stdout":
        for metric in batch:
            print(
                json.dumps(
                    {"run": run, "key": metric.key, "value": metric.value},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        return True
    try:
        config = load_config()
        credentials, credential_project = _credentials()
        project = config.project or credential_project
        url = (
            "https://bigquery.googleapis.com/bigquery/v2/projects/"
            f"{urllib.parse.quote(project, safe='')}/datasets/"
            f"{urllib.parse.quote(config.dataset, safe='')}/tables/"
            f"{urllib.parse.quote(config.table, safe='')}/insertAll"
        )
        for offset in range(0, len(batch), 500):
            if not _send_chunk(credentials, url, run, batch[offset : offset + 500]):
                return False
        return True
    except Exception as error:  # noqa: BLE001
        warning(f"BigQuery send failed: {error}")
        return False


def _send_chunk(credentials: object, url: str, run: str, batch: list[Metric]) -> bool:
    body = json.dumps(
        {
            "skipInvalidRows": False,
            "ignoreUnknownValues": False,
            "rows": _rows(run, batch),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    for attempt in range(4):
        headers = {"Content-Type": "application/json; charset=utf-8"}
        credentials.before_request(_AuthRequest(), "POST", url, headers)
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status, response_body = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, response_body = error.code, error.read()
        except (OSError, TimeoutError) as error:
            if attempt == 3:
                warning(f"BigQuery request failed: {error}")
                return False
            _backoff(attempt)
            continue
        if status in RETRYABLE_STATUS and attempt < 3:
            _backoff(attempt)
            continue
        if not 200 <= status < 300:
            warning(
                f"BigQuery returned HTTP {status}: {response_body.decode(errors='replace')}"
            )
            return False
        response_json = json.loads(response_body or b"{}")
        if response_json.get("insertErrors"):
            warning(
                f"BigQuery rejected rows: {json.dumps(response_json['insertErrors'])}"
            )
            return False
        return True
    return False


class _AuthRequest:
    def __call__(
        self,
        url: str,
        method: str = "GET",
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        **_: object,
    ):
        request = urllib.request.Request(
            url, data=body, headers=dict(headers or {}), method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return _Response(
                    response.status, response.read(), dict(response.headers)
                )
        except urllib.error.HTTPError as error:
            return _Response(error.code, error.read(), dict(error.headers))


@dataclass(frozen=True, slots=True)
class _Response:
    status: int
    data: bytes
    headers: Mapping[str, str]


def _backoff(attempt: int) -> None:
    time.sleep((2**attempt) * random.uniform(0.5, 1.0))
