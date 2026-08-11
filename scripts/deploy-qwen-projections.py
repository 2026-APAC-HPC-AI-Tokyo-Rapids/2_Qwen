#!/usr/bin/env python3
"""Render or apply the v2 physical projection table DDL.

Rendering is the default and is network/credential free.  ``--apply`` is an
explicit opt-in that submits Standard SQL through the BigQuery REST API.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]+$")
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DDL = _ROOT / "sql/qwen_projection_tables_v2.sql"
_DEFAULT_REPLAY = _ROOT / "sql/qwen_projection_replay_views_v2.sql"
_BIGQUERY_SCOPE = "https://www.googleapis.com/auth/bigquery"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deploy-qwen-projections")
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--location", default="asia-northeast1")
    parser.add_argument("--ddl", type=Path, default=_DEFAULT_DDL)
    parser.add_argument("--replay-sql", type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser


def render(ddl: str, *, project: str, dataset: str) -> str:
    for label, value in (("project", project), ("dataset", dataset)):
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError(f"{label} is not a safe BigQuery identifier")
    if "__PROJECT__" not in ddl or "__DATASET__" not in ddl:
        raise ValueError("DDL is missing its explicit render placeholders")
    rendered = ddl.replace("__PROJECT__", project).replace("__DATASET__", dataset)
    if "__PROJECT__" in rendered or "__DATASET__" in rendered:
        raise ValueError("DDL placeholder replacement is incomplete")
    return rendered


def apply_ddl(
    query: str,
    *,
    project: str,
    location: str,
    sender: Callable[[urllib.request.Request], bytes] | None = None,
    access_token: str | None = None,
) -> None:
    request_fn = sender or _urlopen
    token = access_token or _access_token()
    project_path = urllib.parse.quote(project, safe="")
    response = _response_json(
        request_fn(
            urllib.request.Request(
                f"https://bigquery.googleapis.com/bigquery/v2/projects/{project_path}/queries",
                data=json.dumps(
                    {
                        "query": query,
                        "useLegacySql": False,
                        "location": location,
                        "timeoutMs": 200_000,
                    },
                    separators=(",", ":"),
                ).encode(),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                method="POST",
            )
        )
    )
    for _ in range(30):
        _raise_query_errors(response)
        if response.get("jobComplete") is True:
            return
        reference = response.get("jobReference")
        if not isinstance(reference, dict) or not isinstance(reference.get("jobId"), str):
            raise TypeError("BigQuery query response has no job reference")
        job_id = urllib.parse.quote(reference["jobId"], safe="")
        parameters = urllib.parse.urlencode(
            {"location": location, "timeoutMs": 200_000}
        )
        response = _response_json(
            request_fn(
                urllib.request.Request(
                    f"https://bigquery.googleapis.com/bigquery/v2/projects/{project_path}/queries/{job_id}?{parameters}",
                    headers={"Authorization": f"Bearer {token}"},
                    method="GET",
                )
            )
        )
    raise TimeoutError("BigQuery projection deployment did not complete")


def _access_token() -> str:
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(scopes=[_BIGQUERY_SCOPE])
    credentials.refresh(google.auth.transport.requests.Request())
    if not credentials.token:
        raise RuntimeError("Google credentials returned no access token")
    return credentials.token


def _urlopen(request: urllib.request.Request) -> bytes:
    with urllib.request.urlopen(request, timeout=210) as response:
        return response.read()


def _response_json(response: bytes) -> dict[str, object]:
    value = json.loads(response or b"{}")
    if not isinstance(value, dict):
        raise TypeError("BigQuery query response is not an object")
    return value


def _raise_query_errors(response: dict[str, object]) -> None:
    if response.get("errors"):
        raise RuntimeError(
            "BigQuery projection deployment failed: "
            + json.dumps(response["errors"], ensure_ascii=False, sort_keys=True)
        )


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    ddl = arguments.ddl.read_text(encoding="utf-8")
    rendered = render(ddl, project=arguments.project, dataset=arguments.dataset)
    replay_path = arguments.replay_sql
    if replay_path is None and arguments.ddl.resolve() == _DEFAULT_DDL.resolve():
        replay_path = _DEFAULT_REPLAY
    if replay_path is not None:
        replay = replay_path.read_text(encoding="utf-8")
        rendered = "\n".join((
            rendered,
            render(replay, project=arguments.project, dataset=arguments.dataset),
        ))
    if not arguments.apply:
        print(rendered, end="")
        return 0
    apply_ddl(
        rendered,
        project=arguments.project,
        location=arguments.location,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
