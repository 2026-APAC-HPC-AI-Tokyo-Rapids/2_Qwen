#!/usr/bin/env python3
"""Render or apply the v2 physical projection table DDL.

Rendering is the default and is network/credential free.  ``--apply`` is an
explicit opt-in that invokes the local ``bq`` CLI with Standard SQL.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]+$")
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DDL = _ROOT / "sql/qwen_projection_tables_v2.sql"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deploy-qwen-projections")
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--location", default="asia-northeast1")
    parser.add_argument("--ddl", type=Path, default=_DEFAULT_DDL)
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


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    ddl = arguments.ddl.read_text(encoding="utf-8")
    rendered = render(ddl, project=arguments.project, dataset=arguments.dataset)
    if not arguments.apply:
        print(rendered, end="")
        return 0
    subprocess.run(
        [
            "bq",
            "query",
            f"--project_id={arguments.project}",
            f"--location={arguments.location}",
            "--use_legacy_sql=false",
        ],
        input=rendered.encode("utf-8"),
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
