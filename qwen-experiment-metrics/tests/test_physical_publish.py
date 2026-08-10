from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from qwen_experiment_metrics.physical_publish import (
    TABLE_ALLOWLIST,
    PhysicalPublisherConfig,
    PhysicalPublishError,
    build_insert_body,
    deterministic_insert_id,
    physical_table_name,
    projection_json_row,
    publish_projection_batch,
)
from qwen_experiment_metrics.projections import (
    RELATION_REGISTRY,
    ProjectionBatch,
    ProjectionConflictError,
    ProjectionRow,
    deduplicate_rows,
)

RUN_ID = "r-0123456789abcdef0123456789abcdef"
SOURCE = "source/projection.json"
SOURCE_SHA = "a" * 64
CONFIG = PhysicalPublisherConfig("project", "dataset", "asia-northeast1")


def _request(ordinal: int) -> ProjectionRow:
    return ProjectionRow.from_mapping(
        "qwen_workload_requests_raw_v2",
        RUN_ID,
        SOURCE,
        SOURCE_SHA,
        {"workload_kind": "measured", "ordinal": ordinal, "request": {
            "source": {"kind": "generated", "generator_id": "fixture", "seed": 1, "generator_ordinal": ordinal},
            "prompt_payload": {"kind": "text", "text": "hi", "input_token_count": 1},
            "requested_output_token_count": 1,
            "sampling_params": [],
        }},
    )


def _network_row(*, bandwidth: str) -> ProjectionRow:
    return ProjectionRow.from_mapping(
        "qwen_mpi_network_results_v2",
        RUN_ID,
        SOURCE,
        SOURCE_SHA,
        {"benchmark": "PingPong", "process_count": 2, "message_bytes": 8,
         "iterations": 1, "time_min_us": None, "time_avg_us": None,
         "time_max_us": bandwidth, "bandwidth_mbytes_per_sec": bandwidth},
    )


def test_allowlist_and_closed_row_boundary() -> None:
    assert len(TABLE_ALLOWLIST) == 25
    assert physical_table_name("qwen_profiler_events_raw_v2") == "qwen_profiler_events_raw_v2"
    with pytest.raises(PhysicalPublishError):
        physical_table_name("qwen_metrics")
    row = _request(0)
    value = projection_json_row(row)
    assert "projection_relation" not in value
    assert value["schema_version"] == "workload-request-projection/v1"
    assert "source_member" not in value
    assert deterministic_insert_id(row) == deterministic_insert_id(row)


def test_insert_body_has_strict_flags_and_deterministic_id() -> None:
    row = _request(0)
    body = json.loads(build_insert_body((row,)))
    assert body["skipInvalidRows"] is False
    assert body["ignoreUnknownValues"] is False
    assert body["rows"] == [{"insertId": deterministic_insert_id(row), "json": projection_json_row(row)}]


def test_publisher_chunks_at_500_and_replays_stably() -> None:
    rows = tuple(_request(index) for index in range(501))
    requests: list[object] = []

    def sender(request: object) -> bytes:
        requests.append(request)
        return b'{"insertErrors": []}'

    report = publish_projection_batch(ProjectionBatch(rows), CONFIG, sender=sender)
    assert report.relation_rows == 501
    assert report.request_count == 2
    assert len(requests) == 2
    first_body = json.loads(requests[0].data)  # type: ignore[union-attr]
    second_body = json.loads(requests[1].data)  # type: ignore[union-attr]
    assert len(first_body["rows"]) == 500
    assert len(second_body["rows"]) == 1
    assert first_body["rows"][0]["insertId"] == deterministic_insert_id(rows[0])


def test_conflicts_fail_before_network_and_partial_errors_fail() -> None:
    conflicting = deduplicate_rows((_network_row(bandwidth="2.0"), _network_row(bandwidth="3.0")))
    called = False

    def sender(_: object) -> bytes:
        nonlocal called
        called = True
        return b"{}"

    with pytest.raises(ProjectionConflictError):
        publish_projection_batch(conflicting, CONFIG, sender=sender)
    assert not called
    with pytest.raises(PhysicalPublishError):
        publish_projection_batch(
            ProjectionBatch((_request(0),)), CONFIG,
            sender=lambda _: b'{"insertErrors": [{"index": 0}]}'
        )


def test_dry_run_renders_ddl_without_network() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/deploy-qwen-projections.py"),
         "--project", "project", "--dataset", "dataset"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.count("CREATE TABLE IF NOT EXISTS") == 25
    assert "__PROJECT__" not in result.stdout
    assert "__DATASET__" not in result.stdout


def test_apply_streams_ddl_to_bq_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/deploy-qwen-projections.py"
    specification = importlib.util.spec_from_file_location(
        "deploy_qwen_projections", script
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    ddl = tmp_path / "tables.sql"
    ddl.write_text(
        "CREATE TABLE `__PROJECT__.__DATASET__.fixture` (value STRING);\n",
        encoding="utf-8",
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv: list[str], **kwargs: object) -> None:
        calls.append((argv, kwargs))

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main(
        [
            "--project",
            "project",
            "--dataset",
            "dataset",
            "--ddl",
            str(ddl),
            "--apply",
        ]
    ) == 0
    assert calls == [
        (
            [
                "bq",
                "query",
                "--project_id=project",
                "--location=asia-northeast1",
                "--use_legacy_sql=false",
            ],
            {
                "input": b"CREATE TABLE `project.dataset.fixture` (value STRING);\n",
                "check": True,
            },
        )
    ]


def test_ddl_has_every_allowlisted_closed_field() -> None:
    ddl = (Path(__file__).resolve().parents[2] / "sql/qwen_projection_tables_v2.sql").read_text(encoding="utf-8")
    for relation, spec in RELATION_REGISTRY.items():
        match = re.search(
            rf"CREATE TABLE IF NOT EXISTS `__PROJECT__\.__DATASET__\.{re.escape(relation)}` \((.*?)\n\);",
            ddl,
            re.DOTALL,
        )
        assert match is not None, relation
        names = re.findall(
            r"(?:^|,)\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s+"
            r"(?:STRING|INT64|BIGNUMERIC|JSON)",
            match.group(1),
        )
        expected = ["schema_version", "run_id", *spec.fields]
        if spec.lineage:
            expected.extend(("source_member", "source_sha256"))
        assert set(names) == set(expected), relation
        assert len(names) == len(expected), relation
