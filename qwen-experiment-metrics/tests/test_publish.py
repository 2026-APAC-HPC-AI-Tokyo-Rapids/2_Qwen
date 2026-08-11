from __future__ import annotations

import json

import pytest

from qwen_experiment_metrics.publish import BigQueryConfig, _access_token, send_raw

CONFIG = BigQueryConfig("project", "dataset", "table", "location")


def test_send_raw_chunks_without_insert_ids_and_allows_duplicates() -> None:
    requests = []
    rows = [{"run": "r", "key": "same", "value": "value"}] * 1001
    assert send_raw(
        "r",
        rows,
        config=CONFIG,
        request=lambda request: requests.append(request) or b"{}",
    )
    bodies = [json.loads(request.data) for request in requests]
    assert [len(body["rows"]) for body in bodies] == [500, 500, 1]
    assert set(bodies[0]["rows"][0]) == {"json"}
    assert bodies[0]["rows"][0] == bodies[0]["rows"][1]


def test_send_raw_validates_exact_three_string_fields() -> None:
    invalid = [
        {"run": "", "key": "k", "value": "v"},
        {"run": "r", "key": "", "value": "v"},
        {"run": "r", "key": "k", "value": 1},
        {"run": "r", "key": "k", "value": "v", "extra": "x"},
    ]
    for row in invalid:
        with pytest.warns(RuntimeWarning):
            assert not send_raw("r", [row], config=CONFIG, request=lambda _: b"{}")


def test_send_raw_warns_and_does_not_retry() -> None:
    calls = 0

    def reject(_request: object) -> bytes:
        nonlocal calls
        calls += 1
        return b'{"insertErrors":[{"index":0}]}'

    with pytest.warns(RuntimeWarning):
        assert not send_raw(
            "r", [{"run": "r", "key": "k", "value": "v"}], config=CONFIG, request=reject
        )
    assert calls == 1


def test_credentials_file_must_be_mode_0600(tmp_path) -> None:
    path = tmp_path / "credential.json"
    path.write_text("{}")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="mode-0600"):
        _access_token(path)
