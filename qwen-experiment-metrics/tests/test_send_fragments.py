from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_experiment_metrics.publish import RawRow, send_fragments


def test_send_fragments_reads_only_completed_rows(tmp_path: Path) -> None:
    directory = tmp_path / "domain/fragments"
    directory.mkdir(parents=True)
    rows = [
        {"run": "run-1", "key": "a", "value": "1"},
        {"run": "run-1", "key": "b", "value": "2"},
    ]
    (directory / "producer--YQ.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    observed: list[RawRow] = []

    def sender(run_id: str, values: object) -> bool:
        assert run_id == "run-1"
        observed.extend(values)  # type: ignore[arg-type]
        return True

    assert send_fragments("run-1", tmp_path, sender=sender)
    assert [(row.key, row.value) for row in observed] == [("a", "1"), ("b", "2")]


def test_send_fragments_rejects_partial_or_foreign_rows(tmp_path: Path) -> None:
    directory = tmp_path / "domain/fragments"
    directory.mkdir(parents=True)
    (directory / "producer--YQ.jsonl.tmp").write_text("")
    with pytest.raises(ValueError, match="partial"):
        send_fragments("run-1", tmp_path, sender=lambda *_: True)
    (directory / "producer--YQ.jsonl.tmp").unlink()
    (directory / "producer--YQ.jsonl").write_text(
        '{"run":"run-2","key":"a","value":"1"}\n'
    )
    with pytest.raises(ValueError, match="another run"):
        send_fragments("run-1", tmp_path, sender=lambda *_: True)
