from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_direct_raw_migration_recreates_only_three_string_columns() -> None:
    sql = (ROOT / "sql/migrate_direct_raw.sql").read_text(encoding="utf-8")
    assert "DROP TABLE IF EXISTS" in sql
    assert "DROP VIEW IF EXISTS" in sql
    assert "qwen_metrics`\n(run STRING NOT NULL, key STRING NOT NULL, value STRING NOT NULL)" in sql


def test_runtime_has_no_merge_projection_or_metrics_sidecar() -> None:
    package = ROOT / "qwen-experiment-metrics/src/qwen_experiment_metrics"
    for name in ("merge.py", "projections.py", "projection_router.py", "physical_publish.py"):
        assert not (package / name).exists()
    workflow = (ROOT / ".github/workflows/genkai.yml").read_text(encoding="utf-8")
    assert "metrics.jsonl" not in workflow
    assert '"${metrics_cli}" merge' not in workflow
    assert 'qwen-experiment-metrics" publish' not in workflow
