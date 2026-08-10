from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SQL = ROOT / "sql"


def _read(name: str) -> str:
    return (SQL / name).read_text(encoding="utf-8").lower()


def test_qwen_sql_contract_has_conflict_safe_layers() -> None:
    facts = _read("qwen_run_facts_v2.sql")
    conflicts = _read("qwen_run_fact_conflicts_v2.sql")
    consumer = _read("qwen_run_consumer_v2.sql")

    assert "create or replace view" in facts
    assert "qwen_run_facts_v2" in facts
    assert "select distinct" in facts
    assert "count(*)" in facts
    assert "distinct_value_count = 1" in facts
    assert "qwen_metrics" in facts
    assert "as `run`" in facts
    assert "as `key`" in facts
    assert "as `value`" in facts

    assert "qwen_run_fact_conflicts_v2" in conflicts
    assert "select distinct" in conflicts
    assert "having count(*) > 1" in conflicts
    assert "qwen_metrics" in conflicts
    assert "as `run`" in conflicts
    assert "as `key`" in conflicts
    assert "as `value`" in conflicts

    assert "qwen_run_consumer_v2" in consumer
    assert "qwen_run_facts_v2" in consumer
    assert "qwen_run_fact_conflicts_v2" in consumer
    assert "collection_complete" in consumer
    assert "has_conflicts" in consumer
    assert "result_returned_count" in consumer
    assert "result_output_tokens_per_sec" in consumer
    assert "any_value(" not in consumer


def test_qwen_sql_is_read_only_ddl() -> None:
    for path in SQL.glob("qwen_*_v2.sql"):
        text = path.read_text(encoding="utf-8").lower()
        assert "delete " not in text
        assert "insert " not in text
        assert "update " not in text
        assert "drop table" not in text
