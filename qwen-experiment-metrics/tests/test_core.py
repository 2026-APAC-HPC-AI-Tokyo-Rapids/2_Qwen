import enum
from pathlib import Path

from qwen_experiment_metrics.core import Metric, _rows, flatten, load_config, scalar


class Choice(enum.Enum):
    VALUE = "selected"


def test_scalar_and_flatten() -> None:
    assert scalar(True) == "true"
    assert scalar(float("nan")) == ""
    assert flatten("root", {"nested": [1, None, Choice.VALUE]}) == [
        Metric("root.nested.0", "1"),
        Metric("root.nested.1", ""),
        Metric("root.nested.2", "selected"),
    ]
    assert flatten("root", {}) == [Metric("root", "")]


def test_rows_have_only_three_columns() -> None:
    rows = _rows("qwen-run", [Metric("a", "1")])
    assert set(rows[0]["json"]) == {"run", "key", "value"}
    assert len(rows[0]["insertId"]) == 64


def test_config() -> None:
    path = Path(__file__).parents[1] / "pyproject.toml"
    config = load_config(path)
    assert config.dataset == "qwen_experiment_metrics"
    assert config.table == "qwen_metrics"
