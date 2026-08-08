import enum
from pathlib import Path

import pytest

from qwen_experiment_metrics.collectors import log_artifact
from qwen_experiment_metrics.core import Metric, _rows, flatten, load_config, scalar


class Choice(enum.Enum):
    VALUE = "selected"


def test_scalar_and_flatten() -> None:
    assert scalar(True) == "true"
    assert scalar(float("nan")) == ""
    assert flatten("root", {"nested": [1, None, Choice.VALUE]}) == [
        Metric("root.nested", '[1,null,"selected"]'),
    ]
    assert flatten("root", {}) == [Metric("root", "")]
    assert flatten("root", []) == [Metric("root", "[]")]
    assert flatten("root", [{"z": 1, "a": 2}]) == [Metric("root", '[{"a":2,"z":1}]')]


def test_cuda_graph_batch_sizes_are_one_metric_per_list() -> None:
    metrics = flatten(
        "sglang.server_arg",
        {
            "cuda_graph_config": {
                "prefill": {"bs": [4, 8, 12]},
                "decode": {"bs": [1, 2, 4]},
            }
        },
    )

    assert metrics == [
        Metric("sglang.server_arg.cuda_graph_config.prefill.bs", "[4,8,12]"),
        Metric("sglang.server_arg.cuda_graph_config.decode.bs", "[1,2,4]"),
    ]
    assert not any(metric.key.rsplit(".", 1)[-1].isdigit() for metric in metrics)


def test_rows_have_only_three_columns() -> None:
    rows = _rows("qwen-run", [Metric("a", "1")])
    assert set(rows[0]["json"]) == {"run", "key", "value"}
    assert len(rows[0]["insertId"]) == 64


def test_metric_rejects_composite_values() -> None:
    with pytest.raises(TypeError, match="scalar string"):
        Metric("invalid", [1, 2])  # type: ignore[arg-type]


def test_config() -> None:
    path = Path(__file__).parents[1] / "pyproject.toml"
    config = load_config(path)
    assert config.dataset == "qwen_experiment_metrics"
    assert config.table == "qwen_metrics"


def test_log_artifact(tmp_path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "123.log").write_text("complete\n")
    values = {metric.key: metric.value for metric in log_artifact(tmp_path, "123")}
    assert values["artifact.pjm_log.path"] == "logs/123.log"
    assert values["artifact.pjm_log.size_bytes"] == "9"
    assert not any(key.endswith(".sha256") for key in values)
