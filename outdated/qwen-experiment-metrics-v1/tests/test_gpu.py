import json

from qwen_experiment_metrics.collectors import parse_static_gpu
from qwen_experiment_metrics.core import Metric
from qwen_experiment_metrics.stage import (
    Sample,
    _node_gpu_metrics,
    _percentile,
    aggregate_gpu,
    load_node_gpu_metrics,
)


def test_tg1_static_gpu_format() -> None:
    metrics = parse_static_gpu(
        "0, NVIDIA RTX A6000, GPU-test, 00000000:31:00.0, 8.6, 49140, 300.00, [N/A], 595.71.05\n"
    )
    values = {metric.key: metric.value for metric in metrics}
    assert values["hardware.gpu.count"] == "1"
    assert values["hardware.gpu.0.uuid"] == "GPU-test"
    assert values["hardware.gpu.0.memory.total_bytes"] == str(49140 * 1024 * 1024)
    assert values["hardware.gpu.0.mig.mode.current"] == ""


def test_percentile_linear_interpolation() -> None:
    assert _percentile([0, 10], 0.95) == 9.5


def test_gpu_aggregate_and_energy() -> None:
    samples = {
        "GPU-test": [
            Sample(
                1.0,
                {
                    "utilization.gpu": "10",
                    "memory.used": "1",
                    "power.draw": "100",
                    "pstate": "P0",
                },
            ),
            Sample(
                3.0,
                {
                    "utilization.gpu": "30",
                    "memory.used": "2",
                    "power.draw": "200",
                    "pstate": "P0",
                },
            ),
        ]
    }
    metrics = {metric.key: metric.value for metric in aggregate_gpu(samples, 1.0)}
    prefix = "resource.stage.benchmark.gpu.0"
    assert metrics[f"{prefix}.utilization.gpu_percent.mean"] == "20.0"
    assert metrics[f"{prefix}.memory.used_bytes.max"] == str(2 * 1024 * 1024.0)
    assert metrics[f"{prefix}.energy_joule"] == "300.0"
    assert metrics[f"{prefix}.pstate.p0.sample_count"] == "2"


def test_node_gpu_metrics_are_scalar_and_aggregated(tmp_path) -> None:
    for node_rank in (0, 1):
        metrics = _node_gpu_metrics(
            node_rank,
            f"b000{node_rank}",
            [Metric("hardware.gpu.count", "4")],
            [
                Metric(
                    "resource.stage.benchmark.gpu.0.utilization.gpu_percent.mean",
                    "75.0",
                )
            ],
        )
        path = tmp_path / f"node-rank-{node_rank}-gpu-metrics.json"
        path.write_text(
            json.dumps(
                [{"key": metric.key, "value": metric.value} for metric in metrics]
            )
        )

    values = {
        metric.key: metric.value for metric in load_node_gpu_metrics(tmp_path)
    }

    assert values["hardware.node_count.observed"] == "2"
    assert values["hardware.gpu.count.observed"] == "8"
    assert values["hardware.gpu.count"] == "8"
    assert values["hardware.node.0.gpu.count"] == "4"
    assert (
        values[
            "resource.stage.benchmark.node.1.gpu.0.utilization.gpu_percent.mean"
        ]
        == "75.0"
    )
    assert not list(tmp_path.glob("node-rank-*-gpu-metrics.json"))


def test_total_gpu_count_is_empty_when_a_node_is_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PJM_NODE", "2")
    path = tmp_path / "node-rank-0-gpu-metrics.json"
    path.write_text(
        json.dumps(
            [
                {"key": "hardware.node.0.hostname", "value": "b0000"},
                {"key": "hardware.node.0.gpu.count", "value": "4"},
            ]
        )
    )

    values = {
        metric.key: metric.value for metric in load_node_gpu_metrics(tmp_path)
    }

    assert values["hardware.node_count.observed"] == "1"
    assert values["hardware.gpu.count.observed"] == "4"
    assert values["hardware.gpu.count"] == ""
