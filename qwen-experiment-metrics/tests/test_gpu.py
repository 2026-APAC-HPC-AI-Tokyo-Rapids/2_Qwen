from qwen_experiment_metrics.collectors import parse_static_gpu
from qwen_experiment_metrics.stage import Sample, _percentile, aggregate_gpu


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
