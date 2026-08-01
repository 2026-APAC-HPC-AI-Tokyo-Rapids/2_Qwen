import shlex

from qwen_experiment_metrics.collectors import platform_metrics, run_start


def test_run_start_records_one_round_trippable_command() -> None:
    command = ["singularity", "exec", "--env", "NAME=value with space", "image.sif"]

    values = {metric.key: metric.value for metric in run_start(command)}

    assert shlex.split(values["run.command"]) == command
    assert not any(key.startswith("run.command.") for key in values)


def test_platform_records_model_name(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PATH", "/models/Qwen3.5-35B-A3B-FP8")

    values = {metric.key: metric.value for metric in platform_metrics()}

    assert values["model.name"] == "Qwen3.5-35B-A3B-FP8"
