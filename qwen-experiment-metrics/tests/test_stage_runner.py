from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

from qwen_experiment_metrics.canonical import load_json_strict
from qwen_experiment_metrics.recording import Recorder
from qwen_experiment_metrics.stage_runner import StageRunner


def _fragment_values(path: Path) -> dict[str, object]:
    rows = [load_json_strict(line) for line in path.read_bytes().splitlines()]
    return {row["key"]: load_json_strict(row["value"]) for row in rows}


def test_stage_runner_records_exact_launch_exit_and_timing(
    manifest_path: Path, runtime: Path, tmp_path: Path
) -> None:
    argv = [sys.executable, "-c", "import sys; assert sys.argv[1] == 'a b'", "a b"]
    with Recorder.open(manifest_path, None, "stage:benchmark", runtime) as recorder:
        termination = StageRunner(recorder).run("benchmark", argv, cwd=tmp_path)
        completed = recorder.completed_path
    assert termination.kind == "exit" and termination.code == 0
    values = _fragment_values(completed)
    assert values["run.stage.benchmark.launch"] == {
        "argv": argv,
        "cwd": str(tmp_path.resolve()),
        "stderr_mode": "inherit",
    }
    assert values["run.stage.benchmark.termination"] == {"kind": "exit", "code": 0}
    assert float(values["perf.stage.benchmark.wall_time_sec"]) >= 0


def test_stage_runner_distinguishes_signal_and_spawn_error(
    manifest_path: Path, runtime: Path, tmp_path: Path
) -> None:
    with Recorder.open(manifest_path, None, "stage:signalled", runtime) as recorder:
        signal_result = StageRunner(recorder).run(
            "signalled",
            [
                sys.executable,
                "-c",
                f"import os; os.kill(os.getpid(), {signal.SIGTERM})",
            ],
            cwd=tmp_path,
        )
    assert signal_result.kind == "signal" and signal_result.signal == signal.SIGTERM

    second_runtime = runtime.parent / "second"
    second_manifest = second_runtime / "control" / "run-manifest.json"
    second_manifest.parent.mkdir(parents=True)
    second_manifest.write_bytes(manifest_path.read_bytes())
    with Recorder.open(
        second_manifest, None, "stage:missing", second_runtime
    ) as recorder:
        missing = StageRunner(recorder).run(
            "missing", [os.fspath(tmp_path / "does-not-exist")], cwd=tmp_path
        )
        completed = recorder.completed_path
    assert missing.kind == "spawn_error" and missing.shell_returncode() == 127
    assert _fragment_values(completed)["run.stage.missing.termination"] == {
        "kind": "spawn_error"
    }


def test_stage_runner_owns_combined_stdout_file(
    manifest_path: Path, runtime: Path, tmp_path: Path
) -> None:
    output = tmp_path / "combined.log"
    argv = [
        sys.executable,
        "-c",
        "import os; os.write(1, b'out\\n'); os.write(2, b'err\\n')",
    ]
    with Recorder.open(manifest_path, None, "stage:benchmark", runtime) as recorder:
        termination = StageRunner(recorder).run(
            "benchmark",
            argv,
            cwd=tmp_path,
            stderr_mode="stdout",
            stdout_path=output,
        )
        completed = recorder.completed_path

    assert termination.kind == "exit" and termination.code == 0
    assert output.read_bytes() == b"out\nerr\n"
    assert _fragment_values(completed)["run.stage.benchmark.launch"] == {
        "argv": argv,
        "cwd": str(tmp_path.resolve()),
        "stderr_mode": "stdout",
    }
