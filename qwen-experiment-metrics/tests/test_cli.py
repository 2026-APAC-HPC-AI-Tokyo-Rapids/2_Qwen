from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conftest import EXECUTION_SETTINGS

from qwen_experiment_metrics.canonical import load_json_strict
from qwen_experiment_metrics.cli import main


def test_cli_composes_init_stage_and_merge(tmp_path: Path, capsys: object) -> None:
    runtime = tmp_path / "runtime"
    run_id = "r-" + "1" * 32
    settings = tmp_path / "execution-settings.json"
    settings.write_text(json.dumps(EXECUTION_SETTINGS), encoding="utf-8")
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    assert (
        main(
            [
                "init-run",
                "--runtime-dir",
                str(runtime),
                "--purpose",
                "bench",
                "--object-hash-algorithm",
                "sha1",
                "--base-role",
                "score_reference",
                "--base-commit",
                commit,
                "--head-commit",
                commit,
                "--control-plane-sha256",
                "c" * 64,
                "--instrumentation-bundle-sha256",
                "d" * 64,
                "--execution-settings",
                str(settings),
                "--repository",
                str(repository),
                "--run-id",
                run_id,
            ]
        )
        == 0
    )
    manifest = runtime / "control/run-manifest.json"
    assert (
        main(
            [
                "run-stage",
                "--manifest",
                str(manifest),
                "--runtime-dir",
                str(runtime),
                "--name",
                "smoke",
                "--cwd",
                str(tmp_path),
                "--",
                sys.executable,
                "-c",
                "pass",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "merge",
                "--manifest",
                str(manifest),
                "--runtime-dir",
                str(runtime),
            ]
        )
        == 1
    )
    rows = [
        load_json_strict(line)
        for line in (runtime / "metrics.jsonl").read_bytes().splitlines()
    ]
    assert {row["key"] for row in rows} >= {
        "run.stage.smoke.launch",
        "run.stage.smoke.termination",
        "perf.stage.smoke.wall_time_sec",
        "collection_complete",
    }
