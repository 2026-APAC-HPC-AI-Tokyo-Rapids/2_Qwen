from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def make_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_selector_reuses_valid_first_parent_artifact(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repository, check=True)
    (repository / "tracked.txt").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()

    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    archive = archive_dir / "100.sqsh"
    archive.write_bytes(b"container fixture")
    digest = subprocess.run(["sha256sum", str(archive)], check=True, capture_output=True, text=True).stdout.split()[0]
    sidecar = {"schema_version": "container-sidecar/v1", "container_artifact": {"size_bytes": archive.stat().st_size, "sha256": digest}}
    (archive_dir / "100.container-sidecar.json").write_text(json.dumps(sidecar), encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_executable(fake_bin / "gh", "#!/usr/bin/env bash\nprintf '100\\n'\n")
    output = tmp_path / "github-output"
    environment = os.environ | {"CONTAINER_ARCHIVE_DIR": str(archive_dir), "GITHUB_OUTPUT": str(output), "GITHUB_RUN_ID": "999", "GITHUB_SHA": commit, "REQUIRE_BUILD": "false", "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    subprocess.run([ROOT / "scripts/select-qwen-container.sh"], cwd=repository, env=environment, check=True)
    assert output.read_text(encoding="utf-8") == "action=reuse\nsource_run_id=100\n"


def test_publisher_removes_credential_file(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    cli = runtime / "qwen-experiment-metrics/.venv/bin/qwen-experiment-metrics"
    cli.parent.mkdir(parents=True)
    (runtime / "control").mkdir()
    (runtime / "qwen-experiment-metrics/pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (runtime / "control/run-manifest.json").write_text('{"run_id":"fixture-run"}\n', encoding="utf-8")
    make_executable(cli, "#!/usr/bin/env bash\ntest -f \"${EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_FILE}\"\ntest \"$(stat -c '%a' \"${EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_FILE}\")\" = 600\nprintf '%s\\n' \"${EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_FILE}\" \"${EXPERIMENT_METRICS_BIGQUERY_CONFIG}\" \"$*\" >\"${RUNTIME_DIR}/publisher-capture.txt\"\n")
    environment = os.environ | {"METRICS_CREDENTIAL_JSON": "{}", "RUNTIME_DIR": str(runtime)}
    subprocess.run([ROOT / "scripts/send-qwen-fragments.sh"], env=environment, check=True)
    credential_path, config_path, arguments = (runtime / "publisher-capture.txt").read_text(encoding="utf-8").splitlines()
    assert not Path(credential_path).exists()
    assert config_path == str(runtime / "qwen-experiment-metrics/pyproject.toml")
    assert arguments == f"send-fragments --run fixture-run --runtime-dir {runtime}"


@pytest.mark.parametrize(("run_purpose", "include_profile"), [("bench", False), ("profile", True)])
def test_packager_derives_members_and_profile_from_manifest(tmp_path: Path, run_purpose: str, include_profile: bool) -> None:
    runtime = tmp_path / "runtime"
    for relative in ("control", "domain", "logs", "results", "source", "profile/raw", "qwen-experiment-metrics/.venv/bin"):
        (runtime / relative).mkdir(parents=True, exist_ok=True)
    (runtime / "control/run-manifest.json").write_text(json.dumps({"run_purpose": run_purpose}), encoding="utf-8")
    for relative in ("domain/fragment.jsonl", "logs/benchmark.log", "results/result.json", "source/model.json", "profile/raw/trace.json"):
        (runtime / relative).write_text("fixture\n", encoding="utf-8")
    (runtime / "run-success").touch()
    cli = runtime / "qwen-experiment-metrics/.venv/bin/qwen-experiment-metrics"
    make_executable(cli, "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" >\"${RUNTIME_DIR}/packager-capture.txt\"\n")
    subprocess.run([ROOT / "scripts/package-qwen-runtime.sh"], env=os.environ | {"RUNTIME_DIR": str(runtime)}, check=True)
    arguments = (runtime / "packager-capture.txt").read_text(encoding="utf-8").splitlines()
    assert arguments[0] == "package"
    assert "control=control/run-manifest.json" in arguments
    assert "domain=domain/fragment.jsonl" in arguments
    assert "logs=logs/benchmark.log" in arguments
    assert "results=results/result.json" in arguments
    assert "source=source/model.json" in arguments
    assert "success=run-success" in arguments
    assert ("profile/raw/trace.json" in arguments) is include_profile
