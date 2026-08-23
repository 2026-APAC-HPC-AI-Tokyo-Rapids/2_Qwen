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

    digest = "a" * 64
    image_size = 17
    sidecar_key = f"containers/{commit}/100-1/container-sidecar.json"
    sidecar = {"schema_version": "container-sidecar/v1", "container_artifact": {"size_bytes": image_size, "sha256": digest}, "source_commits": {"qwen": commit}}

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_executable(fake_bin / "aws", f'''#!/usr/bin/env bash
if [[ "$1 $2" == "s3api list-objects-v2" ]]; then printf '%s\\n' '{sidecar_key}'; exit 0; fi
if [[ "$1 $2" == "s3 cp" ]]; then printf '%s\\n' '{json.dumps(sidecar)}'; exit 0; fi
if [[ "$1 $2" == "s3api head-object" ]]; then printf '%s\\n' '{{"ContentLength":{image_size},"Metadata":{{"sha256":"{digest}"}}}}'; exit 0; fi
exit 1
''')
    output = tmp_path / "github-output"
    environment = os.environ | {"AWS_REGION": "ap-southeast-1", "CONTAINER_BUCKET": "fixture-bucket", "GITHUB_OUTPUT": str(output), "GITHUB_SHA": commit, "REQUIRE_BUILD": "false", "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    subprocess.run([ROOT / "scripts/select-qwen-container.sh"], cwd=repository, env=environment, check=True)
    assert output.read_text(encoding="utf-8") == f"action=reuse\nsource_commit={commit}\ncontainer_key=containers/{commit}/100-1/sglang.sqsh\nsidecar_key={sidecar_key}\n"


def test_selector_forces_build_without_touching_s3(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_executable(fake_bin / "aws", "#!/usr/bin/env bash\nexit 99\n")
    output = tmp_path / "github-output"
    environment = os.environ | {"AWS_REGION": "ap-southeast-1", "CONTAINER_BUCKET": "fixture-bucket", "GITHUB_OUTPUT": str(output), "GITHUB_SHA": "a" * 40, "REQUIRE_BUILD": "true", "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    subprocess.run([ROOT / "scripts/select-qwen-container.sh"], env=environment, check=True)
    assert output.read_text(encoding="utf-8") == "action=build\n"


def test_aws_cost_estimate_includes_ephemeral_storage_and_external_distribution(tmp_path: Path) -> None:
    output = tmp_path / "cost.json"
    environment = os.environ | {"BUILD_ELAPSED_SECONDS": "3600", "CONTAINER_SIZE_BYTES": "12718784512", "EXTERNAL_DOWNLOAD_COUNT": "3", "COST_OUTPUT": str(output)}
    subprocess.run([ROOT / "scripts/estimate-aws-build-cost.sh"], env=environment, check=True)
    estimate = json.loads(output.read_text(encoding="utf-8"))
    assert estimate["estimated_cost_usd"]["ec2"] == pytest.approx(0.504)
    assert estimate["estimated_cost_usd"]["ebs_gp3"] == pytest.approx(0.09)
    assert estimate["estimated_cost_usd"]["public_ipv4"] == pytest.approx(0.005)
    assert estimate["estimated_cost_usd"]["build_subtotal"] == pytest.approx(0.606635)
    assert estimate["estimated_cost_usd"]["external_data_transfer"] == pytest.approx(4.578762424)


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
    for relative in ("domain/fragment.jsonl", "logs/pbs-accounting.txt", "results/result.json", "source/model.json", "profile/raw/trace.json"):
        (runtime / relative).write_text("fixture\n", encoding="utf-8")
    (runtime / "run-success").touch()
    cli = runtime / "qwen-experiment-metrics/.venv/bin/qwen-experiment-metrics"
    make_executable(cli, "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" >\"${RUNTIME_DIR}/packager-capture.txt\"\n")
    subprocess.run([ROOT / "scripts/package-qwen-runtime.sh"], env=os.environ | {"RUNTIME_DIR": str(runtime)}, check=True)
    arguments = (runtime / "packager-capture.txt").read_text(encoding="utf-8").splitlines()
    assert arguments[0] == "package"
    assert "control=control/run-manifest.json" in arguments
    assert "domain=domain/fragment.jsonl" in arguments
    assert "logs=logs/pbs-accounting.txt" in arguments
    assert "logs=logs/pbs.log" not in arguments
    assert "results=results/result.json" in arguments
    assert "source=source/model.json" in arguments
    assert "success=run-success" in arguments
    assert ("profile/raw/trace.json" in arguments) is include_profile
