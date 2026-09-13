from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).parents[2]
PARSER = REPOSITORY / "scripts" / "parse-aiperf-result.py"


def metrics(
    fraction: float = 0.9,
    input_length: float = 4000,
    output_length: float = 200,
    input_statistics: dict[str, float] | None = None,
    output_statistics: dict[str, float] | None = None,
    osl_mismatch_count: float = 0,
) -> dict[str, object]:
    if output_statistics is None:
        output_statistics = {
            "min": output_length,
            "max": output_length,
            "count": 512,
            "sum": output_length * 512,
        }
    result = {
        "goodput": {"unit": "requests/sec", "avg": 4.25},
        "good_request_fraction": {"unit": "fraction", "avg": fraction},
        "request_count": {"unit": "requests", "avg": 512},
        "error_request_count": None,
        "input_sequence_length": {"unit": "tokens", "avg": input_length},
        "output_sequence_length": {
            "unit": "tokens",
            "avg": output_length,
            **output_statistics,
        },
        "osl_mismatch_count": {"unit": "requests", "avg": osl_mismatch_count},
    }
    if input_statistics is not None:
        result["input_sequence_length"].update(input_statistics)
    return result


@pytest.fixture(params=["repository", "submission"])
def parser_path(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    if request.param == "repository":
        return PARSER
    script = (REPOSITORY / "submission/run/submit.sbatch").read_text()
    embedded = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0] + "\n"
    assert embedded == PARSER.read_text()
    destination = tmp_path / "embedded-parser.py"
    destination.write_text(embedded)
    return destination


@pytest.fixture
def official_artifacts(tmp_path: Path) -> list[str]:
    # Exercise the actual Bash array construction and printf %q serialization.
    script = (REPOSITORY / "scripts/run-aiperf.sh").read_text()
    recording = script.split("printf 'aiperf_version=", 1)[0]
    environment = {**os.environ, "AIPERF_MODE": "official", "CONCURRENCY": "16", "HOST": "127.0.0.1:8000", "RESULTS_DIR": str(tmp_path)}
    subprocess.run(["bash", "-c", recording], env=environment, check=True)
    provenance = tmp_path / "aiperf-provenance.txt"
    provenance.write_text("aiperf_version=0.12.0\nmode=official\nconcurrency=16\ngateway=http://127.0.0.1:8000/v1/chat/completions\ntokenizer_revision=ac9c66cc9b46af7306746a9250f23d47083d689e\n")
    return ["--command", str(tmp_path / "aiperf-command.txt"), "--provenance", str(provenance)]


def parse(tmp_path: Path, parser_path: Path, artifacts: list[str], values: dict[str, object]) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    source = tmp_path / "profile_export_aiperf.json"
    output = tmp_path / "score.json"
    source.write_text(json.dumps(values))
    process = subprocess.run([sys.executable, parser_path, source, output, "--concurrency", "16", "--require-eligible", *artifacts], check=False, text=True, capture_output=True)
    return process, json.loads(output.read_text())


@pytest.mark.parametrize("input_length", [4000, 3986])
def test_parser_emits_official_score(tmp_path: Path, parser_path: Path, official_artifacts: list[str], input_length: int) -> None:
    process, result = parse(tmp_path, parser_path, official_artifacts, metrics(input_length=input_length))
    assert process.returncode == 0, process.stderr
    assert result["eligible"] is True
    assert result["score_output_tokens_per_second"] == 850
    assert result["error_request_count"] == 0
    assert result["observed_input_sequence_length_avg"] == input_length
    assert result["official_validation"]["valid"] is True
    assert result["official_validation"]["configured_input_sequence_length"] == 4000
    assert result["official_validation"]["client_input_token_adjustment"] == 4000 - input_length


def test_parser_rejects_ineligible_official_run(tmp_path: Path, parser_path: Path, official_artifacts: list[str]) -> None:
    process, result = parse(tmp_path, parser_path, official_artifacts, metrics(0.899, input_length=3986))
    assert process.returncode != 0
    assert result["score_output_tokens_per_second"] == 0
    assert result["official_validation"]["valid"] is False
    assert "below 0.90" in process.stderr


def test_parser_accepts_measured_cache_bust_distribution(tmp_path: Path, parser_path: Path, official_artifacts: list[str]) -> None:
    observed = 3986.001801801802
    statistics = {"min": 3986, "max": 3987, "count": 555, "sum": 2212231}
    values = metrics(
        input_length=observed,
        input_statistics=statistics,
        output_statistics={"min": 200, "max": 200, "count": 555, "sum": 111000},
    )
    values["request_count"] = {"unit": "requests", "avg": 555}

    process, result = parse(tmp_path, parser_path, official_artifacts, values)

    assert process.returncode == 0, process.stderr
    assert result["observed_input_sequence_length_min"] == 3986
    assert result["observed_input_sequence_length_max"] == 3987
    assert result["official_validation"]["valid"] is True


def test_parser_accepts_measured_output_length_distribution(tmp_path: Path, parser_path: Path, official_artifacts: list[str]) -> None:
    observed = 200.00179211469535
    statistics = {"min": 200, "max": 201, "count": 558, "sum": 111601}
    values = metrics(output_length=observed, output_statistics=statistics)
    values["request_count"] = {"unit": "requests", "avg": 558}

    process, result = parse(tmp_path, parser_path, official_artifacts, values)

    assert process.returncode == 0, process.stderr
    assert result["observed_output_sequence_length_min"] == 200
    assert result["observed_output_sequence_length_max"] == 201
    assert result["official_validation"]["valid"] is True


@pytest.mark.parametrize("minimum,maximum", [(3983, 3987), (3986, 3989)])
def test_parser_rejects_cache_bust_distribution_outside_pinned_range(tmp_path: Path, parser_path: Path, official_artifacts: list[str], minimum: int, maximum: int) -> None:
    statistics = {"min": minimum, "max": maximum, "count": 512, "sum": 2040832}
    process, result = parse(tmp_path, parser_path, official_artifacts, metrics(input_length=3986.0 + 1 / 512, input_statistics=statistics))

    assert process.returncode != 0
    assert result["official_validation"]["valid"] is False
    assert "cache-bust distribution" in process.stderr


@pytest.mark.parametrize("name,value", [("input_length", 3999), ("input_length", 3985), ("input_length", 3986.000001)])
def test_parser_rejects_wrong_official_token_lengths(tmp_path: Path, parser_path: Path, official_artifacts: list[str], name: str, value: float) -> None:
    process, result = parse(tmp_path, parser_path, official_artifacts, metrics(**{name: value}))
    assert process.returncode != 0
    assert result["official_validation"]["valid"] is False
    assert "average" in process.stderr


@pytest.mark.parametrize(
    "statistics,mismatch_count",
    [
        ({"min": 189, "max": 200, "count": 512, "sum": 102389}, 1),
        ({"min": 200, "max": 211, "count": 512, "sum": 102411}, 1),
        ({"min": 199, "max": 201, "count": 511, "sum": 102200}, 0),
        ({"min": 199, "max": 201, "count": 512, "sum": 102399}, 0),
    ],
)
def test_parser_rejects_invalid_output_length_distribution(
    tmp_path: Path,
    parser_path: Path,
    official_artifacts: list[str],
    statistics: dict[str, float],
    mismatch_count: float,
) -> None:
    process, result = parse(
        tmp_path,
        parser_path,
        official_artifacts,
        metrics(output_length=200, output_statistics=statistics, osl_mismatch_count=mismatch_count),
    )

    assert process.returncode != 0
    assert result["official_validation"]["valid"] is False
    assert "output sequence length" in process.stderr


def test_parser_rejects_aiperf_output_length_mismatch(tmp_path: Path, parser_path: Path, official_artifacts: list[str]) -> None:
    process, result = parse(tmp_path, parser_path, official_artifacts, metrics(osl_mismatch_count=1))

    assert process.returncode != 0
    assert result["official_validation"]["valid"] is False
    assert "reported output sequence length mismatches" in process.stderr


@pytest.mark.parametrize("option,value", [("--synthetic-input-tokens-mean", "3986"), ("--warmup-duration", "10"), ("--random-seed", "2027"), ("--cache-bust", "none"), ("--workers-max", "8"), ("--goodput", "time_to_first_token:9000 inter_token_latency:20"), ("--url", "http://127.0.0.2:8000")])
def test_parser_rejects_changed_official_command(tmp_path: Path, parser_path: Path, official_artifacts: list[str], option: str, value: str) -> None:
    command_file = Path(official_artifacts[1])
    command = shlex.split(command_file.read_text())
    command[command.index(option) + 1] = value
    command_file.write_text(shlex.join(command))
    process, result = parse(tmp_path, parser_path, official_artifacts, metrics(input_length=3986))
    assert process.returncode != 0
    assert result["official_validation"]["valid"] is False
    assert "command does not match" in process.stderr


@pytest.mark.parametrize("extra", ["--apply-chat-template", "--request-rate 1", "--synthetic-input-tokens-mean 3986"])
def test_parser_rejects_extra_official_arguments(tmp_path: Path, parser_path: Path, official_artifacts: list[str], extra: str) -> None:
    command_file = Path(official_artifacts[1])
    command_file.write_text(command_file.read_text() + extra)
    process, _ = parse(tmp_path, parser_path, official_artifacts, metrics(input_length=3986))
    assert process.returncode != 0
    assert "command does not match" in process.stderr


@pytest.mark.parametrize("old,new", [("0.12.0", "0.13.0"), ("official", "exploratory"), ("concurrency=16", "concurrency=8"), ("ac9c66cc9b46af7306746a9250f23d47083d689e", "main")])
def test_parser_rejects_wrong_provenance(tmp_path: Path, parser_path: Path, official_artifacts: list[str], old: str, new: str) -> None:
    provenance_file = Path(official_artifacts[3])
    provenance_file.write_text(provenance_file.read_text().replace(old, new))
    process, _ = parse(tmp_path, parser_path, official_artifacts, metrics(input_length=3986))
    assert process.returncode != 0
    assert "provenance does not match" in process.stderr


def test_parser_rejects_missing_official_evidence(tmp_path: Path, parser_path: Path) -> None:
    process, result = parse(tmp_path, parser_path, [], metrics(input_length=3986))
    assert process.returncode != 0
    assert result["official_validation"]["valid"] is False
    assert "requires --command and --provenance" in process.stderr


def test_submission_records_same_official_command(tmp_path: Path, official_artifacts: list[str]) -> None:
    script = (REPOSITORY / "submission/run/submit.sbatch").read_text()
    command_line = next(line for line in script.splitlines() if line.startswith("  command=("))
    process = subprocess.run(["bash", "-c", command_line + "\nprintf '%q ' \"${command[@]}\""], env={**os.environ, "CONCURRENCY": "16"}, check=True, capture_output=True, text=True)
    assert shlex.split(process.stdout) == shlex.split(Path(official_artifacts[1]).read_text())


def test_submission_records_runtime_provenance(tmp_path: Path, official_artifacts: list[str]) -> None:
    script = (REPOSITORY / "submission/run/submit.sbatch").read_text()
    line = next(line for line in script.splitlines() if "/bin/bash -ec" in line)
    arguments = shlex.split(line)
    program = arguments[arguments.index("-ec") + 1]
    interpreter = tmp_path / "version-python"
    interpreter.write_text(f'#!{sys.executable}\nimport sys\nassert sys.argv[1] == "-c"\ncompile(sys.argv[2], "version-probe", "exec")\nprint("0.12.0")\n')
    interpreter.chmod(0o755)
    revision = tmp_path / "tokenizer-revision"
    revision.write_text("ac9c66cc9b46af7306746a9250f23d47083d689e")
    program = program.replace("cd /results", f"cd {shlex.quote(str(tmp_path))}").replace("/opt/sglang/venv/bin/python", shlex.quote(str(interpreter))).replace("/hf-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/refs/main", shlex.quote(str(revision)))
    expected = Path(official_artifacts[3]).read_text()
    process = subprocess.run(["bash", "-ec", program, "_", "16", sys.executable, "-c", "print('benchmark invoked')"], check=True, capture_output=True, text=True)
    assert process.stdout.strip() == "benchmark invoked"
    assert Path(official_artifacts[3]).read_text() == expected


def test_official_command_contains_locked_workload() -> None:
    script = (REPOSITORY / "scripts" / "run-aiperf.sh").read_text()
    required = (
        "--endpoint /v1/chat/completions",
        "--tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507",
        "--warmup-duration 60",
        "--benchmark-duration 120",
        "--benchmark-grace-period 30",
        "--synthetic-input-tokens-mean 4000",
        "--output-tokens-mean 200",
        "--num-dataset-entries 12800",
        "--random-seed 2026",
        "--cache-bust first_turn_prefix",
        '--goodput "time_to_first_token:3000 inter_token_latency:20"',
    )
    assert all(argument in script for argument in required)
    assert "--request-rate" not in script


def test_provision_maps_main_to_pinned_tokenizer_revision() -> None:
    script = (REPOSITORY / "scripts" / "provision-qwen-pd.sh").read_text()
    assert 'hf download "${tokenizer_repository}" config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json --revision "${tokenizer_revision}"' in script
    assert 'test -s "${tokenizer_snapshot}/${tokenizer_file}"' in script
    assert 'printf \'%s\' "${tokenizer_revision}" >"${tokenizer_cache_repository}/refs/main"' in script
