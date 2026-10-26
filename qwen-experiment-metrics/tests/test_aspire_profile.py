"""Run the profile orchestration on CPU with external processes stubbed."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("client_status,capture_status", [(0, 0), (7, 0), (0, 9)])
def test_profile_preserves_normal_score_and_drains_capture(
    tmp_path: Path, client_status: int, capture_status: int
) -> None:
    runtime = tmp_path / "runtime"
    for directory in ("source", "profile", "logs", "scripts", "results"):
        (runtime / directory).mkdir(parents=True)
    config = json.loads((ROOT / "configs/aspire-serving.json").read_text())
    (runtime / "source/aspire-serving.json").write_text(json.dumps(config))
    (runtime / "source/node-addresses.tsv").write_text(
        "0\tnode0\t192.0.2.10\n1\tnode1\t192.0.2.11\n"
    )
    original_score = '{"score_output_tokens_per_second":5925.07}\n'
    (runtime / "results/score.json").write_text(original_score)
    for worker in config["workers"]:
        name = f"{worker['role']}-{worker['index']}"
        (runtime / f"profile/nsys-{name}.argv").write_text("launch\n")
    capture = runtime / "scripts/capture-firmus-profile.sh"
    capture.write_text(
        '#!/bin/bash\nset -eu\n'
        'while IFS=$\'\\t\' read -r name host port tp; do '
        'printf report >"$1/profile/nsys-$name.nsys-rep"; done <"$2"\n'
        'printf drained >"$1/profile/capture-drained"\n'
        f"exit {capture_status}\n"
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    enroot = binaries / "enroot"
    enroot.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "root = Path(os.environ['PROFILE_TEST_RUNTIME'])\n"
        "with (root / 'calls.jsonl').open('a') as f:\n"
        " f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if '--version' in sys.argv:\n print('Nsight fixture')\n"
        "elif any(x.endswith('official-aiperf.sh') for x in sys.argv):\n"
        " assert sys.argv[-1] == 'profile'\n"
        f" sys.exit({client_status})\n"
        "elif any(x.endswith('validate-firmus-profile.py') for x in sys.argv):\n"
        " assert '--endpoints' in sys.argv\n"
        " (root / 'profile/validated').touch()\n"
        "else:\n raise AssertionError(sys.argv)\n"
    )
    enroot.chmod(0o755)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    process = subprocess.run(
        ["bash", str(ROOT / "scripts/run-aspire-profile.sh"), str(runtime)],
        env={
            **os.environ,
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "TMPDIR": str(scratch),
            "CONTAINER_IMAGE": "/fixture/image.sqsh",
            "PROFILE_TEST_RUNTIME": str(runtime),
        },
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert process.returncode == (capture_status or client_status), process.stderr
    assert (runtime / "results/score.json").read_text() == original_score
    assert (runtime / "profile/capture-drained").is_file()
    assert (runtime / "profile/validated").is_file() == (capture_status == 0)
    assert not list(scratch.iterdir())
    endpoints = (runtime / "profile/worker-endpoints.tsv").read_text().splitlines()
    assert endpoints == [
        f"{w['role']}-{w['index']}\t192.0.2.{10 + w['node']}\t{w['port']}\t{w['tp']}"
        for w in config["workers"]
    ]
    assert (runtime / "logs/profile-status.txt").read_text() == (
        f"profile_client_status={client_status}\n"
        f"profile_capture_status={capture_status}\n"
    )


def test_profile_starts_only_after_official_pass_and_keeps_failure_status() -> None:
    source = (ROOT / "submission/run/submit.sbatch").read_text()
    normal = source.index('"${script_path}" aiperf >')
    profile = source.index('bash "${OUTPUT_DIR}/scripts/run-aspire-profile.sh"')
    failure = source.index('(( benchmark_status == 0 )) || exit')
    success = source.index('touch "${OUTPUT_DIR}/run-success"')
    assert normal < profile < failure < success


@pytest.mark.parametrize(
    "platform,profile,status",
    [
        ("aspire2ap", "true", 0),
        ("aspire2ap", "true", 7),
        ("aspire2ap", "false", 7),
        ("firmus", "true", 7),
        ("genkai", "true", 7),
    ],
)
def test_only_aspire_profile_defers_benchmark_failure(
    tmp_path: Path, platform: str, profile: str, status: int
) -> None:
    for directory in ("scripts", "logs"):
        (tmp_path / directory).mkdir()
    benchmark = tmp_path / "benchmark"
    benchmark.write_text(f"#!/bin/bash\nexit {status}\n")
    benchmark.chmod(0o755)
    (tmp_path / "scripts/run-aspire-profile.sh").write_text(
        'touch "$1/profile-called"\n'
    )
    source = (ROOT / "submission/run/submit.sbatch").read_text()
    tail = source[source.index("benchmark_status=0\n"):]
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + tail],
        env={
            **os.environ,
            "QWEN_PLATFORM": platform,
            "PROFILE": profile,
            "OUTPUT_DIR": str(tmp_path),
            "script_path": str(benchmark),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == status, result.stderr
    assert (tmp_path / "profile-called").exists() == (
        platform == "aspire2ap" and profile == "true"
    )
    assert (tmp_path / "run-success").exists() == (status == 0)
