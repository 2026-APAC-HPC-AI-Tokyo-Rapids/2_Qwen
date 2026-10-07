"""Exercise the shared distribution boundary with local S3 fixtures."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_materialize_checks_framework_and_content(
    tmp_path: Path, framework: str, corrupt: bool
) -> None:
    source = tmp_path / "s3/containers/commit/123-1"
    source.mkdir(parents=True)
    image = b"fixture squashfs bytes"
    digest = hashlib.sha256(image).hexdigest()
    # Keep the length unchanged so this exercises the checksum, not just size.
    (source / f"{framework}.sqsh").write_bytes(b"X" + image[1:] if corrupt else image)
    sidecar = {
        "schema_version": "container-sidecar/v1",
        "container_artifact": {
            "artifact_relative_path": f"{framework}.sqsh",
            "size_bytes": len(image),
            "sha256": digest,
        },
    }
    (source / "container-sidecar.json").write_text(json.dumps(sidecar))
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "123.sqsh").write_bytes(b"previous verified image")
    (archive / "123.container-sidecar.json").write_text("previous sidecar")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name, code in {
        "uvx": "import os, shutil, sys\nshutil.copyfile(os.environ['MOCK_S3'] + '/' + sys.argv[-2], sys.argv[-1])\n",
        "stat": "import os, sys\nprint(os.stat(sys.argv[-1]).st_size)\n",
    }.items():
        path = fake_bin / name
        path.write_text("#!/usr/bin/env python3\n" + code)
        path.chmod(0o755)
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "MOCK_S3": str(tmp_path / "s3"),
        "ACTION": "build",
        "CONTAINER_ARCHIVE_DIR": str(archive),
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "123",
        "CONTAINER_BUCKET": "fixture-bucket",
        "CONTAINER_KEY": f"containers/commit/123-1/{framework}.sqsh",
        "SIDECAR_KEY": "containers/commit/123-1/container-sidecar.json",
        "CONTAINER_SHA256": digest,
        "CONTAINER_SIZE": str(len(image)),
        "GITHUB_WORKSPACE": str(ROOT),
    }
    # SGLang must keep working for existing callers that omit the new variable.
    environment.pop("CONTAINER_FRAMEWORK", None)
    if framework == "vllm":
        environment["CONTAINER_FRAMEWORK"] = framework
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/materialize-qwen-container.sh")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if corrupt:
        assert result.returncode != 0
        assert (archive / "123.sqsh").read_bytes() == b"previous verified image"
        assert (archive / "123.container-sidecar.json").read_text() == "previous sidecar"
    else:
        assert result.returncode == 0, result.stderr
        assert (archive / "123.sqsh").read_bytes() == image
        assert json.loads((archive / "123.container-sidecar.json").read_text()) == sidecar
    assert not list(archive.glob(".*.tmp"))


def test_aws_entrypoint_rejects_unknown_framework_before_any_work() -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "aws/run-build-instance.sh"),
            "fixture-bucket",
            "inputs/123/1/input.tar.zst",
            "containers/commit/123-1",
            "123",
            "1",
            "a" * 40,
            "123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/cache",
            "../wrong",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "aws:" not in result.stderr
