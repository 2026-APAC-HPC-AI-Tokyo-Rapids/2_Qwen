"""A build/source pin must still verify the real archive against its SHA."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("corruption", [None, "source", "archive", "size", "hash"])
def test_commit_pin_rejects_changed_image_or_provenance(tmp_path, corruption):
    image = tmp_path / "123.sqsh"
    image.write_bytes(b"fixture image")
    commit = "a" * 40
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    config = {"container_build_run_id": 123, "container_size_bytes": image.stat().st_size, "container_sha256": None, "container_source_commit": commit}
    sidecar = {"schema_version": "container-sidecar/v1", "source_commits": {"qwen": commit}, "container_artifact": {"artifact_relative_path": "sglang.sqsh", "sha256": digest, "size_bytes": image.stat().st_size}}
    if corruption == "source":
        sidecar["source_commits"]["qwen"] = "b" * 40
    elif corruption == "archive":
        image.write_bytes(b"fixture wrong")
    elif corruption == "size":
        config["container_size_bytes"] += 1
    elif corruption == "hash":
        sidecar["container_artifact"]["sha256"] = "0" * 64
    (tmp_path / "123.container-sidecar.json").write_text(json.dumps(sidecar))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    # This fixture exercises the Linux stat interface on macOS as well.
    binaries = tmp_path / "bin"
    binaries.mkdir()
    stat = binaries / "stat"
    stat.write_text('#!/usr/bin/env python3\nimport os,sys\nassert sys.argv[1:3] == ["-c", "%s"]\nprint(os.stat(sys.argv[3]).st_size)\n')
    stat.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(["bash", str(ROOT / "scripts/select-qwen-container.sh"), str(path)], capture_output=True, text=True, env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}", "CONTAINER_ARCHIVE_DIR": str(tmp_path), "GITHUB_OUTPUT": str(output), "GITHUB_RUN_ID": "456", "GITHUB_SHA": "c" * 40, "REQUIRE_BUILD": "false"}, check=False)
    assert (result.returncode == 0) is (corruption is None), result.stderr
    if corruption is None:
        assert output.read_text() == f"action=reuse\nsource_run_id=123\ncontainer_sha256={digest}\ncontainer_size=13\n"
    else:
        assert not output.exists()
