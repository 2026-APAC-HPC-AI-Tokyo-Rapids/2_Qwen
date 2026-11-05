"""Failure cases identified in the independent review of the submission entrypoints."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import submission_harness

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def harness(tmp_path):
    for name in ("bin", "srun", "launch"):
        (tmp_path / name).mkdir()
    submission_harness.sbatch_stubs(tmp_path)
    return tmp_path


@pytest.mark.parametrize("kind,failure,warning", [("bonus-sbatch", "dispatch", "dispatch extraction failed"), ("bonus-sbatch", "greedy", "Greedy probe failed"), ("bonus-sbatch", "probe-timeout", "Greedy probe failed"), ("basic-sbatch", "cleanup", "Compiler cache cleanup failed")])
def test_optional_diagnostic_failure_preserves_completed_benchmark(harness, kind, failure, warning):
    run = submission_harness.run_sbatch(kind, ROOT, harness, failure=failure)
    assert (run / "run-success").exists()
    assert (run / "results/score.json").exists()
    assert warning in (harness / "sbatch.out").read_text()


@pytest.mark.parametrize("kind", ["bonus-sbatch", "basic-sbatch"])
def test_failed_benchmark_never_marks_success(harness, kind):
    with pytest.raises(RuntimeError, match="exited 22"):
        submission_harness.run_sbatch(kind, ROOT, harness, failure="aiperf")
    assert not (harness / "run/run-success").exists()
    assert not (harness / "run/results/score.json").exists()
    if kind == "bonus-sbatch":
        assert "srun greedy-probe" not in (harness / "events.log").read_text()


def test_wrong_bonus_image_stops_before_download_or_overlay(harness):
    with pytest.raises(RuntimeError, match="Image SHA-256 differs"):
        submission_harness.run_prepare("bonus-prepare", ROOT, harness, image_sha256="0" * 64)
    assert not (harness / "events.log").exists()
    assert not (harness / "out/run/overlay").exists()


@pytest.mark.parametrize("remaining,reason", [(30, "less than 90 seconds"), (None, "SLURM_JOB_END_TIME is unavailable")])
def test_probe_skips_with_the_correct_reason(harness, remaining, reason):
    run = submission_harness.run_sbatch("bonus-sbatch", ROOT, harness, remaining=remaining)
    assert (run / "run-success").exists()
    assert not (harness / "probe-timeout-argv").exists()
    assert reason in (harness / "sbatch.out").read_text()


def test_probe_has_a_single_overall_timeout(harness):
    submission_harness.run_sbatch("bonus-sbatch", ROOT, harness)
    assert (harness / "probe-timeout-argv").read_text().splitlines()[:2] == ["--kill-after=5s", "60s"]


@pytest.mark.parametrize("config_sha,sidecar_sha,metadata", [("0" * 64, None, None), (None, "0" * 64, None), ("0" * 64, "0" * 64, None), (None, None, "size"), (None, None, "commit")])
def test_basic_rejects_image_and_metadata_mismatches(harness, config_sha, sidecar_sha, metadata):
    with pytest.raises(RuntimeError, match="Image, sidecar and Basic configuration do not match"):
        submission_harness.run_prepare("basic-prepare", ROOT, harness, image_sha256=config_sha, sidecar_sha256=sidecar_sha, basic_mismatch=metadata)
    assert not (harness / "events.log").exists()


@pytest.mark.parametrize("kind,failure", [("bonus", None), ("basic", None), ("bonus", "existing-build"), ("basic", "existing-build"), ("bonus", "nested-archive"), ("basic", "nested-archive"), ("bonus", "missing-mirror")])
def test_build_wrapper_isolates_storage_before_calling_frozen_builder(tmp_path, kind, failure):
    wrapper = ROOT / "submission" / ("build/setup.sh" if kind == "bonus" else "basic/build/setup.sh")
    build = tmp_path / "build"
    package = tmp_path / "package"
    (package / "source/scripts").mkdir(parents=True)
    shutil.copy(wrapper, package / "setup.sh")
    mirror = tmp_path / "mirror"
    (mirror / "5_SGLang_Src.git").mkdir(parents=True)
    if failure == "existing-build":
        build.mkdir()
        (build / "keep").write_text("unrelated data")
    if failure == "missing-mirror":
        (mirror / "5_SGLang_Src.git").rmdir()
    archive = build / "archive" if failure == "nested-archive" else tmp_path / "archive"
    script = package / "source/scripts" / ("build-sglang-container.sh" if kind == "bonus" else "build-basic-container.sh")
    script.write_text('#!/bin/bash\nprintf \'%s\\n\' "$BUILD_STORAGE_ROOT" >"$BUILD_DIR/called"\n')
    script.chmod(0o755)
    env = {**os.environ, "BUILD_DIR": str(build), "CONTAINER_ARCHIVE_DIR": str(archive), "GIT_MIRROR_DIR": str(mirror), "BUILD_STORAGE_ROOT": "/must-not-use-shared-storage"}
    result = subprocess.run(["bash", str(package / "setup.sh")], env=env, capture_output=True, text=True, timeout=10, check=False)
    if failure:
        assert result.returncode != 0
        assert not (build / "called").exists()
        if failure == "existing-build":
            assert (build / "keep").read_text() == "unrelated data"
    else:
        assert result.returncode == 0, result.stderr
        assert (build / "called").read_text().strip() == str(build / "storage")


def test_draft_adapter_records_download_identity_without_changing_weights(tmp_path):
    draft = tmp_path / "draft"
    draft.mkdir()
    original = {"architectures": ["LlamaForCausalLM"], "torch_dtype": "float32"}
    (draft / "config.json").write_text(json.dumps(original))
    (draft / "README.md").write_text("Public pinned draft")
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
    weights = struct.pack("<Q", len(header)) + header + b"\x00" * 4
    (draft / "model.safetensors").write_bytes(weights)
    pins = json.loads((ROOT / "submission/config/serving.json").read_text())["speculation"]
    records = tmp_path / "records"
    subprocess.run([sys.executable, str(ROOT / "submission/run/prepare_eagle.py"), str(draft), str(records), pins["draft_repository"], pins["draft_revision"]], check=True)
    provenance = json.loads((records / "draft-provenance.json").read_text())
    assert provenance["repository"] == pins["draft_repository"]
    assert provenance["revision"] == pins["draft_revision"]
    assert provenance["weight_sha256"] == hashlib.sha256(weights).hexdigest()
    assert provenance["parameter_count"] == 2
    assert (draft / "model.safetensors").read_bytes() == weights
    assert json.loads((records / "draft-original-config.json").read_text()) == original
    assert json.loads((draft / "config.json").read_text()) == {"architectures": ["LlamaForCausalLMEagle3"], "torch_dtype": "bfloat16"}
