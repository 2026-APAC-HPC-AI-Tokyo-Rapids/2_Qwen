from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import EXECUTION_SETTINGS, make_manifest

from qwen_experiment_metrics.canonical import jcs_bytes, load_json_strict
from qwen_experiment_metrics.sources import (
    capture_runtime_launchers,
    create_cache_manifest,
    create_input_manifests,
    create_submission_request,
    execution_facts,
    load_execution_settings,
    validate_cache_manifest,
    validate_runtime_launchers,
)


def test_execution_settings_are_closed_and_grid_consistent(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_bytes(jcs_bytes(EXECUTION_SETTINGS) + b"\n")
    settings = load_execution_settings(path)
    assert settings.run_purpose == "bench"
    assert {fact.key for fact in execution_facts(settings)} == {
        f"scheduler.request.{name}"
        for name in EXECUTION_SETTINGS
        if name not in {"schema_version", "run_purpose"}
    }

    invalid = dict(EXECUTION_SETTINGS, tensor_parallel_size=7)
    path.write_bytes(jcs_bytes(invalid) + b"\n")
    with pytest.raises(ValueError, match="full GPU grid"):
        load_execution_settings(path)


def test_input_manifests_bind_immutable_revisions_and_small_files(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "run"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "chat_template.json").write_text("{}\n", encoding="utf-8")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_bytes(b"[]\n")

    model_manifest, dataset_manifest = create_input_manifests(
        runtime,
        model_path=model,
        model_repository="organization/model",
        model_revision="a" * 40,
        dataset_path=dataset,
        dataset_repository="organization/dataset",
        dataset_revision="b" * 40,
        dataset_relative_path="dataset.json",
    )
    model_value = load_json_strict(model_manifest.read_bytes())
    assert [row["path"] for row in model_value["small_metadata"]] == [
        "chat_template.json",
        "config.json",
        "model.safetensors.index.json",
    ]
    assert load_json_strict(dataset_manifest.read_bytes())["full_revision"] == "b" * 40


def test_cache_identity_excludes_run_binding(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "kernel.bin").write_bytes(b"kernel")
    (cache / "current").symlink_to("kernel.bin")
    facts = []
    for name in ("first", "second"):
        runtime = tmp_path / name
        manifest = make_manifest(runtime)
        source, fact = create_cache_manifest(
            manifest, runtime, node_rank=0, cache_root=cache
        )
        validate_cache_manifest(source, load_json_strict(manifest.read_bytes())["run_id"], 0)
        facts.append(fact.logical_value)
    assert facts[0] == facts[1]


def test_runtime_launcher_source_binds_resolved_files_and_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launchers = tmp_path / "bin"
    launchers.mkdir()
    for role in ("mpirun", "singularity"):
        path = launchers / role
        path.write_text("#!/bin/sh\nprintf '%s version\\n' \"$0\"\n", encoding="utf-8")
        path.chmod(0o755)
    monkeypatch.setattr(
        "qwen_experiment_metrics.sources.shutil.which",
        lambda role: str(launchers / role),
    )
    runtime = tmp_path / "run"
    source = capture_runtime_launchers(runtime)
    value = validate_runtime_launchers(source, runtime)
    assert [row["role"] for row in value["executables"]] == [
        "mpirun",
        "singularity",
    ]
    (runtime / value["versions"][0]["stdout"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash binding"):
        validate_runtime_launchers(source, runtime)


def test_runtime_launcher_invokes_multicall_symlink_without_resolving_argv0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launchers = tmp_path / "bin"
    launchers.mkdir()
    wrapper = launchers / "env.sh"
    wrapper.write_text(
        "#!/bin/sh\n"
        "case $0 in\n"
        "  */mpirun|*/singularity) printf '%s version\\n' \"${0##*/}\" ;;\n"
        "  *) exit 127 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    for role in ("mpirun", "singularity"):
        (launchers / role).symlink_to(wrapper.name)
    monkeypatch.setattr(
        "qwen_experiment_metrics.sources.shutil.which",
        lambda role: str(launchers / role),
    )

    runtime = tmp_path / "run"
    source = capture_runtime_launchers(runtime)
    value = validate_runtime_launchers(source, runtime)

    assert all(
        row["requested_path"].endswith(f"/{row['role']}")
        and row["resolved_realpath"].endswith("/env.sh")
        for row in value["executables"]
    )
    assert all(
        row["argv"][0].endswith(f"/{row['role']}")
        and row["outcome"] == {"exit_code": 0}
        for row in value["versions"]
    )


def test_submission_request_binds_manifest_script_and_exact_argv(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "run"
    manifest = make_manifest(runtime)
    script = tmp_path / "run.pjm"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    argv = ["pjsub", "-N", "qwen", str(script)]
    source, facts = create_submission_request(
        manifest,
        runtime,
        pjm_script=script,
        repository_path="scripts/run-qwen35.pjm",
        argv=argv,
    )
    assert load_json_strict(source.read_bytes())["argv"] == argv
    assert {fact.key for fact in facts} == {
        "scheduler.request.subject_manifest_sha256",
        "scheduler.request.pjm_script.repository_path",
        "scheduler.request.pjm_script.sha256",
        "scheduler.request.argv",
    }
