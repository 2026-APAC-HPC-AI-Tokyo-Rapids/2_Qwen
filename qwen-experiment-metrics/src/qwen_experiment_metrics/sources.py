from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .canonical import jcs_bytes, load_json_strict
from .domain import Fact
from .manifest import SHA256_PATTERN, SourceTrack, load_manifest

_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_EXECUTION_FIELDS = {
    "schema_version",
    "run_purpose",
    "nodes",
    "mpi_processes",
    "gpus_per_node",
    "tensor_parallel_size",
    "resource_group",
    "transport_policy",
    "advertise_source",
    "sglang_port",
    "node_agent_port",
    "rendezvous_timeout_sec",
    "ready_timeout_sec",
}


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    run_purpose: str
    nodes: int
    mpi_processes: int
    gpus_per_node: int
    tensor_parallel_size: int
    resource_group: str
    transport_policy: str
    advertise_source: str
    sglang_port: int
    node_agent_port: int
    rendezvous_timeout_sec: int
    ready_timeout_sec: int


def install_execution_settings(source: Path, runtime_dir: Path) -> Path:
    settings = load_execution_settings(source)
    destination = runtime_dir / "control/execution-settings.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    _atomic_bytes(destination, jcs_bytes(_execution_object(settings)) + b"\n")
    return destination


def load_execution_settings(path: Path) -> ExecutionSettings:
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict) or set(value) != _EXECUTION_FIELDS:
        raise ValueError("execution settings are not a closed v1 object")
    if value["schema_version"] != "qwen-execution-settings/v1":
        raise ValueError("unsupported execution settings schema")
    uints = {
        name: _positive_uint(value[name], name)
        for name in (
            "nodes",
            "mpi_processes",
            "gpus_per_node",
            "tensor_parallel_size",
            "sglang_port",
            "node_agent_port",
            "rendezvous_timeout_sec",
            "ready_timeout_sec",
        )
    }
    for port in (uints["sglang_port"], uints["node_agent_port"]):
        if port > 65535:
            raise ValueError("execution port exceeds 65535")
    if uints["mpi_processes"] != uints["nodes"]:
        raise ValueError("one outer MPI process per node is required")
    if uints["tensor_parallel_size"] != (uints["nodes"] * uints["gpus_per_node"]):
        raise ValueError("tensor parallel size must cover the full GPU grid")
    strings = {
        name: _nonempty_string(value[name], name)
        for name in (
            "resource_group",
            "transport_policy",
            "advertise_source",
        )
    }
    run_purpose = value["run_purpose"]
    if run_purpose not in {"bench", "profile"}:
        raise ValueError("execution run purpose must be bench or profile")
    return ExecutionSettings(run_purpose=run_purpose, **uints, **strings)


def execution_facts(settings: ExecutionSettings) -> tuple[Fact, ...]:
    return tuple(
        Fact(f"scheduler.request.{name}", value)
        for name, value in _execution_object(settings).items()
        if name not in {"schema_version", "run_purpose"}
    )


def validate_source_track_repository(
    repository: Path, source_track: SourceTrack
) -> None:
    checkout = repository.resolve(strict=True)
    if _git(checkout, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("source checkout is not clean")
    if (
        _git(checkout, "rev-parse", "--show-object-format")
        != source_track.object_hash_algorithm
    ):
        raise ValueError("source checkout object algorithm differs")
    for role, oid in (
        ("base", source_track.base_commit),
        ("head", source_track.head_commit),
    ):
        resolved = _git(checkout, "rev-parse", f"{oid}^{{commit}}")
        if resolved != oid:
            raise ValueError(f"source {role} commit is not the exact full OID")
    outcome = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "merge-base",
            "--is-ancestor",
            source_track.base_commit,
            source_track.head_commit,
        ],
        check=False,
    )
    if outcome.returncode != 0:
        raise ValueError("source base commit is not an ancestor of head")


def create_input_manifests(
    runtime_dir: Path,
    *,
    model_path: Path,
    model_repository: str,
    model_revision: str,
    dataset_path: Path,
    dataset_repository: str,
    dataset_revision: str,
    dataset_relative_path: str,
) -> tuple[Path, Path]:
    _require_oid(model_revision, "model revision")
    _require_oid(dataset_revision, "dataset revision")
    model_root = model_path.resolve(strict=True)
    if not model_root.is_dir():
        raise ValueError("model path must be a directory")
    index_path = model_root / "model.safetensors.index.json"
    index = load_json_strict(index_path.read_bytes())
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise TypeError("model weight index is invalid")
    raw_shards = list(index["weight_map"].values())
    if not raw_shards or any(not isinstance(item, str) for item in raw_shards):
        raise ValueError("model weight index contains invalid shard names")
    shards = sorted(set(raw_shards), key=lambda item: item.encode())
    for shard in shards:
        _safe_relative(shard)
        if not (model_root / shard).is_file():
            raise FileNotFoundError(model_root / shard)

    role_by_path = {
        "config.json": "configuration",
        "model.safetensors.index.json": "index",
        "tokenizer.json": "tokenizer",
        "tokenizer_config.json": "tokenizer_config",
        "special_tokens_map.json": "special_tokens",
        "generation_config.json": "generation_config",
        "chat_template.json": "chat_template",
    }
    metadata = []
    for relative, role in sorted(
        role_by_path.items(), key=lambda item: item[0].encode()
    ):
        candidate = model_root / relative
        if candidate.is_file():
            metadata.append(_member(candidate, relative, role=role))
    if not {"configuration", "index"}.issubset({item["role"] for item in metadata}):
        raise ValueError("model manifest requires config and weight index")
    model = {
        "schema_version": "model-manifest/v1",
        "source": {
            "kind": "hub_revision",
            "repository": _nonempty_string(model_repository, "model repository"),
            "revision": model_revision,
        },
        "small_metadata": metadata,
    }

    dataset_file = dataset_path.resolve(strict=True)
    _safe_relative(dataset_relative_path)
    dataset = {
        "schema_version": "dataset-manifest/v1",
        "repository": _nonempty_string(dataset_repository, "dataset repository"),
        "full_revision": dataset_revision,
        "relative_path": dataset_relative_path,
        "size_bytes": dataset_file.stat().st_size,
        "sha256": _sha256(dataset_file),
    }
    source_dir = runtime_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    model_destination = source_dir / "model-manifest.json"
    dataset_destination = source_dir / "dataset-manifest.json"
    for destination, payload in (
        (model_destination, model),
        (dataset_destination, dataset),
    ):
        if destination.exists():
            raise FileExistsError(destination)
        _atomic_bytes(destination, jcs_bytes(payload) + b"\n")
    validate_model_manifest(model_destination)
    validate_dataset_manifest(dataset_destination)
    return model_destination, dataset_destination


def validate_model_manifest(path: Path) -> dict[str, object]:
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "source",
        "small_metadata",
    }:
        raise ValueError("model manifest is not closed")
    if value["schema_version"] != "model-manifest/v1":
        raise ValueError("unsupported model manifest")
    source = value["source"]
    if not isinstance(source, dict):
        raise TypeError("model source is not a closed tagged union")
    if source.get("kind") == "hub_revision":
        if set(source) != {"kind", "repository", "revision"}:
            raise ValueError("hub model source is not closed")
        _nonempty_string(source["repository"], "model repository")
        _require_oid(source["revision"], "model revision")
    elif source.get("kind") == "prebuilt_manifest":
        if set(source) != {"kind", "shards"}:
            raise ValueError("prebuilt model source is not closed")
        shards = source["shards"]
        if not isinstance(shards, list) or not shards:
            raise ValueError("prebuilt model shards must be nonempty")
        shard_paths = [_validate_member(row, require_role=False) for row in shards]
        if shard_paths != sorted(shard_paths, key=lambda item: item.encode()) or len(
            shard_paths
        ) != len(set(shard_paths)):
            raise ValueError("prebuilt model shards must be sorted and unique")
    else:
        raise ValueError("unknown model source kind")
    rows = value["small_metadata"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("model small_metadata must be nonempty")
    paths = [_validate_member(row, require_role=True) for row in rows]
    if paths != sorted(paths, key=lambda item: item.encode()) or len(paths) != len(
        set(paths)
    ):
        raise ValueError("model metadata paths must be sorted and unique")
    roles = [row["role"] for row in rows]
    allowed_roles = {
        "configuration",
        "index",
        "tokenizer",
        "tokenizer_config",
        "special_tokens",
        "generation_config",
        "chat_template",
    }
    if (
        any(role not in allowed_roles for role in roles)
        or len(roles) != len(set(roles))
        or not {"configuration", "index"}.issubset(roles)
    ):
        raise ValueError("model metadata roles are invalid")
    return value


def validate_dataset_manifest(path: Path) -> dict[str, object]:
    value = load_json_strict(path.read_bytes())
    expected = {
        "schema_version",
        "repository",
        "full_revision",
        "relative_path",
        "size_bytes",
        "sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("dataset manifest is not closed")
    if value["schema_version"] != "dataset-manifest/v1":
        raise ValueError("unsupported dataset manifest")
    _nonempty_string(value["repository"], "dataset repository")
    _require_oid(value["full_revision"], "dataset revision")
    _safe_relative(_nonempty_string(value["relative_path"], "dataset path"))
    _uint(value["size_bytes"], "dataset size")
    _require_sha256(value["sha256"], "dataset sha256")
    return value


def input_manifest_facts(runtime_dir: Path) -> tuple[Fact, ...]:
    model = runtime_dir / "source/model-manifest.json"
    dataset = runtime_dir / "source/dataset-manifest.json"
    model_value = validate_model_manifest(model)
    dataset_value = validate_dataset_manifest(dataset)
    return (
        Fact("input.model", model_value),
        Fact("input.dataset", dataset_value),
    )


def create_submission_request(
    manifest_path: Path,
    runtime_dir: Path,
    *,
    pjm_script: Path,
    repository_path: str,
    argv: list[str],
) -> tuple[Path, tuple[Fact, ...]]:
    manifest = load_manifest(manifest_path)
    _safe_relative(repository_path)
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ValueError("submission argv must be a nonempty string array")
    script = pjm_script.resolve(strict=True)
    if not script.is_file():
        raise ValueError("PJM script must be a regular file")
    request = {
        "schema_version": "submission-request/v1",
        "run_id": manifest.run_id,
        "subject_manifest": _member(manifest_path, "control/run-manifest.json"),
        "pjm_script": {
            "repository_path": repository_path,
            "size_bytes": script.stat().st_size,
            "sha256": _sha256(script),
        },
        "argv": argv,
    }
    destination = runtime_dir / "control/submission-request.json"
    if destination.exists():
        raise FileExistsError(destination)
    _atomic_bytes(destination, jcs_bytes(request) + b"\n")
    validate_submission_request(destination, manifest_path, manifest.run_id)
    return destination, (
        Fact(
            "scheduler.request.subject_manifest_sha256",
            request["subject_manifest"]["sha256"],
        ),
        Fact("scheduler.request.pjm_script.repository_path", repository_path),
        Fact("scheduler.request.pjm_script.sha256", request["pjm_script"]["sha256"]),
        Fact("scheduler.request.argv", argv),
    )


def validate_submission_request(
    path: Path, manifest_path: Path, run_id: str
) -> dict[str, object]:
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "run_id",
        "subject_manifest",
        "pjm_script",
        "argv",
    }:
        raise ValueError("submission request is not closed")
    if value["schema_version"] != "submission-request/v1" or value["run_id"] != run_id:
        raise ValueError("submission request identity differs")
    subject = value["subject_manifest"]
    if _validate_member(subject, require_role=False) != "control/run-manifest.json":
        raise ValueError("submission subject path differs")
    if (
        manifest_path.stat().st_size != subject["size_bytes"]
        or _sha256(manifest_path) != subject["sha256"]
    ):
        raise ValueError("submission subject hash binding failed")
    script = value["pjm_script"]
    if not isinstance(script, dict) or set(script) != {
        "repository_path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("submission PJM script record is not closed")
    _validated_relative(script["repository_path"], "PJM repository path")
    _uint(script["size_bytes"], "PJM script size")
    _require_sha256(script["sha256"], "PJM script sha256")
    argv = value["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        raise ValueError("submission argv is invalid")
    return value


def create_submission_result(
    manifest_path: Path, runtime_dir: Path, job_id: str
) -> tuple[Path, Fact]:
    manifest = load_manifest(manifest_path)
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise ValueError("submitted job ID is invalid")
    value = {
        "schema_version": "submission-result/v1",
        "run_id": manifest.run_id,
        "job_id": job_id,
    }
    destination = runtime_dir / "control/submission-result.json"
    if destination.exists():
        raise FileExistsError(destination)
    _atomic_bytes(destination, jcs_bytes(value) + b"\n")
    return destination, Fact("scheduler.submit_job_id", job_id)


def job_entry_fact(path: Path, run_id: str) -> Fact:
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "run_id",
        "pj_job_id",
        "entered_at",
    }:
        raise ValueError("job entry is not closed")
    if value["schema_version"] != "job-entry/v1" or value["run_id"] != run_id:
        raise ValueError("job entry identity differs")
    job_id = value["pj_job_id"]
    if not isinstance(job_id, str) or re.fullmatch(r"[0-9]+", job_id) is None:
        raise ValueError("job entry job ID is invalid")
    return Fact("scheduler.job_entry_job_id", job_id)


def create_cache_manifest(
    manifest_path: Path,
    runtime_dir: Path,
    *,
    node_rank: int,
    cache_root: Path,
) -> tuple[Path, Fact]:
    manifest = load_manifest(manifest_path)
    if node_rank < 0:
        raise ValueError("node rank must be nonnegative")
    root = cache_root.resolve(strict=True)
    members: list[dict[str, object]] = []
    for candidate in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix().encode()
    ):
        relative = candidate.relative_to(root).as_posix()
        _safe_relative(relative)
        if candidate.is_symlink():
            members.append(
                {
                    "type": "symlink",
                    "path": relative,
                    "target_b64": base64.b64encode(
                        os.fsencode(os.readlink(candidate))
                    ).decode(),
                }
            )
        elif candidate.is_file():
            members.append(
                {
                    "type": "file",
                    "path": relative,
                    "size_bytes": candidate.stat().st_size,
                    "sha256": _sha256(candidate),
                }
            )
    semantic = {"schema_version": "cache-members/v1", "members": members}
    identity = hashlib.sha256(jcs_bytes(semantic)).hexdigest()
    source = {
        "schema_version": "cache-manifest/v1",
        "run_id": manifest.run_id,
        "node_rank": node_rank,
        "members": members,
    }
    destination = runtime_dir / f"source/cache/node-{node_rank}-input-manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    _atomic_bytes(destination, jcs_bytes(source) + b"\n")
    validate_cache_manifest(destination, manifest.run_id, node_rank)
    return destination, Fact(f"cache.node.{node_rank}.input_manifest_sha256", identity)


def validate_cache_manifest(
    path: Path, run_id: str, node_rank: int
) -> dict[str, object]:
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "run_id",
        "node_rank",
        "members",
    }:
        raise ValueError("cache manifest is not closed")
    if (
        value["schema_version"] != "cache-manifest/v1"
        or value["run_id"] != run_id
        or value["node_rank"] != node_rank
    ):
        raise ValueError("cache manifest identity mismatch")
    rows = value["members"]
    if not isinstance(rows, list):
        raise TypeError("cache members must be an array")
    paths: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("type") not in {"file", "symlink"}:
            raise ValueError("invalid cache member")
        kind = row["type"]
        expected = (
            {"type", "path", "size_bytes", "sha256"}
            if kind == "file"
            else {"type", "path", "target_b64"}
        )
        if set(row) != expected:
            raise ValueError("cache member is not closed")
        relative = _nonempty_string(row["path"], "cache member path")
        _safe_relative(relative)
        paths.append(relative)
        if kind == "file":
            _uint(row["size_bytes"], "cache member size")
            _require_sha256(row["sha256"], "cache member sha256")
        else:
            base64.b64decode(
                _nonempty_string(row["target_b64"], "symlink target"), validate=True
            )
    if paths != sorted(paths, key=lambda item: item.encode()) or len(paths) != len(
        set(paths)
    ):
        raise ValueError("cache members must be path-byte sorted and unique")
    return value


def capture_runtime_launchers(runtime_dir: Path) -> Path:
    source_dir = runtime_dir / "source/runtime-launchers"
    source_dir.mkdir(parents=True, exist_ok=True)
    executables = []
    versions = []
    for role in ("mpirun", "singularity"):
        requested = shutil.which(role)
        if requested is None:
            raise FileNotFoundError(role)
        # Keep the executable spelling selected by PATH.  Some launchers are
        # multicall symlinks whose behavior depends on argv[0] (HPC-X mpirun
        # is a symlink to env.sh, for example), so invoking the resolved target
        # is not equivalent to invoking the requested launcher.  The resolved
        # regular file remains the immutable content-hash boundary.
        requested = str(Path(requested).absolute())
        resolved = Path(requested).resolve(strict=True)
        executables.append(
            {
                "role": role,
                "requested_path": requested,
                "resolved_realpath": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )
        stdout = source_dir / f"{role}.version.stdout"
        stderr = source_dir / f"{role}.version.stderr"
        outcome = _capture_command([requested, "--version"], stdout, stderr)
        versions.append(
            {
                "role": role,
                "argv": [requested, "--version"],
                "outcome": {"exit_code": outcome},
                "stdout": _member(stdout, stdout.relative_to(runtime_dir).as_posix()),
                "stderr": _member(stderr, stderr.relative_to(runtime_dir).as_posix()),
            }
        )
    source = {
        "schema_version": "runtime-launcher-validation/v1",
        "executables": executables,
        "versions": versions,
    }
    destination = runtime_dir / "source/runtime-launchers.json"
    if destination.exists():
        raise FileExistsError(destination)
    _atomic_bytes(destination, jcs_bytes(source) + b"\n")
    validate_runtime_launchers(destination, runtime_dir)
    return destination


def validate_runtime_launchers(path: Path, runtime_dir: Path) -> dict[str, object]:
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "executables",
        "versions",
    }:
        raise ValueError("runtime launcher source is not closed")
    if value["schema_version"] != "runtime-launcher-validation/v1":
        raise ValueError("unsupported runtime launcher source")
    executables = value["executables"]
    versions = value["versions"]
    if not isinstance(executables, list) or not isinstance(versions, list):
        raise TypeError("runtime launcher arrays are invalid")
    if [row.get("role") for row in executables if isinstance(row, dict)] != [
        "mpirun",
        "singularity",
    ]:
        raise ValueError("runtime launcher executable roles differ")
    if [row.get("role") for row in versions if isinstance(row, dict)] != [
        "mpirun",
        "singularity",
    ]:
        raise ValueError("runtime launcher version roles differ")
    requested_by_role: dict[str, str] = {}
    for row in executables:
        if not isinstance(row, dict) or set(row) != {
            "role",
            "requested_path",
            "resolved_realpath",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("runtime executable record is not closed")
        role = _nonempty_string(row["role"], "launcher role")
        requested = Path(
            _nonempty_string(row["requested_path"], "requested launcher path")
        )
        resolved = Path(_nonempty_string(row["resolved_realpath"], "launcher realpath"))
        if (
            not requested.is_absolute()
            or not requested.is_file()
            or requested.resolve(strict=True) != resolved
            or not resolved.is_absolute()
            or not resolved.is_file()
        ):
            raise ValueError("runtime launcher is not an absolute regular file")
        if (
            resolved.stat().st_size != _uint(row["size_bytes"], "launcher size")
            or _sha256(resolved) != row["sha256"]
        ):
            raise ValueError("runtime launcher hash binding failed")
        requested_by_role[role] = str(requested)
    for row in versions:
        if not isinstance(row, dict) or set(row) != {
            "role",
            "argv",
            "outcome",
            "stdout",
            "stderr",
        }:
            raise ValueError("runtime version record is not closed")
        if row["argv"] != [requested_by_role[row["role"]], "--version"]:
            raise ValueError("runtime version argv is invalid")
        outcome = row["outcome"]
        if not isinstance(outcome, dict) or set(outcome) != {"exit_code"}:
            raise ValueError("runtime version outcome is invalid")
        if _uint(outcome["exit_code"], "runtime version exit code") != 0:
            raise ValueError("runtime version command failed")
        for name in ("stdout", "stderr"):
            member = row[name]
            relative = _validate_member(member, require_role=False)
            candidate = runtime_dir / relative
            if (
                not candidate.is_file()
                or candidate.stat().st_size != member["size_bytes"]
                or _sha256(candidate) != member["sha256"]
            ):
                raise ValueError("runtime version member hash binding failed")
    return value


def runtime_launcher_facts(path: Path, runtime_dir: Path) -> tuple[Fact, ...]:
    value = validate_runtime_launchers(path, runtime_dir)
    versions = []
    for row in value["versions"]:
        stdout = runtime_dir / row["stdout"]["path"]
        stderr = runtime_dir / row["stderr"]["path"]
        versions.append(
            {
                "role": row["role"],
                "stdout": stdout.read_text(encoding="utf-8").strip(),
                "stderr": stderr.read_text(encoding="utf-8").strip(),
            }
        )
    return (
        Fact(
            "runtime.launchers",
            {
                "executables": value["executables"],
                "versions": versions,
            },
        ),
    )


def provenance_facts(runtime_dir: Path) -> tuple[Fact, ...]:
    sidecar = runtime_dir / "source/sif-sidecar.json"
    value = validate_sif_sidecar(sidecar, member_root=runtime_dir / "source")
    sif = runtime_dir / value["sif_artifact"]["artifact_relative_path"]
    if (
        not sif.is_file()
        or sif.stat().st_size != value["sif_artifact"]["size_bytes"]
        or _sha256(sif) != value["sif_artifact"]["sha256"]
    ):
        raise ValueError("provisioned SIF does not match its sidecar")
    source_commits = value["source_commits"]
    qwen = next(
        row for row in source_commits if row["commit_role"] == "sif_build_definition"
    )
    sglang = next(
        row for row in source_commits if row["commit_role"] == "sglang_source"
    )
    source_run_number = (runtime_dir / "sif-source-run.txt").read_text(
        encoding="utf-8"
    ).strip()
    if re.fullmatch(r"[0-9]+", source_run_number) is None:
        raise ValueError("SIF source workflow run number is invalid")
    source_run = int(source_run_number)
    toolchain = []
    for row in value["toolchain_commands"]:
        stdout = runtime_dir / "source" / row["stdout"]["path"]
        stderr = runtime_dir / "source" / row["stderr"]["path"]
        toolchain.append(
            {
                "role": row["role"],
                "stdout": stdout.read_text(encoding="utf-8").strip(),
                "stderr": stderr.read_text(encoding="utf-8").strip(),
            }
        )
    base_oci = value["base_oci"]
    return (
        Fact(
            "build.provenance",
            {
                "sif": {
                    "sha256": value["sif_artifact"]["sha256"],
                    "size_bytes": value["sif_artifact"]["size_bytes"],
                    "source_workflow_run_number": source_run,
                },
                "base_oci": {
                    "reference": base_oci["reference"],
                    "manifest_digest": base_oci["manifest_digest"],
                    "config_digest": base_oci["config_digest"],
                    "layer_digests": base_oci["layer_digests"],
                },
                "toolchain": toolchain,
                "nccl_tests_commit": value["nccl_tests"]["commit"],
            },
        ),
        Fact("source.sif_build.qwen_commit", qwen["oid"]),
        Fact("source.sglang.commit", sglang["oid"]),
        *input_manifest_facts(runtime_dir),
    )


def create_sif_sidecar(
    *,
    workspace: Path,
    sglang_source: Path,
    sif: Path,
    base_oci: Path,
    embedded_manifest: Path,
    runtime_inventory: Path,
    output: Path,
) -> Path:
    workspace = workspace.resolve(strict=True)
    sglang_source = sglang_source.resolve(strict=True)
    sif = sif.resolve(strict=True)
    base = load_json_strict(base_oci.read_bytes())
    if not isinstance(base, dict) or set(base) != {
        "reference",
        "inspect_raw",
        "manifest_digest",
        "config_digest",
        "layer_digests",
    }:
        raise ValueError("base OCI resolution is not closed")
    inventory = load_json_strict(runtime_inventory.read_bytes())
    if not isinstance(inventory, dict) or set(inventory) != {
        "runtime_files",
        "nccl_binaries",
    }:
        raise ValueError("runtime inventory is not closed")
    qwen_algorithm = _git(workspace, "rev-parse", "--show-object-format")
    sglang_algorithm = _git(sglang_source, "rev-parse", "--show-object-format")
    recipe_path = "containers/sglang.def"
    patch_paths = sorted(
        (
            path.relative_to(workspace).as_posix()
            for path in (workspace / "patches/sglang").glob("*.patch")
        ),
        key=lambda item: item.encode(),
    )
    lock_path = sglang_source / "python/uv.lock"
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    build_sources = output.parent / "sif-build"
    if not build_sources.is_dir():
        raise FileNotFoundError(build_sources)
    source_commits = [
        {
            "repository_id": "apac2026-qwen",
            "commit_role": "sif_build_definition",
            "object_hash_algorithm": qwen_algorithm,
            "oid": _git(workspace, "rev-parse", "HEAD"),
        },
        {
            "repository_id": "apac2026-sglang",
            "commit_role": "sglang_source",
            "object_hash_algorithm": sglang_algorithm,
            "oid": _git(sglang_source, "rev-parse", "HEAD"),
        },
    ]
    recipe = {
        "path": recipe_path,
        "blob_oid": _git(workspace, "rev-parse", f"HEAD:{recipe_path}"),
    }
    applied_patches = [
        {
            "path": relative,
            "blob_oid": _git(workspace, "rev-parse", f"HEAD:{relative}"),
        }
        for relative in patch_paths
    ]
    lock_manifest = {
        "path": "python/uv.lock",
        "size_bytes": lock_path.stat().st_size,
        "sha256": _sha256(lock_path),
    }
    embedded = load_json_strict(embedded_manifest.read_bytes())
    expected_embedded = {
        "schema_version": "sif-embedded-manifest/v1",
        "source_commits": source_commits,
        "recipe": recipe,
        "applied_patches": applied_patches,
        "base_oci": {
            name: base[name]
            for name in (
                "reference",
                "manifest_digest",
                "config_digest",
                "layer_digests",
            )
        },
        "lock_manifest": lock_manifest,
    }
    if embedded != expected_embedded:
        raise ValueError("embedded SIF manifest differs from resolved build inputs")
    embedded_semantic = {
        name: embedded[name]
        for name in (
            "source_commits",
            "recipe",
            "applied_patches",
            "base_oci",
            "lock_manifest",
        )
    }
    toolchain_commands = [
        _build_observation(
            output.parent, "singularity_version", ["singularity", "--version"]
        ),
        _build_observation(
            output.parent,
            "python_version",
            [
                "singularity",
                "exec",
                "sglang.sif",
                "/opt/sglang/venv/bin/python",
                "--version",
            ],
        ),
        _build_observation(
            output.parent,
            "cuda_version",
            [
                "singularity",
                "exec",
                "sglang.sif",
                "/usr/local/cuda/bin/nvcc",
                "--version",
            ],
        ),
        _build_observation(
            output.parent,
            "compiler_version",
            ["singularity", "exec", "sglang.sif", "/usr/bin/gcc", "--version"],
        ),
    ]
    source = {
        "schema_version": "sif-sidecar/v1",
        "sif_artifact": {
            "artifact_relative_path": "sglang.sif",
            "size_bytes": sif.stat().st_size,
            "sha256": _sha256(sif),
        },
        "source_commits": source_commits,
        "score_source_tree": [],
        "recipe": recipe,
        "applied_patches": applied_patches,
        "base_oci": base,
        "lock_manifest": lock_manifest,
        "build_command": [
            "singularity",
            "build",
            "--fakeroot",
            "--writable-tmpfs",
            "--sandbox",
            "sglang-rootfs",
            "containers/sglang.def",
        ],
        "toolchain_commands": toolchain_commands,
        "embedded_manifest": {
            "member_path": "/opt/apac/manifest.json",
            "size_bytes": embedded_manifest.stat().st_size,
            "sha256": _sha256(embedded_manifest),
            "manifest_identity": hashlib.sha256(
                jcs_bytes(embedded_semantic)
            ).hexdigest(),
        },
        "runtime_files": inventory["runtime_files"],
        "nccl_tests": {
            "commit": "717b68318278e93f371d8ffb46b076069d7c7851",
            "mpi_enabled": True,
            "json_output_enabled": True,
            "binaries": inventory["nccl_binaries"],
            "build_test": _build_test_observation(output.parent),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    _atomic_bytes(output, jcs_bytes(source) + b"\n")
    validate_sif_sidecar(output, member_root=output.parent)
    return output


def validate_sif_sidecar(
    path: Path, *, member_root: Path | None = None
) -> dict[str, object]:
    value = load_json_strict(path.read_bytes())
    expected = {
        "schema_version",
        "sif_artifact",
        "source_commits",
        "score_source_tree",
        "recipe",
        "applied_patches",
        "base_oci",
        "lock_manifest",
        "build_command",
        "toolchain_commands",
        "embedded_manifest",
        "runtime_files",
        "nccl_tests",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("SIF sidecar is not a closed v1 object")
    if value["schema_version"] != "sif-sidecar/v1":
        raise ValueError("unsupported SIF sidecar")
    _validate_member(
        value["sif_artifact"], require_role=False, path_name="artifact_relative_path"
    )
    commits = value["source_commits"]
    if not isinstance(commits, list) or len(commits) != 2:
        raise ValueError("SIF sidecar requires two source commits")
    expected_roles = ["sif_build_definition", "sglang_source"]
    if [
        row.get("commit_role") for row in commits if isinstance(row, dict)
    ] != expected_roles:
        raise ValueError("SIF source commit roles differ")
    if [row.get("repository_id") for row in commits if isinstance(row, dict)] != [
        "apac2026-qwen",
        "apac2026-sglang",
    ]:
        raise ValueError("SIF source repository roles differ")
    for row in commits:
        if not isinstance(row, dict) or set(row) != {
            "repository_id",
            "commit_role",
            "object_hash_algorithm",
            "oid",
        }:
            raise ValueError("SIF source commit is not closed")
        algorithm = row["object_hash_algorithm"]
        oid = row["oid"]
        if (
            algorithm not in {"sha1", "sha256"}
            or not isinstance(oid, str)
            or len(oid) != (40 if algorithm == "sha1" else 64)
            or re.fullmatch(r"[0-9a-f]+", oid) is None
        ):
            raise ValueError("SIF source object identity is invalid")
    score_tree = value["score_source_tree"]
    if not isinstance(score_tree, list):
        raise TypeError("SIF score source tree is invalid")
    score_paths: list[str] = []
    for row in score_tree:
        if not isinstance(row, dict) or row.get("state") not in {"absent", "present"}:
            raise ValueError("SIF score source entry is invalid")
        expected_entry = (
            {"path", "state"}
            if row["state"] == "absent"
            else {"path", "state", "mode", "blob_oid"}
        )
        if set(row) != expected_entry:
            raise ValueError("SIF score source entry is not closed")
        score_paths.append(_validated_relative(row["path"], "score source path"))
        if row["state"] == "present":
            if (
                re.fullmatch(
                    r"[0-7]{6}", _nonempty_string(row["mode"], "score source mode")
                )
                is None
            ):
                raise ValueError("SIF score source mode is invalid")
            _algorithm_oid(row["blob_oid"], commits[0]["object_hash_algorithm"])
    _require_sorted_unique(score_paths, "SIF score source paths")

    recipe = value["recipe"]
    if not isinstance(recipe, dict) or set(recipe) != {"path", "blob_oid"}:
        raise ValueError("SIF recipe is not closed")
    _validated_relative(recipe["path"], "SIF recipe path")
    _algorithm_oid(recipe["blob_oid"], commits[0]["object_hash_algorithm"])
    patches = value["applied_patches"]
    if not isinstance(patches, list):
        raise TypeError("SIF patch list is invalid")
    patch_paths: list[str] = []
    for row in patches:
        if not isinstance(row, dict) or set(row) != {"path", "blob_oid"}:
            raise ValueError("SIF patch record is not closed")
        patch_paths.append(_validated_relative(row["path"], "SIF patch path"))
        _algorithm_oid(row["blob_oid"], commits[0]["object_hash_algorithm"])
    _require_sorted_unique(patch_paths, "SIF patch paths")

    base = value["base_oci"]
    if not isinstance(base, dict) or set(base) != {
        "reference",
        "inspect_raw",
        "manifest_digest",
        "config_digest",
        "layer_digests",
    }:
        raise ValueError("SIF base OCI record is not closed")
    reference = _nonempty_string(base["reference"], "OCI reference")
    if "@sha256:" not in reference:
        raise ValueError("OCI reference is not digest pinned")
    _validate_member_binding(base["inspect_raw"], member_root)
    for name in ("manifest_digest", "config_digest"):
        if (
            re.fullmatch(r"sha256:[0-9a-f]{64}", _nonempty_string(base[name], name))
            is None
        ):
            raise ValueError(f"invalid OCI {name}")
    if base["inspect_raw"]["sha256"] != base["manifest_digest"].removeprefix("sha256:"):
        raise ValueError("OCI raw manifest hash differs from manifest digest")
    layers = base["layer_digests"]
    if (
        not isinstance(layers, list)
        or not layers
        or any(
            not isinstance(item, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", item) is None
            for item in layers
        )
    ):
        raise ValueError("invalid OCI layers")
    lock_manifest = value["lock_manifest"]
    if not isinstance(lock_manifest, dict):
        raise TypeError("SIF lock manifest is invalid")
    _validate_member(lock_manifest, require_role=False)
    build_command = value["build_command"]
    if (
        not isinstance(build_command, list)
        or not build_command
        or any(not isinstance(item, str) or not item for item in build_command)
    ):
        raise ValueError("SIF build command is invalid")
    toolchain = value["toolchain_commands"]
    toolchain_roles = [
        "singularity_version",
        "python_version",
        "cuda_version",
        "compiler_version",
    ]
    if (
        not isinstance(toolchain, list)
        or [row.get("role") for row in toolchain if isinstance(row, dict)]
        != toolchain_roles
    ):
        raise ValueError("SIF toolchain command roles differ")
    for row in toolchain:
        _validate_observation(row, member_root)
    embedded = value["embedded_manifest"]
    if not isinstance(embedded, dict) or set(embedded) != {
        "member_path",
        "size_bytes",
        "sha256",
        "manifest_identity",
    }:
        raise ValueError("SIF embedded manifest is not closed")
    if embedded["member_path"] != "/opt/apac/manifest.json":
        raise ValueError("SIF embedded manifest member path differs")
    _uint(embedded["size_bytes"], "embedded manifest size")
    _require_sha256(embedded["sha256"], "embedded manifest sha256")
    _require_sha256(embedded["manifest_identity"], "embedded manifest identity")
    embedded_semantic = {
        "source_commits": value["source_commits"],
        "recipe": value["recipe"],
        "applied_patches": value["applied_patches"],
        "base_oci": {
            name: base[name]
            for name in (
                "reference",
                "manifest_digest",
                "config_digest",
                "layer_digests",
            )
        },
        "lock_manifest": value["lock_manifest"],
    }
    if (
        hashlib.sha256(jcs_bytes(embedded_semantic)).hexdigest()
        != embedded["manifest_identity"]
    ):
        raise ValueError("embedded manifest semantic identity differs")
    runtime_files = value["runtime_files"]
    if not isinstance(runtime_files, list) or [
        row.get("role") for row in runtime_files if isinstance(row, dict)
    ] != [
        "python_executable",
        "sglang_package_entry",
        "cuda_runtime",
        "nccl_runtime",
    ]:
        raise ValueError("SIF runtime file roles differ")
    for row in runtime_files:
        _validate_runtime_file(row)
    nccl = value["nccl_tests"]
    if not isinstance(nccl, dict) or set(nccl) != {
        "commit",
        "mpi_enabled",
        "json_output_enabled",
        "binaries",
        "build_test",
    }:
        raise ValueError("SIF nccl-tests record is not closed")
    if (
        nccl.get("commit") != "717b68318278e93f371d8ffb46b076069d7c7851"
        or nccl.get("mpi_enabled") is not True
        or nccl.get("json_output_enabled") is not True
    ):
        raise ValueError("SIF nccl-tests binding is invalid")
    binaries = nccl["binaries"]
    if not isinstance(binaries, list) or [
        row.get("role") for row in binaries if isinstance(row, dict)
    ] != ["all_reduce_perf", "all_gather_perf", "reduce_scatter_perf"]:
        raise ValueError("SIF nccl-tests binary roles differ")
    for row in binaries:
        _validate_runtime_file(row)
    _validate_process_source(nccl["build_test"], member_root, has_argv=False)
    return value


def _build_observation(
    metadata_root: Path, role: str, argv: list[str]
) -> dict[str, object]:
    prefix = metadata_root / f"sif-build/toolchain/{role}"
    return {
        "role": role,
        "argv": argv,
        "stdout": _member(
            prefix.with_suffix(".stdout"), f"sif-build/toolchain/{role}.stdout"
        ),
        "stderr": _member(
            prefix.with_suffix(".stderr"), f"sif-build/toolchain/{role}.stderr"
        ),
        "outcome": _exit_outcome(prefix.with_suffix(".exit-code")),
    }


def _build_test_observation(metadata_root: Path) -> dict[str, object]:
    root = metadata_root / "sif-build"
    return {
        "outcome": _exit_outcome(root / "build-test.exit-code"),
        "stdout": _member(root / "build-test.stdout", "sif-build/build-test.stdout"),
        "stderr": _member(root / "build-test.stderr", "sif-build/build-test.stderr"),
    }


def _exit_outcome(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="ascii").strip()
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise ValueError(f"invalid exit code source: {path}")
    return {"kind": "exit", "code": int(raw)}


def _validate_observation(value: object, member_root: Path | None) -> None:
    if not isinstance(value, dict) or set(value) != {
        "role",
        "argv",
        "stdout",
        "stderr",
        "outcome",
    }:
        raise ValueError("toolchain observation is not closed")
    _nonempty_string(value["role"], "toolchain role")
    _validate_process_source(value, member_root, has_argv=True, allow_role=True)


def _validate_process_source(
    value: object,
    member_root: Path | None,
    *,
    has_argv: bool,
    allow_role: bool = False,
) -> None:
    expected = {"outcome", "stdout", "stderr"}
    if has_argv:
        expected.add("argv")
    if allow_role:
        expected.add("role")
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("process source is not closed")
    if has_argv:
        argv = value["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
        ):
            raise ValueError("process argv is invalid")
    outcome = value["outcome"]
    if not isinstance(outcome, dict) or set(outcome) != {"kind", "code"}:
        raise ValueError("process outcome is not a closed exit outcome")
    if outcome["kind"] != "exit":
        raise ValueError("build observation did not reach an exit outcome")
    if _uint(outcome["code"], "process exit code") != 0:
        raise ValueError("build observation was not successful")
    for name in ("stdout", "stderr"):
        _validate_member_binding(value[name], member_root)


def _validate_member_binding(value: object, member_root: Path | None) -> str:
    relative = _validate_member(value, require_role=False)
    if member_root is not None:
        candidate = member_root / relative
        if (
            not candidate.is_file()
            or candidate.stat().st_size != value["size_bytes"]
            or _sha256(candidate) != value["sha256"]
        ):
            raise ValueError(f"member hash binding failed: {relative}")
    return relative


def _validate_runtime_file(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "role",
        "realpath",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("SIF runtime file record is not closed")
    _nonempty_string(value["role"], "runtime file role")
    realpath = PurePosixPath(
        _nonempty_string(value["realpath"], "runtime file realpath")
    )
    if not realpath.is_absolute() or ".." in realpath.parts:
        raise ValueError("runtime file realpath is not absolute")
    _uint(value["size_bytes"], "runtime file size")
    _require_sha256(value["sha256"], "runtime file sha256")


def _algorithm_oid(value: object, algorithm: object) -> str:
    if algorithm not in {"sha1", "sha256"}:
        raise ValueError("unknown Git object algorithm")
    expected = 40 if algorithm == "sha1" else 64
    if (
        not isinstance(value, str)
        or re.fullmatch(rf"[0-9a-f]{{{expected}}}", value) is None
    ):
        raise ValueError("algorithm-constrained Git object ID is invalid")
    return value


def _validated_relative(value: object, label: str) -> str:
    result = _nonempty_string(value, label)
    _safe_relative(result)
    return result


def _require_sorted_unique(values: list[str], label: str) -> None:
    if values != sorted(values, key=lambda item: item.encode()) or len(values) != len(
        set(values)
    ):
        raise ValueError(f"{label} must be path-byte sorted and unique")


def _execution_object(settings: ExecutionSettings) -> dict[str, object]:
    return {
        "schema_version": "qwen-execution-settings/v1",
        "run_purpose": settings.run_purpose,
        "nodes": settings.nodes,
        "mpi_processes": settings.mpi_processes,
        "gpus_per_node": settings.gpus_per_node,
        "tensor_parallel_size": settings.tensor_parallel_size,
        "resource_group": settings.resource_group,
        "transport_policy": settings.transport_policy,
        "advertise_source": settings.advertise_source,
        "sglang_port": settings.sglang_port,
        "node_agent_port": settings.node_agent_port,
        "rendezvous_timeout_sec": settings.rendezvous_timeout_sec,
        "ready_timeout_sec": settings.ready_timeout_sec,
    }


def _member(path: Path, relative: str, *, role: str | None = None) -> dict[str, object]:
    _safe_relative(relative)
    result: dict[str, object] = {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if role is not None:
        result["role"] = role
    return result


def _validate_member(
    value: object, *, require_role: bool, path_name: str = "path"
) -> str:
    expected = {path_name, "size_bytes", "sha256"} | (
        {"role"} if require_role else set()
    )
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("member reference is not closed")
    relative = _nonempty_string(value[path_name], "member path")
    _safe_relative(relative)
    _uint(value["size_bytes"], "member size")
    _require_sha256(value["sha256"], "member sha256")
    if require_role:
        _nonempty_string(value["role"], "member role")
    return relative


def _capture_command(argv: list[str], stdout: Path, stderr: Path) -> int:
    with stdout.open("xb") as stdout_stream, stderr.open("xb") as stderr_stream:
        return subprocess.run(
            argv,
            stdout=stdout_stream,
            stderr=stderr_stream,
            check=False,
        ).returncode


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _safe_relative(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"unsafe relative path: {value!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _uint(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_uint(value: object, label: str) -> int:
    result = _uint(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _require_oid(value: object, label: str) -> str:
    if not isinstance(value, str) or _OID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase object ID")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value
