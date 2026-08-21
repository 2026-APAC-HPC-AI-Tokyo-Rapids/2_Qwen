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
    for role in ("mpirun", "enroot"):
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
        version_argv = (
            [requested, "version"] if role == "enroot" else [requested, "--version"]
        )
        outcome = _capture_command(version_argv, stdout, stderr)
        versions.append(
            {
                "role": role,
                "argv": version_argv,
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
        "enroot",
    ]:
        raise ValueError("runtime launcher executable roles differ")
    if [row.get("role") for row in versions if isinstance(row, dict)] != [
        "mpirun",
        "enroot",
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
        expected_argv = (
            [requested_by_role[row["role"]], "version"]
            if row["role"] == "enroot"
            else [requested_by_role[row["role"]], "--version"]
        )
        if row["argv"] != expected_argv:
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
    sidecar = load_json_strict(
        (runtime_dir / "source/container-sidecar.json").read_bytes()
    )
    if not isinstance(sidecar, dict) or set(sidecar) != {
        "schema_version",
        "container_artifact",
        "oci_manifest",
        "source_commits",
        "toolchain",
    }:
        raise ValueError("container sidecar is not closed")
    if sidecar["schema_version"] != "container-sidecar/v1":
        raise ValueError("unsupported container sidecar")
    artifact = sidecar["container_artifact"]
    if not isinstance(artifact, dict) or set(artifact) != {
        "artifact_relative_path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("container artifact record is not closed")
    if artifact["artifact_relative_path"] != "sglang.sqsh":
        raise ValueError("container artifact path differs")
    image = runtime_dir / artifact["artifact_relative_path"]
    if (
        not image.is_file()
        or image.stat().st_size != _uint(artifact["size_bytes"], "container size")
        or _sha256(image) != _require_sha256(artifact["sha256"], "container sha256")
    ):
        raise ValueError("provisioned container does not match its sidecar")
    oci_manifest = sidecar["oci_manifest"]
    if not isinstance(oci_manifest, dict) or set(oci_manifest) != {"sha256"}:
        raise ValueError("OCI manifest record is not closed")
    _require_sha256(oci_manifest["sha256"], "OCI manifest sha256")
    commits = sidecar["source_commits"]
    if not isinstance(commits, dict) or set(commits) != {"qwen", "sglang"}:
        raise ValueError("container source commits are not closed")
    qwen_commit = _algorithm_oid(commits["qwen"], "sha1")
    sglang_commit = _algorithm_oid(commits["sglang"], "sha1")
    toolchain = sidecar["toolchain"]
    if not isinstance(toolchain, dict) or set(toolchain) != {
        "buildah",
        "skopeo",
        "umoci",
    }:
        raise ValueError("container toolchain is not closed")
    for name, version in toolchain.items():
        _nonempty_string(version, f"{name} version")
    model = load_json_strict((runtime_dir / "source/model-metadata.json").read_bytes())
    if not isinstance(model, dict) or set(model) != {
        "schema_version",
        "repository",
        "revision",
        "path",
        "weights_present",
        "load_format",
    }:
        raise ValueError("dummy model metadata is not closed")
    if (
        model["schema_version"] != "model-metadata/v1"
        or model["weights_present"] is not False
        or model["load_format"] != "dummy"
    ):
        raise ValueError("dummy model metadata differs")
    _nonempty_string(model["repository"], "model repository")
    _algorithm_oid(model["revision"], "sha1")
    _nonempty_string(model["path"], "model path")
    dataset = load_json_strict(
        (runtime_dir / "source/dataset-manifest.json").read_bytes()
    )
    if not isinstance(dataset, dict) or set(dataset) != {
        "schema_version",
        "dataset_name",
        "seed",
        "num_prompts",
        "input_length",
        "output_length",
        "range_ratio",
    }:
        raise ValueError("dummy dataset manifest is not closed")
    if (
        dataset["schema_version"] != "dummy-random-dataset/v1"
        or dataset["dataset_name"] != "random-ids"
    ):
        raise ValueError("dummy dataset manifest differs")
    for name in (
        "seed",
        "num_prompts",
        "input_length",
        "output_length",
        "range_ratio",
    ):
        _uint(dataset[name], f"dataset {name}")
    return (
        Fact(
            "build.provenance",
            {
                "container": artifact,
                "oci_manifest": oci_manifest,
                "toolchain": toolchain,
            },
        ),
        Fact("source.container_build.qwen_commit", qwen_commit),
        Fact("source.sglang.commit", sglang_commit),
        Fact("input.model", model),
        Fact("input.dataset", dataset),
    )


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
