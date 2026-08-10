from __future__ import annotations

import hashlib
import os
import re
import socket
import tempfile
from pathlib import Path

from .canonical import jcs_bytes, load_json_strict
from .manifest import load_manifest

_NUMA_NODE = re.compile(r"^node([0-9]+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROLE_ORDER = {"proc_meminfo": 0, "numa_cpulist": 1, "numa_meminfo": 2}


def capture_profile_hardware(
    manifest_path: Path,
    runtime_dir: Path,
    *,
    node_rank: int,
    proc_root: Path = Path("/proc"),
    sys_node_root: Path = Path("/sys/devices/system/node"),
    hostname: str | None = None,
) -> Path:
    """Capture the fixed post-run profile-only CPU/NUMA source family."""

    manifest = load_manifest(manifest_path)
    if manifest.run_purpose != "profile":
        raise ValueError("profile hardware capture is profile-only")
    if isinstance(node_rank, bool) or not isinstance(node_rank, int) or node_rank < 0:
        raise ValueError("profile hardware node rank must be a nonnegative integer")
    destination = runtime_dir / f"profile/raw/hardware/node-{node_rank}"
    destination.mkdir(parents=True, exist_ok=False)
    members: list[dict[str, object]] = []

    def capture(role: str, numa_node_id: int | None, source: Path, relative: str) -> None:
        payload = source.read_bytes()
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _exclusive_bytes(path, payload)
        member = path.relative_to(runtime_dir).as_posix()
        members.append(
            {
                "role": role,
                "numa_node_id": numa_node_id,
                "path": member,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    capture("proc_meminfo", None, proc_root / "meminfo", "proc-meminfo.txt")
    node_paths: list[tuple[int, Path]] = []
    for path in sys_node_root.iterdir():
        match = _NUMA_NODE.fullmatch(path.name)
        if match is not None and path.is_dir():
            node_paths.append((int(match.group(1)), path))
    for numa_node_id, path in sorted(node_paths):
        capture(
            "numa_cpulist",
            numa_node_id,
            path / "cpulist",
            f"numa/node-{numa_node_id}-cpulist.txt",
        )
        capture(
            "numa_meminfo",
            numa_node_id,
            path / "meminfo",
            f"numa/node-{numa_node_id}-meminfo.txt",
        )
    members.sort(
        key=lambda item: (
            _ROLE_ORDER[str(item["role"])],
            -1 if item["numa_node_id"] is None else int(item["numa_node_id"]),
            str(item["path"]).encode("utf-8"),
        )
    )
    capture_path = destination / "capture.json"
    _exclusive_bytes(
        capture_path,
        jcs_bytes(
            {
                "schema_version": "profile-hardware-capture/v1",
                "run_id": manifest.run_id,
                "node_rank": node_rank,
                "hostname": hostname or socket.gethostname(),
                "members": members,
            }
        )
        + b"\n",
    )
    validate_profile_hardware_capture(
        capture_path, runtime_dir, manifest.run_id, node_rank
    )
    return capture_path


def validate_profile_hardware_capture(
    path: Path, runtime_dir: Path, run_id: str, node_rank: int
) -> dict[str, object]:
    value = load_json_strict(path.read_bytes())
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "run_id",
        "node_rank",
        "hostname",
        "members",
    }:
        raise ValueError("profile hardware capture is not closed")
    if (
        value["schema_version"] != "profile-hardware-capture/v1"
        or value["run_id"] != run_id
        or value["node_rank"] != node_rank
        or not isinstance(value["hostname"], str)
        or not value["hostname"]
    ):
        raise ValueError("profile hardware capture identity differs")
    members = value["members"]
    if not isinstance(members, list) or not members:
        raise ValueError("profile hardware members must be a nonempty array")
    identities: list[tuple[str, int | None, str]] = []
    roles: dict[tuple[str, int | None], dict[str, object]] = {}
    for member in members:
        if not isinstance(member, dict) or set(member) != {
            "role",
            "numa_node_id",
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("profile hardware member is not closed")
        role = member["role"]
        numa_node_id = member["numa_node_id"]
        if role not in _ROLE_ORDER:
            raise ValueError("profile hardware role is invalid")
        if role == "proc_meminfo":
            if numa_node_id is not None:
                raise ValueError("proc meminfo must not have a NUMA node")
        elif (
            isinstance(numa_node_id, bool)
            or not isinstance(numa_node_id, int)
            or numa_node_id < 0
        ):
            raise ValueError("NUMA profile hardware identity is invalid")
        relative = member["path"]
        if not isinstance(relative, str):
            raise TypeError("profile hardware member path must be a string")
        expected_prefix = f"profile/raw/hardware/node-{node_rank}/"
        if not relative.startswith(expected_prefix) or ".." in Path(relative).parts:
            raise ValueError("profile hardware member path differs")
        size_bytes = member["size_bytes"]
        digest = member["sha256"]
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError("profile hardware member lineage is invalid")
        candidate = runtime_dir / relative
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or candidate.stat().st_size != size_bytes
            or _sha256(candidate) != digest
        ):
            raise ValueError("profile hardware member binding differs")
        identity = (role, numa_node_id, relative)
        identities.append(identity)
        key = (role, numa_node_id)
        if key in roles:
            raise ValueError("profile hardware member identity is duplicated")
        roles[key] = member
    expected_order = sorted(
        identities,
        key=lambda item: (
            _ROLE_ORDER[item[0]],
            -1 if item[1] is None else item[1],
            item[2].encode("utf-8"),
        ),
    )
    if identities != expected_order or ("proc_meminfo", None) not in roles:
        raise ValueError("profile hardware members are incomplete or unordered")
    numa_ids = {node for role, node in roles if role != "proc_meminfo"}
    if any(
        ("numa_cpulist", node) not in roles or ("numa_meminfo", node) not in roles
        for node in numa_ids
    ):
        raise ValueError("profile hardware NUMA pairs are incomplete")
    return value


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    try:
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
