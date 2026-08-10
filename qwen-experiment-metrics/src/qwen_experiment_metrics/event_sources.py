from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .canonical import jcs_bytes, load_json_strict
from .events import DomainEvent, facts_from_event, parse_domain_event


@dataclass(frozen=True, slots=True)
class ArchivedDomainEvent:
    event: DomainEvent
    value: dict[str, object]
    source_member: str
    source_sha256: str


def validate_event_archive(
    path: Path,
    runtime_dir: Path,
    *,
    run_id: str,
    node_rank: int,
) -> tuple[ArchivedDomainEvent, ...]:
    """Validate one node-local canonical domain-event source archive."""

    root = runtime_dir.resolve()
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("event archive escapes the runtime directory")
    relative = path.resolve().relative_to(root).as_posix()
    if relative != f"source/events/node-{node_rank}.jsonl":
        raise ValueError("event archive does not use its fixed node member")
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("event archive must be nonempty and LF terminated")
    digest = hashlib.sha256(raw).hexdigest()
    expected_sequence: dict[str, int] = {}
    result: list[ArchivedDomainEvent] = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        try:
            value = load_json_strict(line)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"event archive line {line_number} is invalid") from error
        if not isinstance(value, dict) or line != jcs_bytes(value) + b"\n":
            raise ValueError(f"event archive line {line_number} is not canonical JCS")
        event = parse_domain_event(value, expected_run_id=run_id)
        facts_from_event(value, expected_run_id=run_id)
        if event.node_rank != node_rank:
            raise ValueError(f"event archive line {line_number} has a foreign node")
        sequence = expected_sequence.get(event.producer_id, 0)
        if event.sequence != sequence:
            raise ValueError(f"event archive line {line_number} has a sequence gap")
        expected_sequence[event.producer_id] = sequence + 1
        result.append(ArchivedDomainEvent(event, value, relative, digest))
    return tuple(result)
