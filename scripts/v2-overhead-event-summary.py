#!/usr/bin/env python3
"""Summarize the validated event archive produced by the real v2 recorder."""

import json
import os
import sys
from collections import Counter
from pathlib import Path


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 4:
        raise ValueError("usage: event-summary ARCHIVE SUMMARY RUN_ID NODE_RANK")
    archive_path = Path(arguments[0])
    summary_path = Path(arguments[1])
    run_id = arguments[2]
    node_rank = int(arguments[3])
    event_types = Counter()
    producer_ids = set()
    payload_bytes = 0
    with archive_path.open("rb") as stream:
        for line in stream:
            value = json.loads(line)
            if value["run_id"] != run_id or value["node_rank"] != node_rank:
                raise ValueError("archived event identity differs")
            event_types[value["event_type"]] += 1
            producer_ids.add(value["producer_id"])
            payload_bytes += len(line.rstrip(b"\n"))
    result = {
        "schema_version": "overhead-event-summary/v2",
        "run_id": run_id,
        "node_rank": node_rank,
        "total_events": sum(event_types.values()),
        "total_payload_bytes": payload_bytes,
        "event_type_counts": dict(sorted(event_types.items())),
        "producer_ids": sorted(producer_ids),
    }
    temporary = summary_path.with_name(f".{summary_path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    os.replace(temporary, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
