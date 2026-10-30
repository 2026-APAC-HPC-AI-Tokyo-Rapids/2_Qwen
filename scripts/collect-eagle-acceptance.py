#!/usr/bin/env python3
"""Collect EAGLE acceptance statistics and the effective PD route."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ACCEPTANCE_LOG = re.compile(
    r"Decode batch.*?accept len:\s*(?P<length>[0-9]+(?:\.[0-9]+)?)"
    r".*?accept rate:\s*(?P<rate>[0-9]+(?:\.[0-9]+)?)"
)


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def read_endpoints(path: Path) -> list[dict[str, object]]:
    endpoints: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(
                f"{path}:{line_number}: expected four tab-separated fields"
            )
        role, host, port, auxiliary = fields
        if not port.isdigit():
            raise ValueError(f"{path}:{line_number}: invalid port {port!r}")
        endpoints.append(
            {
                "role": role,
                "stage": role.split("-", 1)[0],
                "host": host,
                "port": int(port),
                "auxiliary": auxiliary,
            }
        )
    decode = [endpoint for endpoint in endpoints if endpoint["stage"] == "decode"]
    if not decode:
        raise ValueError(f"{path}: no decode endpoint")
    return endpoints


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def speculation_config(path: Path) -> dict[str, object]:
    config = read_json(path)
    speculation = config.get("speculation")
    if not isinstance(speculation, dict) or speculation.get("algorithm") != "EAGLE3":
        raise ValueError(f"{path}: EAGLE3 speculation configuration is required")
    tokenizer_backend = config.get("tokenizer_backend", "huggingface")
    if not isinstance(tokenizer_backend, str) or not tokenizer_backend:
        raise ValueError(f"{path}: tokenizer_backend must be a non-empty string")
    return {
        "algorithm": speculation["algorithm"],
        "steps": speculation.get("steps"),
        "draft_tokens": speculation.get("draft_tokens"),
        "draft_quantization": speculation.get("draft_quantization"),
        "draft_repository": speculation.get("draft_repository"),
        "draft_revision": speculation.get("draft_revision"),
        "eagle_topk": 1,
        "tokenizer_backend": tokenizer_backend,
    }


def server_info_path(directory: Path, endpoint: dict[str, object]) -> Path:
    role = str(endpoint["role"])
    port = int(endpoint["port"])
    candidates = [
        directory / f"server-info-{role}-{port}.json",
        directory / f"server-info-{role}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(directory.glob(f"server-info-{role}-*.json"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"no server-info JSON for decode endpoint {role}:{port}")


def acceptance_states(info: dict[str, Any], path: Path, tokenizer_backend: str) -> list[dict[str, object]]:
    if info.get("speculative_algorithm") not in (None, "EAGLE3"):
        raise ValueError(f"{path}: server is not using EAGLE3")
    if tokenizer_backend != "huggingface":
        states = info.get("internal_states")
        observed = [info, *(states if isinstance(states, list) else [])]
        if any(state.get("tokenizer_backend") != tokenizer_backend for state in observed if isinstance(state, dict)):
            raise ValueError(f"{path}: server tokenizer_backend is not {tokenizer_backend}")
    states = info.get("internal_states")
    if not isinstance(states, list):
        raise TypeError(f"{path}: internal_states is missing")
    result: list[dict[str, object]] = []
    for index, state in enumerate(states):
        if not isinstance(state, dict):
            continue
        value = state.get("avg_spec_accept_length")
        if not finite_number(value):
            continue
        accepted_length = float(value)
        result.append(
            {
                "dp_state": index,
                "accept_length_including_bonus": accepted_length,
                "accepted_draft_length": accepted_length - 1.0,
            }
        )
    if not result:
        raise ValueError(f"{path}: no completed EAGLE acceptance state")
    return result


def log_samples(directory: Path) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for path in sorted(directory.glob("decode-*.log")):
        text = path.read_text(encoding="utf-8", errors="replace").replace("\x1b", "")
        for match in ACCEPTANCE_LOG.finditer(text):
            accepted_length = float(match.group("length"))
            samples.append(
                {
                    "log": path.name,
                    "accept_length_including_bonus": accepted_length,
                    "accepted_draft_length": accepted_length - 1.0,
                    "accept_rate": float(match.group("rate")),
                }
            )
    return samples


def collect(arguments: argparse.Namespace) -> dict[str, object]:
    configured = speculation_config(arguments.config)
    endpoints = read_endpoints(arguments.endpoints)
    workers: list[dict[str, object]] = []
    state_lengths: list[float] = []
    for endpoint in endpoints:
        if endpoint["stage"] != "decode":
            continue
        info_path = server_info_path(arguments.server_info_dir, endpoint)
        info = read_json(info_path)
        if configured["draft_quantization"] is not None and info.get("speculative_draft_model_quantization") != configured["draft_quantization"]:
            raise ValueError(f"{info_path}: requested draft quantization was not applied")
        states = acceptance_states(info, info_path, str(configured["tokenizer_backend"]))
        state_lengths.extend(
            float(state["accept_length_including_bonus"]) for state in states
        )
        workers.append(
            {
                "role": endpoint["role"],
                "host": endpoint["host"],
                "port": endpoint["port"],
                "server_info": info_path.name,
                "tokenizer_backend": info.get("tokenizer_backend"),
                "draft_quantization": info.get("speculative_draft_model_quantization"),
                "states": states,
            }
        )
    if not workers:
        raise ValueError("no decode worker acceptance state was collected")

    samples = log_samples(arguments.log_dir)
    sample_lengths = [
        float(sample["accept_length_including_bonus"]) for sample in samples
    ]
    sample_draft_lengths = [
        float(sample["accepted_draft_length"]) for sample in samples
    ]
    sample_rates = [float(sample["accept_rate"]) for sample in samples]
    scheduler = {
        "genkai": "PJM",
        "aspire2ap": "PBS",
        "firmus": "Slurm",
    }.get(arguments.cluster, "unknown")
    batch_entrypoint = {
        "genkai": "platforms/genkai/run-qwen-pd.pjm",
        "aspire2ap": "platforms/aspire2ap/run-qwen-pd.pbs",
        "firmus": "platforms/firmus/framework-comparison/sglang.sbatch",
    }.get(arguments.cluster, "unknown")
    if arguments.cluster == "firmus" and read_json(arguments.config).get("tier") == "basic":
        batch_entrypoint = "submission/basic/submit.sbatch"
    return {
        "schema_version": "qwen3-eagle-acceptance/v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "cluster": arguments.cluster,
        "method": {
            "primary": "GET /get_server_info internal_states[].avg_spec_accept_length",
            "log_sample": "Decode batch accept len and accept rate",
            "accept_length_definition": "accepted draft tokens plus the bonus token per verification forward",
            "accepted_draft_length_definition": "accept_length_including_bonus minus one bonus token",
            "state_aggregate": "unweighted mean of completed decode DP-state averages",
        },
        "speculation": configured,
        "route": {
            "submission": {
                "control_plane": "GitHub Actions",
                "runner": "self-hosted/home-lab",
                "scheduler": scheduler,
                "batch_entrypoint": batch_entrypoint,
            },
            "client": "AIPerf",
            "gateway": "http://127.0.0.1:8000/v1/chat/completions",
            "stages": [
                "AIPerf",
                "sglang-router",
                "Prefill workers",
                "NIXL KV transfer",
                "Decode workers",
                "EAGLE3 verification",
            ],
            "worker_endpoints": endpoints,
        },
        "decode_workers": workers,
        "statistics": {
            "server_info_state_count": len(state_lengths),
            "server_info_accept_length_including_bonus": summary(state_lengths),
            "server_info_accepted_draft_length": summary(
                [value - 1.0 for value in state_lengths]
            ),
            "decode_log_samples": {
                "accept_length_including_bonus": summary(sample_lengths),
                "accepted_draft_length": summary(sample_draft_lengths),
                "accept_rate": summary(sample_rates),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--endpoints", type=Path, required=True)
    parser.add_argument("--server-info-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cluster", default="unknown")
    arguments = parser.parse_args(argv)
    try:
        result = collect(arguments)
        arguments.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"eagle acceptance collection failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result["statistics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
