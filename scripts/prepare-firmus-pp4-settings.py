#!/usr/bin/env python3
"""Pin prior kernel settings for one PP4 serving trial without claiming compatibility."""

import hashlib
import json
import runpy
import shlex
import sys
from pathlib import Path


def prepare(config, repo, runtime):
    p = config["prefill"]
    expected = {
        "pp": 4,
        "tp": 1,
        "dp": 1,
        "ep": 1,
        "pp_max_micro_batch_size": 2,
        "chunked_prefill_size": 8192,
        "expert_location": "trivial",
        "ep_num_redundant_experts": 0,
        "optimistic_prefill_attempts": 0,
    }
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported PP4 trial layout")
    reference = config["kernel_settings_reference"]
    if reference != "configs/firmus-kernel-selection-40313004834.json":
        raise ValueError("Unexpected kernel settings reference")
    record = json.loads((repo / reference).read_text())
    previous = runpy.run_path(str(repo / "scripts/reuse-firmus-kernel-selection.py"))
    if record["environment"] != previous["ENVIRONMENT"]:
        raise ValueError("Prior kernel settings changed")
    # Only the previously measured sources are asserted here. The PP extension
    # changes host dispatch and topology and is explicitly NOT a prior result.
    for entry in record["patches"] + record["extra_files"]:
        previous["file_sha"](repo, entry)
    for role in ("prefill", "decode"):
        names = {
            k: v
            for k, v in record["environment"].items()
            if (k == "SGLANG_PREFILL_FA3_VERIFY") == (role == "prefill")
        }
        (runtime / "source" / f"{role}-kernels.env").write_text(
            "".join(f"{k}={shlex.quote(v)}\n" for k, v in names.items())
        )
    payload = {
        "schema": "pp4-kernel-settings/v1",
        "reference": reference,
        "reference_sha256": hashlib.sha256((repo / reference).read_bytes()).hexdigest(),
        "environment": record["environment"],
        "no_gpu_selection_rerun": True,
        "scope": "Prior settings only. New image, EP1, PP scheduling, batching and transfers require the ordinary serving trial; no prior PP4 numeric/performance eligibility.",
    }
    for directory in ("source", "results"):
        (runtime / directory / "pp4-kernel-settings.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )


if __name__ == "__main__":
    config, repo, runtime = map(Path, sys.argv[1:])
    prepare(json.loads(config.read_text()), repo, runtime)
