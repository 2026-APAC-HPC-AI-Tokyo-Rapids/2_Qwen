#!/usr/bin/env python3
"""Pin prior kernel settings and the repeated PP4 selections for one PP4 serving trial."""

import hashlib
import json
import runpy
import shlex
import sys
from pathlib import Path

# PP4 in-job selections that chose the same winner in both runs that ran them, 40849459077 and
# 40883389891 (image 40849459077), and no longer run in the PP4 job (the scripts stay):
# - select-firmus-prefill-fa3.py screen of n192-varlen and n192-direct-o-varlen on all twelve Prefill
#   GPUs: n192-direct-o-varlen, the only eligible row in both (minimum saving 11.7 / 13.6 us, median
#   417.1 / 416.7 us per call against n192-varlen 423.9 / 423.8 us, which missed the margin).
# - select-firmus-prefill-moe-input.py: fused, 103.99-105.18 us against control 123.57-124.04 us per
#   layer on all four GPUs in both runs; fused-gate stayed below its margin.
# - select-firmus-decode-dense.py --pinned-env: wide norm/quant (2.51 against 6.05-6.08 us); the
#   SwapAB rows lost in both runs (QKV 25.2-25.6 against 23.0-23.1 us, O 34.6-35.0 against
#   17.4-17.6 us), so the pinned QKV config and the empty SwapAB names stay.
PP4_SELECTED = {
    "SGLANG_PREFILL_FA3_VERIFY": "n192-direct-o-varlen",
    "SGLANG_PREFILL_MOE_INPUT": "fused",
    "SGLANG_DECODE_NORM_QUANT": "wide",
}
PP4_SELECTED_RUNS = [40849459077, 40883389891]


def prepare(config, repo, runtime):
    p = config["prefill"]
    expected = {
        "pp": 4,
        "tp": 1,
        "dp": 1,
        "ep": 1,
        "pp_max_micro_batch_size": 2,
        "pp_async_batch_depth": 1,
        "pp_layer_partition": [24, 24, 24, 22],
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
    environment = {**record["environment"], **PP4_SELECTED}
    for role in ("prefill", "decode"):
        names = {
            k: v
            for k, v in environment.items()
            if k.startswith("SGLANG_PREFILL_") == (role == "prefill")
        }
        (runtime / "source" / f"{role}-kernels.env").write_text(
            "".join(f"{k}={shlex.quote(v)}\n" for k, v in names.items())
        )
    payload = {
        "schema": "pp4-kernel-settings/v1",
        "reference": reference,
        "reference_sha256": hashlib.sha256((repo / reference).read_bytes()).hexdigest(),
        "environment": environment,
        "pp4_selected": PP4_SELECTED,
        "pp4_selected_runs": PP4_SELECTED_RUNS,
        "no_gpu_selection_rerun": True,
        "scope": "Prior settings plus the PP4 selections that won in both PP4 runs. New image, EP1, PP scheduling, batching and transfers require the ordinary serving trial.",
    }
    for directory in ("source", "results"):
        (runtime / directory / "pp4-kernel-settings.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )


if __name__ == "__main__":
    config, repo, runtime = map(Path, sys.argv[1:])
    prepare(json.loads(config.read_text()), repo, runtime)
