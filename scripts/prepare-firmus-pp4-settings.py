#!/usr/bin/env python3
"""Pin prior kernel settings and the settled PP4 selections for one PP4 serving trial."""

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
#   The PP4 job screened n192-direct-o-varlen-pipe (pipelined M-tiles) against this pin until
#   40991744289, which adopted it and stopped serving; the row stays in the image and candidates file.
# - select-firmus-prefill-moe-input.py: fused, 103.99-105.18 us against control 123.57-124.04 us per
#   layer on all four GPUs in both runs; fused-gate stayed below its margin. Superseded by onepass below.
# - select-firmus-decode-dense.py --pinned-env: wide norm/quant (2.51 against 6.05-6.08 us); the
#   SwapAB rows lost in both runs (QKV 25.2-25.6 against 23.0-23.1 us, O 34.6-35.0 against
#   17.4-17.6 us), so the pinned QKV config and the empty SwapAB names stay.
# select-firmus-prefill-attention.py chose the same Prefill Q/K norm writer in both runs that ran it,
# 40911121619 and 40925558339 (image 40911121619), and no longer selects it:
# - qknorm: fp32, 71.37-71.40 / 71.29-71.36 us against control 100.38-100.54 / 100.39-100.46 us per
#   layer on all four GPUs, every gate case passed (V and NaNs bit-equal to control, Q/K one E4M3 step
#   at most from the FP64 reference). Valid only without the FA3 V byte image, which the pinned
#   n192-direct-o-varlen row does not use.
# select-firmus-prefill-attention.py then selected only the one-pass Prefill MoE input over the pinned
# fused path, and was removed after the one run that ran it, 40964714831:
# - moe_input: onepass, 88.89-89.00 us against fused 102.91-102.94 us per layer (24-layer graph at 7988
#   rows) on all four GPUs, every gate case passed (residual bit-equal to fused; FP8 input, scales and
#   top-k within the FP64 reference gate at 7988, 7989, 3994 and 2049 rows). One run suffices here: the
#   13.9-14.0 us (13.6%) saving on every GPU is about ten times the 1-2% noise of these per-layer
#   timings (the twelve rounds of this run spread 88.6-89.4 and 102.4-103.2 us), against a 3 us margin.
PP4_SELECTED = {
    "SGLANG_PREFILL_FA3_VERIFY": "n192-direct-o-varlen",
    "SGLANG_PREFILL_MOE_INPUT": "onepass",
    "SGLANG_PREFILL_QKNORM": "fp32",
    "SGLANG_DECODE_NORM_QUANT": "wide",
}
# The runs in which each pinned value won.
PP4_SELECTED_RUNS = {
    "SGLANG_PREFILL_FA3_VERIFY": [40849459077, 40883389891],
    "SGLANG_PREFILL_MOE_INPUT": [40964714831],
    "SGLANG_PREFILL_QKNORM": [40911121619, 40925558339],
    "SGLANG_DECODE_NORM_QUANT": [40849459077, 40883389891],
}


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
        "scope": "Prior settings plus the PP4 selections that won in the PP4 runs listed in pp4_selected_runs. New image, EP1, PP scheduling, batching and transfers require the ordinary serving trial.",
    }
    for directory in ("source", "results"):
        (runtime / directory / "pp4-kernel-settings.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )


if __name__ == "__main__":
    config, repo, runtime = map(Path, sys.argv[1:])
    prepare(json.loads(config.read_text()), repo, runtime)
