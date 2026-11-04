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
#   40991744289, which adopted it and stopped serving; the fixed row is pinned below.
# - select-firmus-prefill-moe-input.py: fused, 103.99-105.18 us against control 123.57-124.04 us per
#   layer on all four GPUs in both runs; fused-gate stayed below its margin. Superseded by onepass below.
# - select-firmus-decode-dense.py --pinned-env: wide norm/quant (2.51 against 6.05-6.08 us); the
#   SwapAB rows lost in both runs (QKV 25.2-25.6 against 23.0-23.1 us, O 34.6-35.0 against
#   17.4-17.6 us), so the pinned QKV config and the empty SwapAB names stay.
# select-firmus-prefill-attention.py chose the same Prefill Q/K norm writer in both runs that ran it,
# 40911121619 and 40925558339 (image 40911121619), and no longer selects it:
# - qknorm: fp32, 71.37-71.40 / 71.29-71.36 us against control 100.38-100.54 / 100.39-100.46 us per
#   layer on all four GPUs, every gate case passed (V and NaNs bit-equal to control, Q/K one E4M3 step
#   at most from the FP64 reference). Valid only without the FA3 V byte image, which neither
#   direct-o-varlen row uses.
# select-firmus-prefill-attention.py then selected only the one-pass Prefill MoE input over the pinned
# fused path, and was removed after the one run that ran it, 40964714831:
# - moe_input: onepass, 88.89-89.00 us against fused 102.91-102.94 us per layer (24-layer graph at 7988
#   rows) on all four GPUs, every gate case passed (residual bit-equal to fused; FP8 input, scales and
#   top-k within the FP64 reference gate at 7988, 7989, 3994 and 2049 rows). One run suffices here: the
#   13.9-14.0 us (13.6%) saving on every GPU is about ten times the 1-2% noise of these per-layer
#   timings (the twelve rounds of this run spread 88.6-89.4 and 102.4-103.2 us), against a 3 us margin.
# Settled selections from 41005872470, 41020974704 and 41040348302; no in-job selection remains:
# - MegaMoE: four-stage-lean-split won and served in the first two runs and was the adopted winner
#   in the third. Set all three switches: 4STAGE, LEAN and SPLIT_ACCUM. The shared reciprocal was
#   only 0.3% faster (61111 against 61295 us total) in 41040348302 and failed all twelve GPUs'
#   600-call endurance gates on all four routings: identical input and variant, different output bits.
# - Prefill FA3: n192-direct-o-varlen-pipe saved at least 8.1 us, mean 15.0 us per layer call on all
#   twelve GPUs in 41040348302, against a 404.5 us median. Commits d8ba09f and 50ab866 fixed the
#   row that hung in 40991744289 and added its gate. One run suffices: every GPU passed the
#   1200-call endurance gate, then about 1480 PP0 microbatches per worker served without a stall.
#   PP0 periods were 114.6 / 115.0 / 118.3 ms against 115.0 / 115.5 / 118.6 ms in 41020974704.
# - Decode MoE input: fused; QKV >128 rows: 192,80,5,1,0 (keep 128,144,5,1,0 for smaller rows);
#   PP4 proxy: copy. Each run's decode-selection env files and copy-engine selection.json agree.
# - Keep onepass and fp32: onepass-split lost 88.9 -> 90.2 us, fp32-heads4 lost 72.0 -> 81.0 us
#   at 8064 rows, and fp32-heads8 was slower at 4032 rows in 41040348302.
# Round 34, run 41767886175: all twelve GPUs passed the published-result gate,
# all twelve serving processes prewarmed/dispatched first-K, official score 10054.49.
# Retire the repeated FC1 gate; retain the process-local preparation and dispatch log.
# Round 38, run 41811427804: dense first-K passed all twelve gates and served at
# 10059.13; retain production prewarm/dispatch and retire its repeated selection.
# Round 39, run 41822975123: B/C passed the r37 gate on all twelve GPUs,
# score 10058.37; keep their per-process prewarm and post-aiperf dispatch logs.
PP4_SELECTED = {
    "SGLANG_PREFILL_FA3_VERIFY": "n192-direct-o-varlen-pipe",
    "SGLANG_PREFILL_MOE_INPUT": "onepass",
    "SGLANG_PREFILL_QKNORM": "fp32",
    "SGLANG_DECODE_NORM_QUANT": "wide",
    "DG_SM90_PREFILL_4STAGE": "1",
    "DG_SM90_PREFILL_LEAN": "1",
    "DG_SM90_PREFILL_SPLIT_ACCUM": "1",
    "DG_SM90_PREFILL_FC1_FIRST_K": "1",
    "DG_SM90_PREFILL_DENSE_FIRST_K": "1",
    "SGLANG_PREFILL_QKNORM_FIXED": "1",
    "SGLANG_PREFILL_ONEPASS_DIV448": "1",
    "SGLANG_PP4_PROXY_TRANSPORT": "copy",
    "SGLANG_DECODE_MOE_INPUT": "fused",
    "DG_SM90_1D2D_DECODE_CONFIG": "QKV:128,144,5,1,0;QKV>128:192,80,5,1,0",
}
# The runs in which each pinned value won.
PP4_SELECTED_RUNS = {
    "SGLANG_PREFILL_FA3_VERIFY": [41040348302],
    "SGLANG_PREFILL_MOE_INPUT": [40964714831],
    "SGLANG_PREFILL_QKNORM": [40911121619, 40925558339],
    "SGLANG_DECODE_NORM_QUANT": [40849459077, 40883389891],
    "DG_SM90_PREFILL_4STAGE": [41005872470, 41020974704, 41040348302],
    "DG_SM90_PREFILL_LEAN": [41005872470, 41020974704, 41040348302],
    "DG_SM90_PREFILL_SPLIT_ACCUM": [41005872470, 41020974704, 41040348302],
    "DG_SM90_PREFILL_FC1_FIRST_K": [41767886175],
    "DG_SM90_PREFILL_DENSE_FIRST_K": [41811427804],
    "SGLANG_PREFILL_QKNORM_FIXED": [41822975123],
    "SGLANG_PREFILL_ONEPASS_DIV448": [41822975123],
    "SGLANG_PP4_PROXY_TRANSPORT": [41005872470, 41020974704, 41040348302],
    "SGLANG_DECODE_MOE_INPUT": [41005872470, 41020974704, 41040348302],
    "DG_SM90_1D2D_DECODE_CONFIG": [41005872470, 41020974704, 41040348302],
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
    environment = {**record["environment"], **PP4_SELECTED, "DG_SM90_PREFILL_DENSE_N64": "0", "SGLANG_DECODE_GREEDY_VERIFY": "0"}
    for role in ("prefill", "decode"):
        names = {
            k: v
            for k, v in environment.items()
            if k.startswith(("SGLANG_PREFILL_", "DG_SM90_PREFILL_", "SGLANG_PP4_")) == (role == "prefill")
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
        "no_gpu_selection_rerun": False,  # The bounded dense r44 gate runs before Prefill.
        "scope": "Prior settings plus the settled PP4 selections evidenced by pp4_selected_runs, split explicitly between Prefill and Decode. The optional fa3_pair_gate may replace the Prefill FA3 pin after all twelve GPUs pass; results/prefill-fa3-pair/selection.json records that decision. The dense r44 gate starts from DG_SM90_PREFILL_DENSE_N64=0 and may update only the Prefill env after all twelve GPUs pass; results/prefill-dense-r44/summary.json records its decision. Ordinary serving validates the complete image, EP1, PP scheduling, batching and transfers.",
    }
    # Round-17 P followed by optional P3; P3 remains off until a serving commit.
    payload["fa3_prepack_gate"] = p.get("fa3_prepack_gate", False)
    payload["fa3_v3_gate"] = p.get("fa3_v3_gate", False)
    assert not payload["fa3_v3_gate"] or payload["fa3_prepack_gate"], "V3 gate requires prepack gate"
    payload["decode_greedy_verify_gate"] = config.get("decode_greedy_verify_gate", False)
    if payload["fa3_prepack_gate"] or payload["decode_greedy_verify_gate"]:
        payload["no_gpu_selection_rerun"] = False

    for directory in ("source", "results"):
        (runtime / directory / "pp4-kernel-settings.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )


if __name__ == "__main__":
    config, repo, runtime = map(Path, sys.argv[1:])
    prepare(json.loads(config.read_text()), repo, runtime)
