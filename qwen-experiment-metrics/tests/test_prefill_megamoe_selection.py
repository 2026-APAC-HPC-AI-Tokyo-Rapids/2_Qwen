import json
import re
import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def selector():
    return runpy.run_path(str(ROOT / "scripts/select-firmus-prefill-megamoe.py"))


# The split defaults (2790, 2795 us) are below the 0.5% margin, so they never win unless a test lowers them.
# The shared reciprocal is not bit-equal; its distance to the control is the CPU model's (within the bounds).
def write_ranks(selector, directory, workers=3, control=2800.0, four=2740.0, lean=2720.0, pair=2700.0, split=2790.0, rcp=2795.0):
    times = {"four-stage": four, "four-stage-lean": lean, "four-stage-lean-pair": pair, "four-stage-lean-split": split, "four-stage-lean-split-rcp": rcp}
    base = {"finite": True, "relative_l2": 0.012, "max_scaled_error": 0.02, "reference_passed": True}
    close = {"bitwise_equal": False, "control_relative_l2": 9.8e-5, "control_max_scaled_error": 4.7e-3}
    match = {name: close if name in selector["TOLERANCE_GATED"] else {"bitwise_equal": True} for name in times}
    for worker in range(workers):
        path = directory / f"prefill-{worker}"
        path.mkdir(parents=True)
        for rank in range(4):
            routings = {}
            for kind in selector["ROUTINGS"]:
                routings[kind] = {"control": {**base, "us": [control] * selector["ROUNDS"]}, **{name: {**base, **match[name], "us": [us] * selector["ROUNDS"]} for name, us in times.items()}}
            for kind in selector["EXACT_ROUTINGS"]:
                routings[kind] = {"control": {**base, "us": []}, **{name: {**base, **match[name], "us": []} for name in times}}
            (path / f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "ranks_per_group": 1, "tokens": 8064, "routings": routings}))


def edit(directory, worker, rank, kind, variant, **changes):
    path = directory / f"prefill-{worker}" / f"rank-{rank}.json"
    entry = json.loads(path.read_text())
    entry["routings"][kind][variant].update(changes)
    path.write_text(json.dumps(entry))


def write_gate(selector, directory, variant, workers=3, **changes):
    for worker in range(workers):
        path = directory / f"prefill-{worker}"
        path.mkdir(parents=True, exist_ok=True)
        for rank in range(4):
            entry = {"rank": rank, "variant": variant, "calls": selector["ENDURANCE_CALLS"], "needed": True, "reference": variant, "bitwise_equal": {k: True for k in ("balanced", "hot", "edges", "wide")}, "passed": True, **changes}
            (path / f"rank-{rank}.json").write_text(json.dumps(entry))


def aggregate(selector, tmp_path):
    out, env = tmp_path / "selection.json", tmp_path / "prefill-kernels.env"
    env.write_text("SGLANG_PREFILL_FA3_VERIFY=n192\n")
    selector["aggregate"](tmp_path / "results", out, env, 3, tmp_path / "endurance")
    return json.loads(out.read_text()), env.read_text()


def test_adopts_the_fastest_variant_that_passed_everywhere(selector, tmp_path):
    write_ranks(selector, tmp_path / "results")
    write_gate(selector, tmp_path / "endurance", "four-stage-lean-pair")
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage-lean-pair" and all(len(c["savings"]) == 12 * len(selector["ROUTINGS"]) for c in result["choices"].values())
    # The FA3 choice stays; the MegaMoE switches are appended for the Prefill workers.
    assert env == "SGLANG_PREFILL_FA3_VERIFY=n192\nDG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_LEAN=1\nDG_SM90_PREFILL_TAIL_PAIR=1\n"
    assert set(selector["VARIANTS"]) == {"control", "four-stage", "four-stage-lean", "four-stage-lean-pair", "four-stage-lean-split", "four-stage-lean-split-rcp"}
    assert set(selector["CANDIDATES"]) == set(selector["VARIANTS"]) - {"control"}
    assert selector["TOLERANCE_GATED"] == ("four-stage-lean-split-rcp",)


def test_faster_split_wins_with_lean_and_without_pairing(selector, tmp_path):
    write_ranks(selector, tmp_path / "results", split=2650.0)
    write_gate(selector, tmp_path / "endurance", "four-stage-lean-split")
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage-lean-split"
    # The host only honours the split switch on the lean path without tail pairing.
    assert env.endswith("n192\nDG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_LEAN=1\nDG_SM90_PREFILL_SPLIT_ACCUM=1\n")
    edit(tmp_path / "results", 0, 2, "edges", "four-stage-lean-split", bitwise_equal=False)
    write_gate(selector, tmp_path / "endurance", "four-stage-lean-pair")
    result, _ = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage-lean-pair" and result["choices"]["four-stage-lean-split"]["reason"] == "failed rank 2 edges"


@pytest.mark.parametrize("gate,why", [
    (None, "no endurance result from rank 0 (late, hung or failed)"),
    ({"calls": 0, "passed": False}, "rank 0 failed the endurance gate"),
    ({"bitwise_equal": {"balanced": True, "hot": False, "edges": True, "wide": True}, "passed": False}, "rank 0 failed the endurance gate"),
    ({"variant": "four-stage-lean-split"}, "rank 0 gated four-stage-lean-split, not four-stage-lean-split-rcp"),
])
def test_an_unserved_winner_needs_the_endurance_gate_on_every_gpu(selector, tmp_path, gate, why):
    write_ranks(selector, tmp_path / "results", split=2650.0, rcp=2600.0)
    if gate is not None:
        write_gate(selector, tmp_path / "endurance", "four-stage-lean-split-rcp")
        path = tmp_path / "endurance/prefill-0/rank-0.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), **gate}))
    result, env = aggregate(selector, tmp_path)
    # The fastest eligible variant that has served (the split, 41005872470) takes over.
    assert result["winner"] == "four-stage-lean-split" and result["reason"].startswith(f"four-stage-lean-split-rcp not adopted: {why}")
    assert result["choices"]["four-stage-lean-split-rcp"]["endurance"].startswith(why)
    assert env.endswith("n192\nDG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_LEAN=1\nDG_SM90_PREFILL_SPLIT_ACCUM=1\n")


def test_unserved_winner_falls_back_to_lean_when_the_split_is_not_eligible(selector, tmp_path):
    write_ranks(selector, tmp_path / "results", split=2650.0, rcp=2600.0)
    edit(tmp_path / "results", 1, 1, "edges", "four-stage-lean-split", bitwise_equal=False)
    result, _ = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage-lean" and result["reason"].startswith("four-stage-lean-split-rcp not adopted: no endurance result")


def test_served_winners_need_no_endurance_gate(selector, tmp_path):
    write_ranks(selector, tmp_path / "results", pair=2760.0)
    result, _ = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage-lean" and result["reason"] == "fastest-eligible"
    write_ranks(selector, tmp_path / "split", pair=2760.0, split=2650.0)
    selector["aggregate"](tmp_path / "split", tmp_path / "split.json", tmp_path / "split.env", 3, tmp_path / "no-gate")
    result = json.loads((tmp_path / "split.json").read_text())
    assert result["winner"] == "four-stage-lean-split" and result["reason"] == "fastest-eligible"
    assert set(selector["UNGATED"]) == {"control", "four-stage", "four-stage-lean", "four-stage-lean-split"}
    assert not set(selector["UNGATED"]) & {"four-stage-lean-pair", "four-stage-lean-split-rcp"}


def test_endurance_run_skips_the_gpu_for_a_served_winner(selector, tmp_path, monkeypatch):
    # The gate recomputes the aggregate's winner from the same rank files before touching a GPU.
    write_ranks(selector, tmp_path / "results", pair=2760.0, split=2650.0)
    monkeypatch.setenv("LOCAL_RANK", "2")
    selector["run_endurance"](tmp_path / "results", tmp_path / "endurance/prefill-1", 3)
    entry = json.loads((tmp_path / "endurance/prefill-1/rank-2.json").read_text())
    assert entry["variant"] == "four-stage-lean-split" and not entry["needed"] and entry["passed"] and entry["calls"] == 0
    assert selector["ENDURANCE_CALLS"] >= 600


def test_faster_shared_rcp_wins_within_the_tolerance_on_top_of_the_split_switch(selector, tmp_path):
    write_ranks(selector, tmp_path / "results", split=2650.0, rcp=2600.0)
    write_gate(selector, tmp_path / "endurance", "four-stage-lean-split-rcp")
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage-lean-split-rcp" and result["reason"] == "fastest-eligible, passed the endurance gate"
    # The host honours the shared-reciprocal switch only on the split path.
    assert env.endswith("n192\nDG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_LEAN=1\nDG_SM90_PREFILL_SPLIT_ACCUM=1\nDG_SM90_PREFILL_SHARED_RCP=1\n")
    rule = result["rule"]
    assert rule["tolerance"] == {"variants": ["four-stage-lean-split-rcp"], "relative_l2": 1e-3, "max_scaled_error": 2e-2}
    assert "four-stage-lean-split-rcp" not in rule["bit_exact"] and "four-stage-lean-split" in rule["bit_exact"]
    assert rule["exact_routings"] == ["edges", "wide"]


@pytest.mark.parametrize("kind,change,reason", [
    # Past either bound, non-finite, or off the FP32 reference on any GPU and routing.
    ("hot", {"control_relative_l2": 1.1e-3}, "failed-or-untimed rank 9 hot"),
    ("balanced", {"control_max_scaled_error": 0.021}, "failed-or-untimed rank 9 balanced"),
    ("wide", {"finite": False}, "failed rank 9 wide"),
    ("edges", {"reference_passed": False}, "failed rank 9 edges"),
])
def test_shared_rcp_outside_the_tolerance_leaves_the_split(selector, tmp_path, kind, change, reason):
    write_ranks(selector, tmp_path / "results", split=2650.0, rcp=2600.0)
    write_gate(selector, tmp_path / "endurance", "four-stage-lean-split-rcp")
    edit(tmp_path / "results", 2, 1, kind, "four-stage-lean-split-rcp", **change)
    result, _ = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage-lean-split" and result["choices"]["four-stage-lean-split-rcp"]["reason"] == reason


def test_bit_exact_variants_get_no_tolerance(selector, tmp_path):
    # Within the shared reciprocal's bounds but not bit-equal: a defect for a bit-exact variant.
    write_ranks(selector, tmp_path / "results", split=2650.0)
    edit(tmp_path / "results", 0, 0, "wide", "four-stage-lean-split", bitwise_equal=False, control_relative_l2=0.0, control_max_scaled_error=0.0)
    result, _ = aggregate(selector, tmp_path)
    # The pair (no endurance result here) is not adopted, so the served lean variant is.
    assert result["winner"] == "four-stage-lean" and result["choices"]["four-stage-lean-split"]["reason"] == "failed rank 0 wide"


def test_wide_case_is_the_balanced_routing_on_larger_hidden_states(selector):
    torch = pytest.importorskip("torch")
    generator = torch.Generator()
    generator.manual_seed(7)
    hidden, ids, weights = selector["inputs"](torch, "wide", 64, generator, "cpu")
    generator.manual_seed(7)
    base_hidden, base_ids, base_weights = selector["inputs"](torch, "balanced", 64, generator, "cpu")
    assert torch.equal(ids, base_ids) and torch.equal(weights, base_weights)
    assert 35 < hidden.float().std() / base_hidden.float().std() < 45 and selector["WIDE_HIDDEN_SCALE"] == 40.0


def test_slower_pair_leaves_lean(selector, tmp_path):
    write_ranks(selector, tmp_path / "results", pair=2760.0)
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage-lean" and result["choices"]["four-stage-lean-pair"]["eligible"]
    assert env.endswith("DG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_LEAN=1\n")


def test_slower_lean_and_pair_leave_four_stages(selector, tmp_path):
    write_ranks(selector, tmp_path / "results", lean=2750.0, pair=2745.0)
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage"
    assert env.endswith("n192\nDG_SM90_PREFILL_4STAGE=1\n")


def test_edge_routing_is_exactness_only(selector, tmp_path):
    # Untimed, but one mismatch on the edge routing rejects the variant everywhere.
    rows = selector["EDGE_EXPERT_ROWS"]
    assert len(rows) == 32 and sum(rows) <= 4096 and {1, 63, 64, 65, 127, 128, 129} <= set(rows)
    write_ranks(selector, tmp_path / "results")
    edit(tmp_path / "results", 1, 0, "edges", "four-stage-lean-pair", bitwise_equal=False)
    result, _ = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage-lean" and result["choices"]["four-stage-lean-pair"]["reason"] == "failed rank 4 edges"
    edit(tmp_path / "results", 2, 3, "edges", "control", reference_passed=False)
    with pytest.raises(SystemExit, match="control failed the FP32 reference"):
        aggregate(selector, tmp_path)


@pytest.mark.parametrize("variant,change,winner", [
    # One slower case or GPU does not reject a variant that saves the summed margin.
    ("four-stage-lean-pair", {"us": [2950.0] * 12}, "four-stage-lean-pair"),
    ("four-stage-lean-pair", {"us": [2790.0] * 12}, "four-stage-lean-pair"),
    # Every candidate must be bit-exact: any difference is a defect, not FP order.
    ("four-stage-lean-pair", {"bitwise_equal": False}, "four-stage-lean"),
    ("four-stage-lean-pair", {"reference_passed": False}, "four-stage-lean"),
    ("four-stage", {"bitwise_equal": False}, "four-stage-lean-pair"),
    ("four-stage", {"us": [2700.0] * 11}, "four-stage-lean-pair"),
    ("four-stage-lean", {"bitwise_equal": False}, "four-stage-lean-pair"),
])
def test_one_failing_gpu_rejects_only_that_variant(selector, tmp_path, variant, change, winner):
    write_ranks(selector, tmp_path / "results")
    write_gate(selector, tmp_path / "endurance", "four-stage-lean-pair")
    edit(tmp_path / "results", 2, 3, "hot", variant, **change)
    result, _ = aggregate(selector, tmp_path)
    assert result["winner"] == winner
    assert not result["choices"][variant]["eligible"] or variant == winner


def test_no_eligible_variant_keeps_three_stages(selector, tmp_path):
    write_ranks(selector, tmp_path / "results", four=2790.0, lean=2795.0, pair=2792.0)
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "control" and result["env"] == {}
    assert env == "SGLANG_PREFILL_FA3_VERIFY=n192\n"


def test_missing_worker_result_keeps_three_stages(selector, tmp_path):
    write_ranks(selector, tmp_path / "results")
    (tmp_path / "results/prefill-1/rank-2.json").unlink()
    result, _ = aggregate(selector, tmp_path)
    assert result == {**result, "winner": "control", "reason": "missing-results", "env": {}}


def test_control_reference_failure_stops_the_job(selector, tmp_path):
    write_ranks(selector, tmp_path / "results")
    edit(tmp_path / "results", 0, 1, "balanced", "control", reference_passed=False, relative_l2=0.4)
    with pytest.raises(SystemExit, match="control failed the FP32 reference"):
        aggregate(selector, tmp_path)


def test_margin_is_the_larger_of_absolute_and_relative(selector, tmp_path):
    # 0.5% of 2800 us is 14 us: a 10 us gain is noise, a 15 us gain is adopted.
    write_ranks(selector, tmp_path / "a", four=2790.0, lean=2790.0, pair=2790.0)
    out = tmp_path / "a.json"
    selector["aggregate"](tmp_path / "a", out, tmp_path / "a.env", 3, tmp_path / "no-gate")
    assert json.loads(out.read_text())["winner"] == "control"
    write_ranks(selector, tmp_path / "b", four=2785.0, lean=2790.0, pair=2790.0)
    selector["aggregate"](tmp_path / "b", out, tmp_path / "b.env", 3, tmp_path / "no-gate")
    assert json.loads(out.read_text())["winner"] == "four-stage"


def test_pp4_job_gates_before_model_load_and_workers_source_the_switch(tmp_path):
    shell = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    pp4 = shell[shell.index('if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then'):shell.index('elif [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]; then')]
    assert "select-firmus-prefill-megamoe.py\" --ranks 1 --output" in pp4
    assert '--aggregate "${ROOT}/results/prefill-megamoe"' in pp4 and '--env-output "${ROOT}/source/prefill-kernels.env"' in pp4
    assert pp4.index("--ranks 1") < pp4.index("--aggregate")
    # The endurance gate runs per worker after all selections and before the aggregate, under its own timeout.
    gate = next(line for line in pp4.splitlines() if "select-firmus-prefill-megamoe.py\" --endurance" in line)
    assert "timeout --kill-after=5 90 " in gate and '--output "${ROOT}/results/prefill-megamoe-endurance/prefill-${worker}"' in gate
    assert pp4.index("--ranks 1") < pp4.index(gate) < pp4.index("--aggregate")
    assert '--endurance "${ROOT}/results/prefill-megamoe-endurance"' in pp4[pp4.index("--aggregate"):]
    assert shell.index("select-firmus-prefill-megamoe.py") < shell.index('bash "${ROOT}/source/worker.sh" prefill-0')
    # worker.sh is unchanged (frozen-submission test): it already exports every prefill-kernels.env line.
    worker = shell.split('cat >"${ROOT}/source/worker.sh" <<\'EOF\'\n', 1)[1].split("\nEOF", 1)[0]
    assert 'if [[ "$role" == prefill ]]; then set -a; source "$(dirname "$0")/prefill-kernels.env"; set +a; fi' in worker
    for line in pp4.splitlines():
        assert not line.rstrip().endswith("\\")
    script = tmp_path / "pp4.sh"
    script.write_text(pp4 + "fi\n")
    subprocess.run(["bash", "-n", str(script)], check=True)
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    assert '"${GITHUB_WORKSPACE}/scripts/select-firmus-prefill-megamoe.py" "${runtime_dir}/scripts/"' in provision


def test_host_and_header_agree_on_the_four_stage_layout():
    host = (ROOT / "patches/deep-gemm-host/sm90-prefill-4stage.patch").read_text()
    header = (ROOT / "patches/deep-gemm-bench/prefill-4stage/sm90-prefill-4stage.patch").read_text()
    assert "sm90-prefill-4stage.patch" in (ROOT / "patches/deep-gemm-host/series").read_text().split()
    assert "align(num_epilogue_warps * 16 * 144, kSmemAlignment)" in host
    assert "kNumEpilogueWarps * 16 * kPrefillWarpCDRowBytes" in header and "kPrefillWarpCDRowBytes = 128 + 16;" in header
    assert 'get_env<int>("DG_SM90_PREFILL_4STAGE", 0)' in host and "prefill_2wg and" in host
    assert re.search(r"kPrefillWarpCD = kPrefill2WG and kNumStages == 4;", header)
    # 1024 expert counters + 8192 dispatch buffers + 18 KiB CD + 144 barrier bytes + four 50,192-byte stages.
    assert 1024 + 8192 + 18432 + 144 + 4 * 50192 <= 232448 < 1024 + 8192 + 65536 + 144 + 4 * 50192
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())["deep_gemm_overlay_patches"]
    pp4_only = ["patches/deep-gemm-bench/prefill-local-ep1/local-ep1.patch", "patches/deep-gemm-bench/prefill-4stage/sm90-prefill-4stage.patch", "patches/deep-gemm-bench/prefill-lean/sm90-prefill-lean.patch", "patches/deep-gemm-bench/prefill-tail-pair/sm90-prefill-tail-pair.patch", "patches/deep-gemm-bench/prefill-split-accum/sm90-prefill-split-accum.patch", "patches/deep-gemm-bench/prefill-shared-rcp/sm90-prefill-shared-rcp.patch"]
    # Only the Decode 1D2D BLOCK_M 192 overlay (a different header) follows the MegaMoE tiles.
    assert config[-1] == "patches/deep-gemm-bench/decode-dense-bm192/sm90-1d2d-decode-bm192.patch"
    assert config[-7:-1] == pp4_only
    main = json.loads((ROOT / "configs/firmus-framework.json").read_text())["deep_gemm_overlay_patches"]
    assert not set(pp4_only[1:]) & set(main)


def test_lean_is_an_ep1_trailing_flag_after_four_stages_with_the_same_bytes():
    host = (ROOT / "patches/deep-gemm-host/sm90-prefill-lean.patch").read_text()
    header = (ROOT / "patches/deep-gemm-bench/prefill-lean/sm90-prefill-lean.patch").read_text()
    series = (ROOT / "patches/deep-gemm-host/series").read_text().split()
    assert series[-6:] == ["sm90-prefill-4stage.patch", "sm90-prefill-lean.patch", "sm90-prefill-tail-pair.patch", "sm90-prefill-split-accum.patch", "sm90-prefill-shared-rcp.patch", "sm90-1d2d-decode-bm192.patch"]
    for gone in ("sm90-prefill-direct-fc2.patch", "sm90-prefill-gather-a.patch", "sm90-prefill-tail-split.patch", "sm90-prefill-split-ldsm.patch", "sm90-prefill-combine-norm.patch"):
        assert not (ROOT / "patches/deep-gemm-host" / gone).exists()
    # ",true" (kPrefillLean) only when set and one rank only.
    assert 'args.prefill_lean ? ",\\n        true" : ""' in host
    assert "num_ranks == 1 and config.block_m == 128" in host and 'config.num_stages == 4 and get_env<int>("DG_SM90_PREFILL_LEAN", 0) != 0' in host
    assert "+    bool kPrefillLean = false,\n" in header
    assert "DG_STATIC_ASSERT(not kPrefillLean or (kPrefillWarpCD and kNumRanks == 1)" in header
    # Counts go straight into the sums the scheduler polls; SM 0's republication and the pull barrier are skipped.
    assert "ptx::atomic_add(kPrefillLean ? workspace.get_expert_recv_count_sum_ptr(i) : workspace.get_expert_send_count_ptr(i), send_value)" in header
    assert "+        if constexpr (not kPrefillLean) {\n" in header
    # Before the combine: grid sync and SM 0's self-signal only.
    assert "[&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }, true, false);" in header
    # stmatrix.x4 lane l addresses row l % 16 of n8 chunk 2q + l / 16; registers 0..3 hold
    # (r_0, 2q), (r_1, 2q), (r_0, 2q + 1), (r_1, 2q + 1) as the STS.32 path wrote them.
    assert "(lane_idx % 16) * kPrefillWarpCDRowBytes + (lane_idx / 16) * 16" in header and '"r"(stage_stsm + q * 32)' in header
    sts = {}
    for row_idx in range(8):
        for col_idx in range(4):
            for j in range(8):
                chunk = j
                sts[(row_idx * 144 + col_idx * 4 + j * 16)] = ("r0", chunk, col_idx)
                sts[((row_idx + 8) * 144 + col_idx * 4 + j * 16)] = ("r1", chunk, col_idx)
    stsm = {}
    for q in range(4):
        for matrix, (half, dchunk) in enumerate((("r0", 0), ("r1", 0), ("r0", 1), ("r1", 1))):
            lane_base = 8 * matrix
            for row in range(8):
                address = ((lane_base + row) % 16) * 144 + ((lane_base + row) // 16) * 16 + q * 32
                for t in range(32):
                    if t // 4 == row:
                        stsm[address + 4 * (t % 4)] = (half, 2 * q + dchunk, t % 4)
    assert stsm == sts


def test_split_accumulator_is_a_lean_only_trailing_flag_without_traps_in_the_math_role():
    host = (ROOT / "patches/deep-gemm-host/sm90-prefill-split-accum.patch").read_text()
    header = (ROOT / "patches/deep-gemm-bench/prefill-split-accum/sm90-prefill-split-accum.patch").read_text()
    added = "\n".join(line[1:] for line in header.splitlines() if line.startswith("+") and not line.startswith("+++"))
    # ",true,false,true" (kPrefillLean, kPrefillTailPair=false, kPrefillSplitAccum) on the lean path only.
    assert 'args.prefill_split_accum ? ",\\n        true,\\n        false,\\n        true"' in host
    assert 'prefill_lean and not prefill_tail_pair and get_env<int>("DG_SM90_PREFILL_SPLIT_ACCUM", 0) != 0' in host
    assert "    bool kPrefillSplitAccum = false,\n" in added
    assert "DG_STATIC_ASSERT(not kPrefillSplitAccum or (kPrefillLean and not kPrefillTailPair)" in added
    # Whole straight-line chunks of the FC1 (32) and FC2 (12) K loops.
    unroll = re.search(r"kL1SplitUnroll = (\d+), kL2SplitUnroll = (\d+);", added)
    assert 32 % int(unroll.group(1)) == 0 and 12 % int(unroll.group(2)) == 0
    # The math warpgroups take the setmaxnreg budget (64 x 48 + 64 x 40 + 256 x regs <= 64512).
    regs = int(re.search(r"kNumEpilogueThreads == 256 \? \(kPrefillSplitAccum \? (\d+) : 168\)", added).group(1))
    assert 168 < regs and regs % 8 == 0 and 64 * 48 + 64 * 40 + 256 * regs <= 64512
    # No trap left in the math role of the split instantiation: the pre-combine barrier is
    # inlined without the timeout trap and the runtime combine-buffer assert is compiled out.
    assert "while (ptx::ld_acq_sys(signal_ptr) != target);" in added
    assert "if constexpr (not kPrefillSplitAccum) {" in added
    # One wait per half: the oldest pending group is the same half of the previous K block.
    assert added.count("ptx::warpgroup_wait<1>();") == 2 and added.count("HalfWGMMA::wgmma(") == 1


def test_shared_rcp_is_a_split_only_trailing_flag_with_a_per_element_fallback():
    host = (ROOT / "patches/deep-gemm-host/sm90-prefill-shared-rcp.patch").read_text()
    header = (ROOT / "patches/deep-gemm-bench/prefill-shared-rcp/sm90-prefill-shared-rcp.patch").read_text()
    added = "\n".join(line[1:] for line in header.splitlines() if line.startswith("+") and not line.startswith("+++"))
    # ",true,false,true,true" (kPrefillLean, kPrefillTailPair=false, kPrefillSplitAccum, kPrefillSharedRcp) on the split path only.
    assert 'args.prefill_shared_rcp ? ",\\n        true,\\n        false,\\n        true,\\n        true"' in host
    assert 'prefill_split_accum and fast_math and std::isinf(activation_clamp) and get_env<int>("DG_SM90_PREFILL_SHARED_RCP", 0) != 0' in host
    assert "    bool kPrefillSharedRcp = false,\n" in added
    assert "DG_STATIC_ASSERT(not kPrefillSharedRcp or (kPrefillSplitAccum and kFastMath and" in added
    # The same EX2 and RCP instructions; one RCP per column pair, one per element on the fallback.
    assert added.count('asm("ex2.approx.ftz.f32 %0, %1;"') == 1 and added.count('asm("rcp.approx.ftz.f32 %0, %1;"') == 1
    assert "const float d = fmaf(d_0, e_1, d_0);" in added and "const float r = rcp(d) * s_gate;" in added
    assert "v_0 = (g_0 * fmaf(e_1, r, r)) * u_0;" in added and "v_1 = (g_1 * (d_0 * r)) * u_1;" in added
    assert "return (g * (rcp(1.0f + ex2(g * t)) * s_gate)) * u;" in added
    # The fallback bound keeps rcp(d) * s_gate normal, and the vote makes the fallback branch warp-uniform.
    assert "fminf(fminf(l1_final_scale[0], l1_final_scale[1]), 1.0f) * 0x1p126f" in added
    assert "if (__any_sync(0xffffffffu, not (max_d < max_d_bound))) {" in added
    # The split loop leaves the final scales to the epilogue; amax and the quantizer take s_up.
    assert "l1_final_scale[2] = prev_scale_a_0 * prev_up_sf;" in added and "amax_r0 *= l1_final_scale[2];" in added and "sf_inv_pair.x *= l1_final_scale[2];" in added


def test_shared_rcp_formula_matches_the_per_element_sigmoid():
    np = pytest.importorskip("numpy")
    # s_gate / d_0 = (1 + e_1) (R s_gate) and s_gate / d_1 = d_0 (R s_gate) with R = 1 / (d_0 (1 + e_1)), in FP32.
    rng = np.random.default_rng(0)
    g = rng.standard_normal((2, 4096)).astype(np.float32) * 20
    s_gate = np.float32(2e-5)
    with np.errstate(over="ignore"):
        # Gates below about -88 give e = inf in FP32, as EX2 does; their pairs take the fallback.
        e = np.exp2(g.astype(np.float64) * -1.4426950408889634).astype(np.float32)
        d_0 = np.float32(1) + e[0]
        d = (d_0.astype(np.float64) * e[1] + d_0).astype(np.float32)
    bound = min(s_gate, np.float32(1)) * np.float32(2.0 ** 126)
    ok = d < bound
    r = (np.float32(1) / d * s_gate).astype(np.float32)
    sig_0 = ((e[1].astype(np.float64) * r + r).astype(np.float32))
    sig_1 = (d_0 * r).astype(np.float32)
    want_0, want_1 = s_gate / (np.float32(1) + e[0]), s_gate / (np.float32(1) + e[1])
    assert ok.mean() > 0.99 and np.all(r[ok] >= np.float32(2.0 ** -126))
    assert np.max(np.abs(sig_0[ok] / want_0[ok] - 1)) < 4e-7 and np.max(np.abs(sig_1[ok] / want_1[ok] - 1)) < 4e-7
