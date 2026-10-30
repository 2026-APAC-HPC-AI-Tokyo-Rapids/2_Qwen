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


# The split default (2790 us) is below the 0.5% margin, so it never wins unless a test lowers it.
def write_ranks(selector, directory, workers=3, control=2800.0, four=2740.0, lean=2720.0, pair=2700.0, split=2790.0):
    for worker in range(workers):
        path = directory / f"prefill-{worker}"
        path.mkdir(parents=True)
        for rank in range(4):
            routings = {}
            times = {"four-stage": four, "four-stage-lean": lean, "four-stage-lean-pair": pair, "four-stage-lean-split": split}
            base = {"finite": True, "relative_l2": 0.012, "max_scaled_error": 0.02, "reference_passed": True}
            for kind in selector["ROUTINGS"]:
                routings[kind] = {"control": {**base, "us": [control] * selector["ROUNDS"]}, **{name: {**base, "bitwise_equal": True, "us": [us] * selector["ROUNDS"]} for name, us in times.items()}}
            for kind in selector["EXACT_ROUTINGS"]:
                routings[kind] = {"control": {**base, "us": []}, **{name: {**base, "bitwise_equal": True, "us": []} for name in times}}
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
            entry = {"rank": rank, "variant": variant, "calls": selector["ENDURANCE_CALLS"], "needed": True, "bitwise_equal": {k: True for k in ("balanced", "hot", "edges")}, "passed": True, **changes}
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
    assert set(selector["VARIANTS"]) == {"control", "four-stage", "four-stage-lean", "four-stage-lean-pair", "four-stage-lean-split"}


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
    ({"bitwise_equal": {"balanced": True, "hot": False, "edges": True}, "passed": False}, "rank 0 failed the endurance gate"),
    ({"variant": "four-stage-lean"}, "rank 0 gated four-stage-lean, not four-stage-lean-split"),
])
def test_an_unserved_winner_needs_the_endurance_gate_on_every_gpu(selector, tmp_path, gate, why):
    write_ranks(selector, tmp_path / "results", split=2650.0)
    if gate is not None:
        write_gate(selector, tmp_path / "endurance", "four-stage-lean-split")
        path = tmp_path / "endurance/prefill-0/rank-0.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), **gate}))
    result, env = aggregate(selector, tmp_path)
    # The fastest eligible variant that has served (lean, 12k+ batches in 40964714831) takes over; pair is unserved too.
    assert result["winner"] == "four-stage-lean" and result["reason"].startswith(f"four-stage-lean-split not adopted: {why}")
    assert result["choices"]["four-stage-lean-split"]["endurance"].startswith(why)
    assert env.endswith("n192\nDG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_LEAN=1\n")


def test_served_winners_need_no_endurance_gate(selector, tmp_path):
    write_ranks(selector, tmp_path / "results", pair=2760.0)
    result, _ = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage-lean" and result["reason"] == "fastest-eligible"
    assert set(selector["UNGATED"]) == {"control", "four-stage", "four-stage-lean"}
    assert not set(selector["UNGATED"]) & {"four-stage-lean-pair", "four-stage-lean-split"}


def test_endurance_run_skips_the_gpu_for_a_served_winner(selector, tmp_path, monkeypatch):
    # The gate recomputes the aggregate's winner from the same rank files before touching a GPU.
    write_ranks(selector, tmp_path / "results", pair=2760.0)
    monkeypatch.setenv("LOCAL_RANK", "2")
    selector["run_endurance"](tmp_path / "results", tmp_path / "endurance/prefill-1", 3)
    entry = json.loads((tmp_path / "endurance/prefill-1/rank-2.json").read_text())
    assert entry["variant"] == "four-stage-lean" and not entry["needed"] and entry["passed"] and entry["calls"] == 0
    assert selector["ENDURANCE_CALLS"] >= 600


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
    pp4_only = ["patches/deep-gemm-bench/prefill-local-ep1/local-ep1.patch", "patches/deep-gemm-bench/prefill-4stage/sm90-prefill-4stage.patch", "patches/deep-gemm-bench/prefill-lean/sm90-prefill-lean.patch", "patches/deep-gemm-bench/prefill-tail-pair/sm90-prefill-tail-pair.patch", "patches/deep-gemm-bench/prefill-split-accum/sm90-prefill-split-accum.patch", "patches/deep-gemm-bench/prefill-split-ldsm/sm90-prefill-split-ldsm.patch"]
    # Only the Decode 1D2D BLOCK_M 192 overlay (a different header) follows the MegaMoE tiles.
    assert config[-1] == "patches/deep-gemm-bench/decode-dense-bm192/sm90-1d2d-decode-bm192.patch"
    assert config[-7:-1] == pp4_only
    main = json.loads((ROOT / "configs/firmus-framework.json").read_text())["deep_gemm_overlay_patches"]
    assert not set(pp4_only[1:]) & set(main)


def test_lean_is_an_ep1_trailing_flag_after_four_stages_with_the_same_bytes():
    host = (ROOT / "patches/deep-gemm-host/sm90-prefill-lean.patch").read_text()
    header = (ROOT / "patches/deep-gemm-bench/prefill-lean/sm90-prefill-lean.patch").read_text()
    series = (ROOT / "patches/deep-gemm-host/series").read_text().split()
    assert series[-6:] == ["sm90-prefill-4stage.patch", "sm90-prefill-lean.patch", "sm90-prefill-tail-pair.patch", "sm90-prefill-split-accum.patch", "sm90-prefill-split-ldsm.patch", "sm90-1d2d-decode-bm192.patch"]
    for gone in ("sm90-prefill-direct-fc2.patch", "sm90-prefill-gather-a.patch", "sm90-prefill-tail-split.patch"):
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


def test_ldmatrix_split_is_a_split_only_trailing_flag_whose_addresses_give_the_wgmma_a_fragment():
    host = (ROOT / "patches/deep-gemm-host/sm90-prefill-split-ldsm.patch").read_text()
    header = (ROOT / "patches/deep-gemm-bench/prefill-split-ldsm/sm90-prefill-split-ldsm.patch").read_text()
    added = "\n".join(line[1:] for line in header.splitlines() if line.startswith("+") and not line.startswith("+++"))
    # ",true,false,true,true" (kPrefillLean, kPrefillTailPair=false, kPrefillSplitAccum, kPrefillSplitLdsm) on the split path only.
    assert 'args.prefill_split_ldsm ? ",\\n        true,\\n        false,\\n        true,\\n        true"' in host
    assert 'prefill_split_accum and get_env<int>("DG_SM90_PREFILL_SPLIT_LDSM", 0) != 0' in host
    assert "    bool kPrefillSplitLdsm = false,\n" in added
    assert "DG_STATIC_ASSERT(not kPrefillSplitLdsm or kPrefillSplitAccum" in added
    # Two K blocks of A registers alternate inside a chunk, so the chunks have even length.
    assert "kUnroll % 2 == 0 and kSwizzleAMode == 128 and BLOCK_K == 128" in added
    assert "issue_half(h, a_regs[u % 2]);" in added and "load_a(a_regs[(u + 1) % 2], next_stage);" in added
    assert "MMA_64x128x32_F32E4M3E4M3_RS_TN" in added
    # The ldmatrix addresses (row and swizzled 16-byte chunk per lane) as written in the patch.
    assert "(warp_idx_in_wg * 16 + ((lane_idx >> 3) & 1u) * 8 + (lane_idx & 7u)) * BLOCK_K" in added
    assert "(((2 * j + (lane_idx >> 4)) ^ (lane_idx & 7u)) << 4)" in added
    # Replay them against the TMA 128B swizzle (chunk c of row r at c ^ (r % 8) within a
    # 128-byte row, 1024-byte aligned) and the m64k32 FP8 A fragment of thread t: register
    # 4j + i holds row 16 w + t / 4 + 8 (i & 1), bytes 32 j + 16 (i >> 1) + 4 (t % 4) .. +3.
    physical = {}
    for row in range(64):
        for byte in range(128):
            physical[row * 128 + (((byte // 16) ^ (row % 8)) << 4) + byte % 16] = (row, byte)
    for warp in range(4):
        for j in range(4):
            lane_address = {}
            for lane in range(32):
                row = warp * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)
                lane_address[lane] = row * 128 + (((2 * j + (lane >> 4)) ^ (lane & 7)) << 4)
            for t in range(32):
                for i in range(4):
                    # ldmatrix: thread t gets bytes 4 (t % 4) .. +3 of row t / 4 of matrix i,
                    # whose row r address comes from lane 8 i + r.
                    base = lane_address[8 * i + t // 4] + 4 * (t % 4)
                    got = [physical[base + b] for b in range(4)]
                    want = [(16 * warp + t // 4 + 8 * (i & 1), 32 * j + 16 * (i >> 1) + 4 * (t % 4) + b) for b in range(4)]
                    assert got == want
