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


def write_ranks(selector, directory, workers=3, control=2800.0, four=2740.0, split=2700.0, direct=2720.0, gather=2730.0, direct_gather=2725.0):
    for worker in range(workers):
        path = directory / f"prefill-{worker}"
        path.mkdir(parents=True)
        for rank in range(4):
            routings = {}
            for kind in selector["ROUTINGS"]:
                base = {"finite": True, "relative_l2": 0.012, "max_scaled_error": 0.02, "reference_passed": True}
                exact = {"bitwise_equal": True, "relative_l2_vs_control": 0.0, "max_scaled_error_vs_control": 0.0}
                times = {"four-stage": four, "four-stage-split": split, "four-stage-direct": direct, "four-stage-gather": gather, "four-stage-direct-gather": direct_gather}
                routings[kind] = {"control": {**base, "us": [control] * selector["ROUNDS"]}, **{name: {**base, **exact, "us": [us] * selector["ROUNDS"]} for name, us in times.items()}}
            (path / f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "ranks_per_group": 1, "tokens": 8064, "routings": routings}))


def edit(directory, worker, rank, kind, variant, **changes):
    path = directory / f"prefill-{worker}" / f"rank-{rank}.json"
    entry = json.loads(path.read_text())
    entry["routings"][kind][variant].update(changes)
    path.write_text(json.dumps(entry))


def aggregate(selector, tmp_path):
    out, env = tmp_path / "selection.json", tmp_path / "prefill-kernels.env"
    env.write_text("SGLANG_PREFILL_FA3_VERIFY=n192\n")
    selector["aggregate"](tmp_path / "results", out, env, 3)
    return json.loads(out.read_text()), env.read_text()


def test_adopts_the_fastest_variant_that_passed_everywhere(selector, tmp_path):
    write_ranks(selector, tmp_path / "results")
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage-split" and all(len(c["savings"]) == 12 * len(selector["ROUTINGS"]) for c in result["choices"].values())
    # The FA3 choice stays; the MegaMoE switches are appended for the Prefill workers.
    assert env == "SGLANG_PREFILL_FA3_VERIFY=n192\nDG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_TAIL_SPLIT=1\n"


def test_slower_split_leaves_four_stages(selector, tmp_path):
    write_ranks(selector, tmp_path / "results", split=2760.0, direct=2750.0, gather=2750.0, direct_gather=2750.0)
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage" and result["choices"]["four-stage-split"]["eligible"]
    assert env.endswith("DG_SM90_PREFILL_4STAGE=1\n")


def test_fastest_direct_fc2_writes_its_switch(selector, tmp_path):
    write_ranks(selector, tmp_path / "results", direct=2690.0)
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage-direct"
    assert env == "SGLANG_PREFILL_FA3_VERIFY=n192\nDG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_DIRECT_FC2=1\n"


def test_fastest_direct_gather_writes_both_switches(selector, tmp_path):
    write_ranks(selector, tmp_path / "results", direct_gather=2680.0)
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage-direct-gather"
    assert env == "SGLANG_PREFILL_FA3_VERIFY=n192\nDG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_DIRECT_FC2=1\nDG_SM90_PREFILL_GATHER_A=1\n"


@pytest.mark.parametrize("variant,change,winner", [
    # One slower case or GPU does not reject a variant that saves the summed margin.
    ("four-stage-split", {"us": [2950.0] * 12}, "four-stage-split"),
    ("four-stage-split", {"us": [2790.0] * 12}, "four-stage-split"),
    ("four-stage-split", {"bitwise_equal": False, "relative_l2_vs_control": 0.0004, "max_scaled_error_vs_control": 0.004}, "four-stage-split"),
    ("four-stage-split", {"bitwise_equal": False, "relative_l2_vs_control": 0.05, "max_scaled_error_vs_control": 0.3}, "four-stage-direct"),
    ("four-stage-split", {"reference_passed": False}, "four-stage-direct"),
    ("four-stage", {"bitwise_equal": False, "relative_l2_vs_control": 0.0004, "max_scaled_error_vs_control": 0.004}, "four-stage-split"),
    ("four-stage", {"us": [2700.0] * 11}, "four-stage-split"),
    # Direct FC2 stores must be bit-exact: any difference is a layout defect, not FP order.
    ("four-stage-direct", {"bitwise_equal": False, "relative_l2_vs_control": 0.0004, "max_scaled_error_vs_control": 0.004}, "four-stage-split"),
    # So must the A gather: valid rows see the same bytes.
    ("four-stage-gather", {"bitwise_equal": False, "relative_l2_vs_control": 0.0004, "max_scaled_error_vs_control": 0.004}, "four-stage-split"),
])
def test_one_failing_gpu_rejects_only_that_variant(selector, tmp_path, variant, change, winner):
    write_ranks(selector, tmp_path / "results")
    edit(tmp_path / "results", 2, 3, "hot", variant, **change)
    result, _ = aggregate(selector, tmp_path)
    assert result["winner"] == winner
    assert not result["choices"][variant]["eligible"] or variant == winner


def test_no_eligible_variant_keeps_three_stages(selector, tmp_path):
    write_ranks(selector, tmp_path / "results", four=2790.0, split=2795.0, direct=2792.0, gather=2793.0, direct_gather=2791.0)
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
    write_ranks(selector, tmp_path / "a", four=2790.0, split=2790.0, direct=2790.0, gather=2790.0, direct_gather=2790.0)
    out = tmp_path / "a.json"
    selector["aggregate"](tmp_path / "a", out, tmp_path / "a.env", 3)
    assert json.loads(out.read_text())["winner"] == "control"
    write_ranks(selector, tmp_path / "b", four=2785.0, split=2790.0, direct=2790.0, gather=2790.0, direct_gather=2790.0)
    selector["aggregate"](tmp_path / "b", out, tmp_path / "b.env", 3)
    assert json.loads(out.read_text())["winner"] == "four-stage"


def test_pp4_job_gates_before_model_load_and_workers_source_the_switch(tmp_path):
    shell = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    pp4 = shell[shell.index('if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then'):shell.index('elif [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]; then')]
    assert "select-firmus-prefill-megamoe.py\" --ranks 1 --output" in pp4
    assert '--aggregate "${ROOT}/results/prefill-megamoe"' in pp4 and '--env-output "${ROOT}/source/prefill-kernels.env"' in pp4
    assert pp4.index("--ranks 1") < pp4.index("--aggregate")
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
    pp4_only = ["patches/deep-gemm-bench/prefill-local-ep1/local-ep1.patch", "patches/deep-gemm-bench/prefill-4stage/sm90-prefill-4stage.patch", "patches/deep-gemm-bench/prefill-tail-split/sm90-prefill-tail-split.patch", "patches/deep-gemm-bench/prefill-direct-fc2/sm90-prefill-direct-fc2.patch", "patches/deep-gemm-bench/prefill-gather-a/sm90-prefill-gather-a.patch"]
    # Only the dense 1D2D schedule overlay (select-firmus-prefill-attention.py, a different header) follows the MegaMoE tiles.
    assert config[-1] == "patches/deep-gemm-bench/prefill-dense-schedule/sm90-1d2d-prefill-schedule.patch"
    assert config[-6:-1] == pp4_only
    main = json.loads((ROOT / "configs/firmus-framework.json").read_text())["deep_gemm_overlay_patches"]
    assert not set(pp4_only[1:]) & set(main)


def test_tail_split_is_only_a_trailing_template_flag_of_the_four_stage_tile():
    host = (ROOT / "patches/deep-gemm-host/sm90-prefill-tail-split.patch").read_text()
    header = (ROOT / "patches/deep-gemm-bench/prefill-tail-split/sm90-prefill-tail-split.patch").read_text()
    series = (ROOT / "patches/deep-gemm-host/series").read_text().split()
    assert series.index("sm90-prefill-tail-split.patch") > series.index("sm90-prefill-4stage.patch")
    # Generated text gains ",true" only when set, so every other kernel keeps its JIT text.
    assert 'args.prefill_tail_split ? ",\\n        true" : ""' in host
    assert "config.num_stages == 4 and get_env<int>(\"DG_SM90_PREFILL_TAIL_SPLIT\", 0) != 0" in host
    assert "+    bool kPrefillTailSplit = false,\n" in header
    assert "DG_STATIC_ASSERT(not kPrefillTailSplit or kPrefillWarpCD" in header
    # Both warpgroups publish a split tail's FC1 rows: 12 N blocks x 2 warpgroups x 4 warps.
    assert "expected = kNumL1BlockNs * kNumEpilogueWarpgroups * kL1OutputArrivalParts;" in header


def test_direct_fc2_is_a_trailing_template_flag_and_its_permutation_round_trips():
    host = (ROOT / "patches/deep-gemm-host/sm90-prefill-direct-fc2.patch").read_text()
    header = (ROOT / "patches/deep-gemm-bench/prefill-direct-fc2/sm90-prefill-direct-fc2.patch").read_text()
    series = (ROOT / "patches/deep-gemm-host/series").read_text().split()
    assert series[-3:] == ["sm90-prefill-tail-split.patch", "sm90-prefill-direct-fc2.patch", "sm90-prefill-gather-a.patch"]
    # ",false,true" (kPrefillTailSplit, kPrefillDirectFC2) only when set and never with the split.
    assert 'args.prefill_direct_fc2 ? ",\\n        false,\\n        true" : (args.prefill_tail_split ? ",\\n        true" : "")' in host
    assert 'config.num_stages == 4 and not prefill_tail_split and get_env<int>("DG_SM90_PREFILL_DIRECT_FC2", 0) != 0' in host
    assert "+    bool kPrefillDirectFC2 = false,\n" in header
    assert "DG_STATIC_ASSERT(not kPrefillDirectFC2 or (kPrefillWarpCD and not kPrefillTailSplit)" in header
    assert "DG_STATIC_ASSERT(not kPrefillDirectFC2 or kPairedCombine" in header
    # Writer: lane quad position c stores n8 chunks 4j..4j+3 (columns 8i + 2c, +1) at byte 64 j + 16 c + 4 q.
    assert "n_idx * sizeof(nv_bfloat16) + col_idx * sizeof(uint4)), dst_rank_idx);" in header and "dst[j * 4] = packed;" in header
    written = {64 * j + 16 * c + 4 * q: 8 * (4 * j + q) + 2 * c for c in range(4) for j in range(8) for q in range(4)}
    assert sorted(written) == list(range(0, 512, 4))
    # Combine: uint4 lane L of a 512-byte block, word q, goes to natural byte 64 (L / 4) + 16 q + 4 (L % 4).
    assert "(lane_idx / 4) * 16 + lane_idx % 4;" in header and "ptx::st_shared(natural + 12, casted.w);" in header
    for lane in range(32):
        for q in range(4):
            assert 2 * written[16 * lane + 4 * q] == 64 * (lane // 4) + 16 * q + 4 * (lane % 4)


def test_gather_is_an_ep1_trailing_flag_after_direct_fc2():
    host = (ROOT / "patches/deep-gemm-host/sm90-prefill-gather-a.patch").read_text()
    header = (ROOT / "patches/deep-gemm-bench/prefill-gather-a/sm90-prefill-gather-a.patch").read_text()
    # kPrefillTailSplit=false, kPrefillDirectFC2, kPrefillGatherA=true; one rank only.
    assert '",\\n        false,\\n        true,\\n        true" : ",\\n        false,\\n        false,\\n        true"' in host
    assert "num_ranks == 1 and config.block_m == 128" in host and 'get_env<int>("DG_SM90_PREFILL_GATHER_A", 0) != 0' in host
    assert "+    bool kPrefillGatherA = false,\n" in header
    assert "DG_STATIC_ASSERT(not kPrefillGatherA or (kPrefillWarpCD and not kPrefillTailSplit and kNumRanks == 1)" in header
    # Each lane tracks its copies on the stage's full barrier before the loader's single arrival.
    assert header.index("cp.async.mbarrier.arrive.shared::cta.b64") < header.index("full_barriers[stage_idx]->arrive_and_expect_tx(BLOCK_M * sizeof(float));")
    assert header.count('asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");') == 2
    # The gathered placement matches TMA SWIZZLE_128B: 16-byte chunk c of row r at r * 128 + 16 (c ^ r % 8).
    for i in range(0, 32, 4):
        for j in range(4):
            for chunk in range(8):
                dst = ((i * 128 + ((chunk * 16) ^ ((i % 8) * 16))) ^ (j * 16)) + j * 128
                row = i + j
                assert dst == row * 128 + 16 * (chunk ^ (row % 8))
