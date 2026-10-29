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


def write_ranks(selector, directory, workers=3, control=2800.0, four=2740.0, lean=2720.0, pair=2700.0):
    for worker in range(workers):
        path = directory / f"prefill-{worker}"
        path.mkdir(parents=True)
        for rank in range(4):
            routings = {}
            times = {"four-stage": four, "four-stage-lean": lean, "four-stage-lean-pair": pair}
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


def aggregate(selector, tmp_path):
    out, env = tmp_path / "selection.json", tmp_path / "prefill-kernels.env"
    env.write_text("SGLANG_PREFILL_FA3_VERIFY=n192\n")
    selector["aggregate"](tmp_path / "results", out, env, 3)
    return json.loads(out.read_text()), env.read_text()


def test_adopts_the_fastest_variant_that_passed_everywhere(selector, tmp_path):
    write_ranks(selector, tmp_path / "results")
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage-lean-pair" and all(len(c["savings"]) == 12 * len(selector["ROUTINGS"]) for c in result["choices"].values())
    # The FA3 choice stays; the MegaMoE switches are appended for the Prefill workers.
    assert env == "SGLANG_PREFILL_FA3_VERIFY=n192\nDG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_LEAN=1\nDG_SM90_PREFILL_TAIL_PAIR=1\n"
    assert set(selector["VARIANTS"]) == {"control", "four-stage", "four-stage-lean", "four-stage-lean-pair"}


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
    selector["aggregate"](tmp_path / "a", out, tmp_path / "a.env", 3)
    assert json.loads(out.read_text())["winner"] == "control"
    write_ranks(selector, tmp_path / "b", four=2785.0, lean=2790.0, pair=2790.0)
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
    pp4_only = ["patches/deep-gemm-bench/prefill-local-ep1/local-ep1.patch", "patches/deep-gemm-bench/prefill-4stage/sm90-prefill-4stage.patch", "patches/deep-gemm-bench/prefill-lean/sm90-prefill-lean.patch", "patches/deep-gemm-bench/prefill-tail-pair/sm90-prefill-tail-pair.patch"]
    # Only the dense 1D2D schedule overlay (select-firmus-prefill-attention.py, a different header) follows the MegaMoE tiles.
    assert config[-1] == "patches/deep-gemm-bench/prefill-dense-schedule/sm90-1d2d-prefill-schedule.patch"
    assert config[-5:-1] == pp4_only
    main = json.loads((ROOT / "configs/firmus-framework.json").read_text())["deep_gemm_overlay_patches"]
    assert not set(pp4_only[1:]) & set(main)


def test_lean_is_an_ep1_trailing_flag_after_four_stages_with_the_same_bytes():
    host = (ROOT / "patches/deep-gemm-host/sm90-prefill-lean.patch").read_text()
    header = (ROOT / "patches/deep-gemm-bench/prefill-lean/sm90-prefill-lean.patch").read_text()
    series = (ROOT / "patches/deep-gemm-host/series").read_text().split()
    assert series[-3:] == ["sm90-prefill-4stage.patch", "sm90-prefill-lean.patch", "sm90-prefill-tail-pair.patch"]
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
