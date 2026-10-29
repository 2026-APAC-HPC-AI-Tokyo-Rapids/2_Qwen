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


def write_ranks(selector, directory, workers=3, control=2800.0, candidate=2740.0):
    for worker in range(workers):
        path = directory / f"prefill-{worker}"
        path.mkdir(parents=True)
        for rank in range(4):
            routings = {}
            for kind in selector["ROUTINGS"]:
                routings[kind] = {"finite": True, "relative_l2": 0.012, "max_scaled_error": 0.02, "control_reference_passed": True, "bitwise_equal": True, "control_us": [control] * selector["PAIRS"], "candidate_us": [candidate] * selector["PAIRS"]}
            (path / f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "ranks_per_group": 1, "tokens": 8064, "routings": routings}))


def edit(directory, worker, rank, kind, **changes):
    path = directory / f"prefill-{worker}" / f"rank-{rank}.json"
    entry = json.loads(path.read_text())
    entry["routings"][kind].update(changes)
    path.write_text(json.dumps(entry))


def aggregate(selector, tmp_path):
    out, env = tmp_path / "selection.json", tmp_path / "prefill-kernels.env"
    env.write_text("SGLANG_PREFILL_FA3_VERIFY=n192\n")
    selector["aggregate"](tmp_path / "results", out, env, 3)
    return json.loads(out.read_text()), env.read_text()


def test_adopts_only_when_every_gpu_is_exact_and_faster(selector, tmp_path):
    write_ranks(selector, tmp_path / "results")
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "four-stage" and len(result["savings"]) == 12 * len(selector["ROUTINGS"])
    # The FA3 choice stays; the MegaMoE switch is appended for the Prefill workers.
    assert env == "SGLANG_PREFILL_FA3_VERIFY=n192\nDG_SM90_PREFILL_4STAGE=1\n"


@pytest.mark.parametrize("change", [{"candidate_us": [2790.0] * 12}, {"bitwise_equal": False}, {"candidate_us": [2700.0] * 11}])
def test_one_slow_inexact_or_untimed_gpu_keeps_three_stages(selector, tmp_path, change):
    write_ranks(selector, tmp_path / "results")
    edit(tmp_path / "results", 2, 3, "hot", **change)
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
    edit(tmp_path / "results", 0, 1, "balanced", control_reference_passed=False, relative_l2=0.4)
    with pytest.raises(SystemExit, match="control failed the FP32 reference"):
        aggregate(selector, tmp_path)


def test_margin_is_the_larger_of_absolute_and_relative(selector, tmp_path):
    # 0.5% of 2800 us is 14 us: a 10 us gain is noise, a 15 us gain is adopted.
    write_ranks(selector, tmp_path / "a", candidate=2790.0)
    out = tmp_path / "a.json"
    selector["aggregate"](tmp_path / "a", out, tmp_path / "a.env", 3)
    assert json.loads(out.read_text())["winner"] == "control"
    write_ranks(selector, tmp_path / "b", candidate=2785.0)
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
    assert config[-2:] == ["patches/deep-gemm-bench/prefill-local-ep1/local-ep1.patch", "patches/deep-gemm-bench/prefill-4stage/sm90-prefill-4stage.patch"]
    main = json.loads((ROOT / "configs/firmus-framework.json").read_text())["deep_gemm_overlay_patches"]
    assert "patches/deep-gemm-bench/prefill-4stage/sm90-prefill-4stage.patch" not in main
