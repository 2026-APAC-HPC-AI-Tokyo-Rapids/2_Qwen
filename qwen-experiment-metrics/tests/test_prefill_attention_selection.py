import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def selector():
    return runpy.run_path(str(ROOT / "scripts/select-firmus-prefill-attention.py"))


def write_gpus(selector, directory, qknorm=(130.0, 75.0), dense=(930.0, 880.0)):
    directory.mkdir(parents=True)
    for rank in range(4):
        cases = {selection: {case: {"passed": True} for case in names} for selection, names in selector["cases"]().items()}
        us = {"qknorm": {"control": [qknorm[0]] * selector["ROUNDS"], "fp32": [qknorm[1]] * selector["ROUNDS"]},
              "dense": {"control": [dense[0]] * selector["ROUNDS"], "schedule": [dense[1]] * selector["ROUNDS"]}}
        (directory / f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "cases": cases, "us": us}))


def edit(directory, rank, change):
    path = directory / f"rank-{rank}.json"
    entry = json.loads(path.read_text())
    change(entry)
    path.write_text(json.dumps(entry))


def aggregate(selector, tmp_path, directory=None):
    out, env = tmp_path / "selection.json", tmp_path / "prefill-kernels.env"
    env.write_text("SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen\nSGLANG_PREFILL_MOE_INPUT=fused\n")
    selector["aggregate"](directory or tmp_path / "results", out, env)
    return json.loads(out.read_text()), env.read_text()


def test_both_candidates_are_appended_when_faster_everywhere(selector, tmp_path):
    write_gpus(selector, tmp_path / "results")
    result, env = aggregate(selector, tmp_path)
    assert result["qknorm"]["winner"] == "fp32" and result["dense"]["winner"] == "schedule"
    assert all(len(result[s]["savings"]) == 4 for s in ("qknorm", "dense"))
    # Earlier Prefill choices stay; the Prefill workers source the appended switches.
    assert env == "SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen\nSGLANG_PREFILL_MOE_INPUT=fused\nSGLANG_PREFILL_QKNORM=fp32\nSGLANG_PREFILL_DENSE_GEMM=schedule\n"


def test_selections_are_independent(selector, tmp_path):
    # 2% of 930 us is 18.6 us: an 18 us dense gain is noise while the Q/K writer still wins.
    write_gpus(selector, tmp_path / "results", dense=(930.0, 912.0))
    result, env = aggregate(selector, tmp_path)
    assert result["qknorm"]["winner"] == "fp32" and result["dense"]["winner"] == "control" and result["dense"]["reason"] == "below-margin"
    assert env.endswith("SGLANG_PREFILL_MOE_INPUT=fused\nSGLANG_PREFILL_QKNORM=fp32\n")


def test_margin_is_the_larger_of_three_us_and_two_percent(selector, tmp_path):
    # 2% of 130 us is 2.6 us, so 3 us applies: 2.9 us is noise, 3.1 us is adopted.
    write_gpus(selector, tmp_path / "a", qknorm=(130.0, 127.1), dense=(930.0, 930.0))
    assert aggregate(selector, tmp_path, tmp_path / "a")[0]["qknorm"]["winner"] == "control"
    write_gpus(selector, tmp_path / "b", qknorm=(130.0, 126.9), dense=(930.0, 930.0))
    assert aggregate(selector, tmp_path, tmp_path / "b")[0]["qknorm"]["winner"] == "fp32"


@pytest.mark.parametrize("selection,case", [("qknorm", "2049-int32"), ("dense", "7989")])
def test_one_failed_case_on_one_gpu_rejects_only_that_candidate(selector, tmp_path, selection, case):
    write_gpus(selector, tmp_path / "results")
    edit(tmp_path / "results", 3, lambda e: e["cases"][selection][case].update(passed=False))
    result, env = aggregate(selector, tmp_path)
    other = "dense" if selection == "qknorm" else "qknorm"
    assert result[selection]["winner"] == "control" and result[selection]["reason"] == "failed-or-untimed gpu 3"
    assert result[other]["winner"] != "control" and selector["SELECTIONS"][selection][2] not in env


def test_slower_gpu_or_missing_case_keeps_control(selector, tmp_path):
    write_gpus(selector, tmp_path / "results")
    edit(tmp_path / "results", 1, lambda e: e["us"]["dense"].update(schedule=[935.0] * selector["ROUNDS"]))
    edit(tmp_path / "results", 2, lambda e: e["cases"]["qknorm"].pop("3994-int64"))
    result, env = aggregate(selector, tmp_path)
    assert result["dense"]["reason"] == "below-margin" and result["qknorm"]["reason"] == "failed-or-untimed gpu 2"
    assert result["env"] == {} and env == "SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen\nSGLANG_PREFILL_MOE_INPUT=fused\n"


def test_missing_or_untimed_gpu_keeps_control(selector, tmp_path):
    write_gpus(selector, tmp_path / "results")
    (tmp_path / "results/rank-2.json").unlink()
    result, _ = aggregate(selector, tmp_path)
    assert result["qknorm"]["reason"] == result["dense"]["reason"] == "missing-results" and result["env"] == {}
    write_gpus(selector, tmp_path / "short")
    edit(tmp_path / "short", 0, lambda e: e["us"]["qknorm"].update(control=[130.0] * (selector["ROUNDS"] - 1)))
    assert aggregate(selector, tmp_path, tmp_path / "short")[0]["qknorm"]["winner"] == "control"
    assert aggregate(selector, tmp_path, tmp_path / "absent")[0]["env"] == {}


def test_e4m3_step_distance_is_monotone_across_zero_and_sign(selector):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch without float8")
    values = torch.tensor([-448.0, -2.0, -0.0, 0.0, 2.0 ** -9, 1.0, 448.0]).to(torch.float8_e4m3fn)
    steps = selector["e4m3_steps"](torch, values[1:], values[:-1])
    assert steps.tolist() == [62, 64, 0, 1, 55, 70]


def test_pp4_job_selects_on_decode_gpus_before_model_load():
    shell = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    pp4 = shell[shell.index('if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then'):shell.index('elif [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]; then')]
    launch = next(line for line in pp4.splitlines() if "select-firmus-prefill-attention.py\" --output" in line)
    assert '-w "${TAIL}" --gres=gpu:4' in launch and launch.rstrip().endswith("|| true") and "timeout --kill-after=5 240" in launch
    for cache in ("TRITON_CACHE_DIR", "TVM_FFI_CACHE_DIR", "DG_JIT_CACHE_DIR", "SGLANG_DG_CACHE_DIR"):
        assert f'{cache}="/tmp/qwen-prefill-attention-${{SLURM_JOB_ID}}/' in launch
    # First in the Decode-GPU group (the MoE input path is pinned), before the Decode FA3 selection.
    group = pp4[pp4.rindex("  {\n", 0, pp4.index(launch)):pp4.index("  } &\n") + 6]
    assert group.index("    srun ") == group.index(launch) < group.index("--pinned-env")
    take = shell[shell.index('if [[ -n "${decode_gpu_selections:-}" ]]; then'):]
    take = take[:take.index("\nfi\n")]
    assert take.index('wait "${decode_gpu_selections}" || true') < take.index('--aggregate "${ROOT}/results/prefill-attention"')
    aggregate_line = next(line for line in take.splitlines() if '--aggregate "${ROOT}/results/prefill-attention"' in line)
    assert '--env-output "${ROOT}/source/prefill-kernels.env"' in aggregate_line
    assert shell.index(take) < shell.index('bash "${ROOT}/source/worker.sh" prefill-0')
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    assert '"${GITHUB_WORKSPACE}/scripts/select-firmus-prefill-attention.py" "${runtime_dir}/scripts/"' in provision


def test_overlays_are_pp4_only_and_default_to_control():
    pp4 = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    main = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    assert pp4["sglang_overlay_patches"][-3:] == ["patches/sglang/prefill-qknorm-fp32.patch", "patches/sglang/prefill-dense-schedule.patch", "patches/sglang/decode-moe-input.patch"]
    assert pp4["deep_gemm_overlay_patches"][-2:] == ["patches/deep-gemm-bench/prefill-dense-schedule/sm90-1d2d-prefill-schedule.patch", "patches/deep-gemm-bench/decode-dense-bm192/sm90-1d2d-decode-bm192.patch"]
    for key, patch in (("sglang_overlay_patches", "patches/sglang/prefill-qknorm-fp32.patch"), ("sglang_overlay_patches", "patches/sglang/prefill-dense-schedule.patch"), ("deep_gemm_overlay_patches", "patches/deep-gemm-bench/prefill-dense-schedule/sm90-1d2d-prefill-schedule.patch")):
        assert patch not in main[key]
    qknorm = (ROOT / "patches/sglang/prefill-qknorm-fp32.patch").read_text()
    assert '+_PREFILL_QKNORM = os.environ.get("SGLANG_PREFILL_QKNORM", "control")' in qknorm
    # Only without the FA3 V byte image, whose writer stays the control kernel.
    assert '+                if _PREFILL_QKNORM == "fp32" and self.attn.prefill_v_image is None:' in qknorm
    dense = (ROOT / "patches/sglang/prefill-dense-schedule.patch").read_text()
    assert '+_PREFILL_DENSE_GEMM = os.environ.get("SGLANG_PREFILL_DENSE_GEMM", "control")' in dense
    assert 'compiled_dims="n" if _PREFILL_DENSE_GEMM == "schedule" else "nk"' in dense
    # The DeepGEMM path is keyed on compiled N with run-time K at BM256: the serving control ("nk"),
    # the default ("") and the Decode tiles (BM <= 128) keep the stock instantiation.
    header = (ROOT / "patches/deep-gemm-bench/prefill-dense-schedule/sm90-1d2d-prefill-schedule.patch").read_text()
    assert "kGemmType == GemmType::Normal and SHAPE_N != 0 and SHAPE_K == 0 and" in header
    assert "BLOCK_M == 2 * WAVE_BLOCK_M and WAVE_BLOCK_M == 2 * WGMMA::M and" in header
