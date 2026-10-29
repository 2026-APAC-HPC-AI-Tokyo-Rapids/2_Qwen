import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def selector():
    return runpy.run_path(str(ROOT / "scripts/select-firmus-prefill-attention.py"))


def write_gpus(selector, directory, moe_input=(104.0, 64.0)):
    directory.mkdir(parents=True)
    for rank in range(4):
        cases = {selection: {case: {"passed": True} for case in names} for selection, names in selector["cases"]().items()}
        us = {"moe_input": {"fused": [moe_input[0]] * selector["ROUNDS"], "onepass": [moe_input[1]] * selector["ROUNDS"]}}
        (directory / f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "cases": cases, "us": us}))


def edit(directory, rank, change):
    path = directory / f"rank-{rank}.json"
    entry = json.loads(path.read_text())
    change(entry)
    path.write_text(json.dumps(entry))


PINNED = "SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen\nSGLANG_PREFILL_MOE_INPUT=fused\nSGLANG_PREFILL_QKNORM=fp32\n"


def aggregate(selector, tmp_path, directory=None):
    out, env = tmp_path / "selection.json", tmp_path / "prefill-kernels.env"
    env.write_text(PINNED)
    selector["aggregate"](directory or tmp_path / "results", out, env)
    return json.loads(out.read_text()), env.read_text()


def test_only_the_moe_input_is_selected(selector):
    # The Q/K norm writer is pinned (prepare-firmus-pp4-settings.py); the dense schedule lost twice and was removed.
    assert selector["SELECTIONS"] == {"moe_input": ("fused", "onepass", "SGLANG_PREFILL_MOE_INPUT")}


def test_slower_gpu_or_missing_case_keeps_fused(selector, tmp_path):
    write_gpus(selector, tmp_path / "a")
    edit(tmp_path / "a", 1, lambda e: e["us"]["moe_input"].update(onepass=[102.0] * selector["ROUNDS"]))
    result, env = aggregate(selector, tmp_path, tmp_path / "a")
    assert result["moe_input"]["reason"] == "below-margin" and result["env"] == {} and env == PINNED
    write_gpus(selector, tmp_path / "b")
    edit(tmp_path / "b", 2, lambda e: e["cases"]["moe_input"].pop("3994"))
    result, env = aggregate(selector, tmp_path, tmp_path / "b")
    assert result["moe_input"]["reason"] == "failed-or-untimed gpu 2" and env == PINNED


def test_missing_or_untimed_gpu_keeps_fused(selector, tmp_path):
    write_gpus(selector, tmp_path / "results")
    (tmp_path / "results/rank-2.json").unlink()
    result, _ = aggregate(selector, tmp_path)
    assert result["moe_input"]["reason"] == "missing-results" and result["env"] == {}
    write_gpus(selector, tmp_path / "short")
    edit(tmp_path / "short", 0, lambda e: e["us"]["moe_input"].update(fused=[104.0] * (selector["ROUNDS"] - 1)))
    assert aggregate(selector, tmp_path, tmp_path / "short")[0]["moe_input"]["winner"] == "fused"
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
    # Triton kernels only (the rope module and the DeepGEMM schedule are no longer compiled here).
    assert 'TRITON_CACHE_DIR="/tmp/qwen-prefill-attention-${SLURM_JOB_ID}/triton"' in launch
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
    assert pp4["sglang_overlay_patches"][-3:] == ["patches/sglang/prefill-qknorm-fp32.patch", "patches/sglang/prefill-moe-input-onepass.patch", "patches/sglang/decode-moe-input.patch"]
    assert pp4["deep_gemm_overlay_patches"][-1] == "patches/deep-gemm-bench/decode-dense-bm192/sm90-1d2d-decode-bm192.patch"
    for patch in ("patches/sglang/prefill-qknorm-fp32.patch", "patches/sglang/prefill-moe-input-onepass.patch"):
        assert patch not in main["sglang_overlay_patches"]
    qknorm = (ROOT / "patches/sglang/prefill-qknorm-fp32.patch").read_text()
    assert '+_PREFILL_QKNORM = os.environ.get("SGLANG_PREFILL_QKNORM", "control")' in qknorm
    # Only without the FA3 V byte image, whose writer stays the control kernel.
    assert '+                if _PREFILL_QKNORM == "fp32" and self.attn.prefill_v_image is None:' in qknorm


def test_onepass_moe_input_overrides_the_pinned_fused_path(selector, tmp_path):
    # 2% of 104 us is 2.08 us, so 3 us applies; the switch is appended after the pinned fused line.
    write_gpus(selector, tmp_path / "a", moe_input=(104.0, 101.1))
    assert aggregate(selector, tmp_path, tmp_path / "a")[0]["moe_input"]["reason"] == "below-margin"
    write_gpus(selector, tmp_path / "b", moe_input=(104.0, 64.0))
    result, env = aggregate(selector, tmp_path, tmp_path / "b")
    assert result["moe_input"]["winner"] == "onepass" and result["env"]["SGLANG_PREFILL_MOE_INPUT"] == "onepass"
    assert env == PINNED + "SGLANG_PREFILL_MOE_INPUT=onepass\n"
    # A failed row count on one GPU keeps fused.
    write_gpus(selector, tmp_path / "c", moe_input=(104.0, 64.0))
    edit(tmp_path / "c", 2, lambda e: e["cases"]["moe_input"]["2049"].update(passed=False))
    result, env = aggregate(selector, tmp_path, tmp_path / "c")
    assert result["moe_input"]["winner"] == "fused" and result["moe_input"]["reason"] == "failed-or-untimed gpu 2" and env == PINNED


def test_onepass_overlay_keeps_the_other_modes_and_the_buffer_contract():
    patch = (ROOT / "patches/sglang/prefill-moe-input-onepass.patch").read_text()
    assert '+assert _PREFILL_MOE_INPUT in ("control", "fused", "fused-gate", "onepass"), _PREFILL_MOE_INPUT' in patch
    # Same entry point and guards as the fused modes (_prefill_moe_input); no y, so no gate GEMM call.
    assert '+        if mode == "onepass":' in patch and '+        if mode == "fused-gate":' in patch
    assert "+            return run_sm90_mega_routed(self, hidden_states, None, None, buf, num_tokens)" in patch
    # Pad slots of the whole buffer, 64-row programs, scales finished only after a CTA barrier.
    assert "+    _onepass_moe_input[(buf_x.shape[0] // 64,)](" in patch and "+    tl.debug_barrier()" in patch
    assert 'tl.store(IDS + out, tl.where(live[:, None], selected_idx, -1).to(tl.int64))' in patch


def test_moe_input_gate_sign_test_tolerates_ties_but_not_a_biased_router(selector):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch without float8")
    sigmas = selector["SET_SIGMAS"]
    # 300 rows where exactly one path hits the FP64 set: 140 / 160 is noise, 120 / 180 is a biased router.
    assert 160 - 140 <= sigmas * 300 ** 0.5 + 2 and not 180 - 120 <= sigmas * 300 ** 0.5 + 2
    step = selector["bf16_logit_step"](torch, torch.tensor([[3.0, -1.0], [-0.75, 0.5]], dtype=torch.float64))
    assert step.flatten().tolist() == [2.0 ** -6, 2.0 ** -8]
