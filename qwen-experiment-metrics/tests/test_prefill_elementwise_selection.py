import json
import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SBATCH = ROOT / "platforms/firmus/framework-comparison/sglang.sbatch"
QKNORM_PATCH = ROOT / "patches/sglang/prefill-qknorm-fp32-heads.patch"
SPLIT_PATCH = ROOT / "patches/sglang/prefill-moe-input-onepass-split.patch"


@pytest.fixture
def selector():
    return runpy.run_path(str(ROOT / "scripts/select-firmus-prefill-elementwise.py"))


def entry(selector, onepass=119.0, split=70.0, fp32=84.0, heads4=70.0, heads8=64.0):
    """One GPU's result: every gate case passed, every candidate stable, the given medians at both shapes."""
    cases = selector["cases"]()
    rounds = selector["ROUNDS"]
    result = {"cases": {"moe_input": {key: {"onepass-split": {"passed": True}} for key in cases["moe_input"]}, "qknorm": {key: {"fp32-heads4": {"passed": True}, "fp32-heads8": {"passed": True}} for key in cases["qknorm"]}}, "us": {}, "endurance": {}}
    for rows in selector["TIMED_ROWS"]:
        half = 1.0 if rows == selector["TIMED_ROWS"][0] else 0.5
        result["us"][f"moe_input-{rows}"] = {"onepass": [onepass * half] * rounds, "onepass-split": [split * half] * rounds}
        result["us"][f"qknorm-{rows}"] = {"fp32": [fp32 * half] * rounds, "fp32-heads4": [heads4 * half] * rounds, "fp32-heads8": [heads8 * half] * rounds}
    for candidate in ("onepass-split", "fp32-heads4", "fp32-heads8"):
        result["endurance"][candidate] = {"calls": selector["ENDURANCE_CALLS"], "stable": True}
    return result


def write(selector, directory, change=None, missing=None, **times):
    for worker in range(3):
        for rank in range(4):
            if (worker, rank) == missing:
                continue
            path = directory / f"prefill-{worker}" / f"rank-{rank}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            result = entry(selector, **times)
            if change is not None and (worker, rank) == (2, 3):
                change(result)
            path.write_text(json.dumps(result))


def aggregate(selector, tmp_path):
    out, env = tmp_path / "selection.json", tmp_path / "prefill-kernels.env"
    env.write_text("SGLANG_PREFILL_MOE_INPUT=onepass\nSGLANG_PREFILL_QKNORM=fp32\n")
    selector["aggregate"](tmp_path / "results", out, env, 3)
    return json.loads(out.read_text()), env.read_text()


def test_both_candidates_are_appended_after_the_pins_when_they_win_everywhere(selector, tmp_path):
    write(selector, tmp_path / "results")
    result, env = aggregate(selector, tmp_path)
    # Sourced with set -a: the appended lines override the pins.
    assert env == "SGLANG_PREFILL_MOE_INPUT=onepass\nSGLANG_PREFILL_QKNORM=fp32\nSGLANG_PREFILL_MOE_INPUT=onepass-split\nSGLANG_PREFILL_QKNORM=fp32-heads8\n"
    assert result["moe_input"]["winner"] == "onepass-split" and result["qknorm"]["winner"] == "fp32-heads8"
    assert len(result["qknorm"]["candidates"]["fp32-heads4"]["savings"]) == 12


def test_the_larger_summed_qknorm_saving_wins(selector, tmp_path):
    write(selector, tmp_path / "results", heads4=60.0, heads8=66.0)
    result, env = aggregate(selector, tmp_path)
    assert result["qknorm"]["winner"] == "fp32-heads4" and env.endswith("SGLANG_PREFILL_QKNORM=fp32-heads4\n")


def test_a_missing_gpu_keeps_both_pins(selector, tmp_path):
    write(selector, tmp_path / "results", missing=(1, 2))
    result, env = aggregate(selector, tmp_path)
    assert result["env"] == {} and env == "SGLANG_PREFILL_MOE_INPUT=onepass\nSGLANG_PREFILL_QKNORM=fp32\n"
    assert result["moe_input"]["candidates"]["onepass-split"]["reason"] == "missing-results"


@pytest.mark.parametrize(("change", "reason"), [
    (lambda r: r["cases"]["moe_input"]["4032"]["onepass-split"].update(passed=False), "failed-gate gpu 11"),
    (lambda r: r["endurance"]["onepass-split"].update(stable=False), "failed-endurance gpu 11"),
    (lambda r: r["endurance"].pop("onepass-split"), "failed-endurance gpu 11"),
    (lambda r: r["us"]["moe_input-8064"].update({"onepass-split": [117.0] * 12}), "below-margin gpu 11 at 8064 rows: 119.0 -> 117.0 us"),
    # Not slower at one request per microbatch either.
    (lambda r: r["us"]["moe_input-4032"].update({"onepass-split": [60.0] * 12}), "below-margin gpu 11 at 4032 rows: 59.5 -> 60.0 us"),
    (lambda r: r["us"]["moe_input-4032"].update({"onepass-split": [60.0] * 5}), "untimed gpu 11"),
])
def test_any_failure_on_one_gpu_keeps_the_onepass_pin(selector, tmp_path, change, reason):
    write(selector, tmp_path / "results", change=change)
    result, env = aggregate(selector, tmp_path)
    assert result["moe_input"]["winner"] == "onepass" and result["moe_input"]["candidates"]["onepass-split"]["reason"] == reason
    assert "SGLANG_PREFILL_MOE_INPUT=onepass-split" not in env and result["env"] == {"SGLANG_PREFILL_QKNORM": "fp32-heads8"}


def test_a_qknorm_candidate_that_is_not_bit_equal_is_never_adopted(selector, tmp_path):
    def change(result):
        result["cases"]["qknorm"]["2049-int32"]["fp32-heads8"]["passed"] = False
    write(selector, tmp_path / "results", change=change)
    result, _ = aggregate(selector, tmp_path)
    assert result["qknorm"]["winner"] == "fp32-heads4" and result["qknorm"]["candidates"]["fp32-heads8"]["reason"] == "failed-gate gpu 11"


def test_gate_cases_cover_the_microbatch_shapes_and_both_position_types(selector):
    assert selector["TIMED_ROWS"] == (8064, 4032) and {8064, 4032, 2049} <= set(selector["ROWS"])
    assert selector["ENDURANCE_CALLS"] >= 300
    assert "8064-int32" in selector["cases"]()["qknorm"] and "2049-int64" in selector["cases"]()["qknorm"]


def test_overlays_follow_the_pp4_chain_and_stay_out_of_the_main_config():
    pp4 = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    main = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    assert pp4["sglang_overlay_patches"][-3:] == ["patches/sglang/decode-moe-input.patch", "patches/sglang/prefill-qknorm-fp32-heads.patch", "patches/sglang/prefill-moe-input-onepass-split.patch"]
    for patch in pp4["sglang_overlay_patches"][-2:]:
        assert patch not in main["sglang_overlay_patches"]


def test_new_modes_are_opt_in_and_keep_the_pinned_kernels():
    qknorm = QKNORM_PATCH.read_text()
    assert '+assert _PREFILL_QKNORM in ("control", "fp32", "fp32-heads4", "fp32-heads8"), _PREFILL_QKNORM' in qknorm
    # Only without the FA3 V byte image, like the fp32 writer; the fp32 branch is unchanged.
    assert "+                if _PREFILL_QKNORM_HEADS is not None and self.attn.prefill_v_image is None:" in qknorm
    assert '+                elif _PREFILL_QKNORM == "fp32" and self.attn.prefill_v_image is None:' in qknorm
    assert "-    if (fp32) {" not in qknorm and "+    if (fp32 && heads) {" in qknorm
    assert 'host::RuntimeCheck(heads == 4 || heads == 8' in qknorm
    split = SPLIT_PATCH.read_text()
    assert '+assert _PREFILL_MOE_INPUT in ("control", "fused", "fused-gate", "onepass", "onepass-split"), _PREFILL_MOE_INPUT' in split
    # The combine norm accepts either one-pass mode; the onepass kernel itself is untouched.
    assert '+        if mode in ("onepass", "onepass-split"):' in split
    assert "def _onepass_moe_input" not in split and "-    _onepass_moe_input[" not in split
    assert "num_warps=8, num_stages=2, maxnreg=128" in split


def test_pp4_keeps_the_elementwise_pins_without_running_the_selection(tmp_path):
    shell = SBATCH.read_text()
    pp4 = shell[shell.index('if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then'):shell.index('elif [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]; then')]
    assert "select-firmus-prefill-elementwise.py" not in shell
    script = tmp_path / "pp4.sh"
    script.write_text(pp4 + "fi\n")
    subprocess.run(["bash", "-n", str(script)], check=True)
