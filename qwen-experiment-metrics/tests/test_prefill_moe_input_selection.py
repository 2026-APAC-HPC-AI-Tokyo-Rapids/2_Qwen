import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def selector():
    return runpy.run_path(str(ROOT / "scripts/select-firmus-prefill-moe-input.py"))


def write_gpus(selector, directory, control=290.0, fused=268.0, gate=258.0):
    directory.mkdir(parents=True)
    for rank in range(4):
        rows = {}
        for count in selector["ROWS"]:
            rows[str(count)] = {"fused": {"passed": True, "y_max_relative": 0.0039}, "fused-gate": {"passed": True, "ids_same_as_router": 0.9995}}
        us = {"control": [control] * selector["ROUNDS"], "fused": [fused] * selector["ROUNDS"], "fused-gate": [gate] * selector["ROUNDS"]}
        (directory / f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "rows": rows, "us": us}))


def edit(directory, rank, change):
    path = directory / f"rank-{rank}.json"
    entry = json.loads(path.read_text())
    change(entry)
    path.write_text(json.dumps(entry))


def aggregate(selector, tmp_path, directory=None):
    out, env = tmp_path / "selection.json", tmp_path / "prefill-kernels.env"
    env.write_text("SGLANG_PREFILL_FA3_VERIFY=n192\n")
    selector["aggregate"](directory or tmp_path / "results", out, env)
    return json.loads(out.read_text()), env.read_text()


def test_fused_gate_is_adopted_when_it_also_beats_fused(selector, tmp_path):
    write_gpus(selector, tmp_path / "results")
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "fused-gate" and all(len(c["savings"]) == 4 for c in result["choices"].values())
    # The FA3 choice stays; the Prefill workers source the appended switch.
    assert env == "SGLANG_PREFILL_FA3_VERIFY=n192\nSGLANG_PREFILL_MOE_INPUT=fused-gate\n"


def test_fused_gate_needs_five_us_over_fused(selector, tmp_path):
    write_gpus(selector, tmp_path / "results", gate=264.0)
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "fused" and result["choices"]["fused-gate"]["reason"] == "below-margin"
    assert env.endswith("SGLANG_PREFILL_MOE_INPUT=fused\n")


def test_margin_is_the_larger_of_three_us_and_two_percent(selector, tmp_path):
    # 2% of 290 us is 5.8 us: a 5 us gain is noise, a 6 us gain is adopted.
    write_gpus(selector, tmp_path / "a", fused=285.0, gate=285.0)
    assert aggregate(selector, tmp_path, tmp_path / "a")[0]["winner"] == "control"
    write_gpus(selector, tmp_path / "b", fused=284.0, gate=284.0)
    assert aggregate(selector, tmp_path, tmp_path / "b")[0]["winner"] == "fused"


@pytest.mark.parametrize("failed,winner", [(("fused", "fused-gate"), "control"), (("fused-gate",), "fused")])
def test_one_failed_shape_on_one_gpu_rejects_the_variant(selector, tmp_path, failed, winner):
    # check_rows fails fused-gate whenever fused fails (same norm kernel).
    write_gpus(selector, tmp_path / "results")
    edit(tmp_path / "results", 3, lambda e: [e["rows"]["2049"][v].update(passed=False) for v in failed])
    result, _ = aggregate(selector, tmp_path)
    assert result["winner"] == winner and not any(result["choices"][v]["eligible"] for v in failed)


def test_fused_gate_alone_can_win(selector, tmp_path):
    # fused below the margin on one GPU, fused-gate well clear of control and fused everywhere.
    write_gpus(selector, tmp_path / "results")
    edit(tmp_path / "results", 1, lambda e: e["us"].update(fused=[289.0] * selector["ROUNDS"]))
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "fused-gate" and not result["choices"]["fused"]["eligible"]
    assert env.endswith("SGLANG_PREFILL_MOE_INPUT=fused-gate\n")


def test_missing_or_untimed_gpu_keeps_pre_dispatch(selector, tmp_path):
    write_gpus(selector, tmp_path / "results")
    (tmp_path / "results/rank-2.json").unlink()
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "control" and result["reason"] == "missing-results" and result["env"] == {}
    assert env == "SGLANG_PREFILL_FA3_VERIFY=n192\n"
    write_gpus(selector, tmp_path / "short")
    edit(tmp_path / "short", 0, lambda e: e["us"].update(control=[290.0] * (selector["ROUNDS"] - 1)))
    assert aggregate(selector, tmp_path, tmp_path / "short")[0]["winner"] == "control"
    assert not (tmp_path / "absent").exists()
    assert aggregate(selector, tmp_path, tmp_path / "absent")[0]["winner"] == "control"


def test_pp4_job_pins_the_onepass_path_and_no_longer_selects_it():
    # fused won on all four GPUs in 40849459077 and 40883389891 and was pinned; onepass then won over the
    # pinned fused path on all four GPUs in 40964714831 (88.9 against 102.9 us per layer, about ten times the
    # noise), so prepare-firmus-pp4-settings.py pins onepass and neither selection runs in the job.
    shell = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    for script in ("select-firmus-prefill-moe-input.py", "select-firmus-prefill-attention.py"):
        assert script not in shell and script not in provision
    assert not (ROOT / "scripts/select-firmus-prefill-attention.py").exists()
    assert "prefill-attention)" not in shell and "results/prefill-attention" not in shell
    prepare = runpy.run_path(str(ROOT / "scripts/prepare-firmus-pp4-settings.py"))
    assert prepare["PP4_SELECTED"]["SGLANG_PREFILL_MOE_INPUT"] == "onepass"
    assert prepare["PP4_SELECTED_RUNS"]["SGLANG_PREFILL_MOE_INPUT"] == [40964714831]
    worker = shell.split('cat >"${ROOT}/source/worker.sh" <<\'EOF\'\n', 1)[1].split("\nEOF", 1)[0]
    assert 'if [[ "$role" == prefill ]]; then set -a; source "$(dirname "$0")/prefill-kernels.env"; set +a; fi' in worker


def test_overlay_is_pp4_only_and_defaults_to_pre_dispatch():
    pp4 = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    main = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    # Only the FP32 Q/K norm writer, the one-pass MoE input (pinned by prepare-firmus-pp4-settings.py),
    # the Decode MoE input overlay, which extends this one to TARGET_VERIFY, and the combine norm follow it.
    assert pp4["sglang_overlay_patches"][-5] == "patches/sglang/prefill-moe-input.patch"
    assert pp4["sglang_overlay_patches"][-2] == "patches/sglang/decode-moe-input.patch"
    # The FA3 varlen overlay touches other files; it stays right before this one.
    assert pp4["sglang_overlay_patches"][-6] == "patches/sglang/prefill-fa3-varlen.patch"
    assert "patches/sglang/prefill-moe-input.patch" not in main["sglang_overlay_patches"]
    patch = (ROOT / "patches/sglang/prefill-moe-input.patch").read_text()
    assert '+_PREFILL_MOE_INPUT = os.environ.get("SGLANG_PREFILL_MOE_INPUT", "control")' in patch
    # Prefill EXTEND batches only, on the trivial placement without observers; Decode keeps pre-dispatch.
    for guard in ('_PREFILL_MOE_INPUT == "control"', "forward_batch.forward_mode != ForwardMode.EXTEND", "hidden_states.shape[0] < _INPUT_NORM_QUANT_MIN_ROWS", 'args.disaggregation_mode == "prefill"', "args.ep_dispatch_algorithm is None", "mlp._decode_routing_capture is None", "forward_batch.num_token_non_padded is not None"):
        assert guard in patch, guard
    assert "+    if topk_ids is not None:\n+        deep_gemm.mega_moe_pre_dispatch_sm90(" in patch


def test_qknorm_and_onepass_overlays_are_pp4_only_and_default_to_control():
    pp4 = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    main = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    assert pp4["sglang_overlay_patches"][-4:] == ["patches/sglang/prefill-qknorm-fp32.patch", "patches/sglang/prefill-moe-input-onepass.patch", "patches/sglang/decode-moe-input.patch", "patches/sglang/prefill-combine-norm.patch"]
    assert pp4["deep_gemm_overlay_patches"][-1] == "patches/deep-gemm-bench/decode-dense-bm192/sm90-1d2d-decode-bm192.patch"
    for patch in ("patches/sglang/prefill-qknorm-fp32.patch", "patches/sglang/prefill-moe-input-onepass.patch"):
        assert patch not in main["sglang_overlay_patches"]
    qknorm = (ROOT / "patches/sglang/prefill-qknorm-fp32.patch").read_text()
    assert '+_PREFILL_QKNORM = os.environ.get("SGLANG_PREFILL_QKNORM", "control")' in qknorm
    # Only without the FA3 V byte image, whose writer stays the control kernel.
    assert '+                if _PREFILL_QKNORM == "fp32" and self.attn.prefill_v_image is None:' in qknorm


def test_onepass_overlay_keeps_the_other_modes_and_the_buffer_contract():
    patch = (ROOT / "patches/sglang/prefill-moe-input-onepass.patch").read_text()
    assert '+assert _PREFILL_MOE_INPUT in ("control", "fused", "fused-gate", "onepass"), _PREFILL_MOE_INPUT' in patch
    # Same entry point and guards as the fused modes (_prefill_moe_input); no y, so no gate GEMM call.
    assert '+        if mode == "onepass":' in patch and '+        if mode == "fused-gate":' in patch
    assert "+            return run_sm90_mega_routed(self, hidden_states, None, None, buf, num_tokens)" in patch
    # Pad slots of the whole buffer, 64-row programs, scales finished only after a CTA barrier.
    assert "+    _onepass_moe_input[(buf_x.shape[0] // 64,)](" in patch and "+    tl.debug_barrier()" in patch
    assert 'tl.store(IDS + out, tl.where(live[:, None], selected_idx, -1).to(tl.int64))' in patch
