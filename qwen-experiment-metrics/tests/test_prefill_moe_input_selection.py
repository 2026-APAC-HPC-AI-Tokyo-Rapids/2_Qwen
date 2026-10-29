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


def test_pp4_job_selects_on_decode_gpus_before_model_load():
    shell = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    pp4 = shell[shell.index('if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then'):shell.index('elif [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]; then')]
    launch = next(line for line in pp4.splitlines() if "select-firmus-prefill-moe-input.py\" --output" in line)
    assert '-w "${TAIL}" --gres=gpu:4' in launch and launch.rstrip().endswith("|| true") and "timeout --kill-after=5 150" in launch
    # On the PP4 bundle the four Decode GPUs run this first, then the Decode selection, in one background
    # group beside the Prefill-GPU selections; the result is taken before the Prefill workers start.
    group = pp4[pp4.rindex("  {\n", 0, pp4.index(launch)):pp4.index("  } &\n") + 6]
    assert group.index(launch) < group.index("--pinned-env") and "decode_gpu_selections=$!" in pp4
    assert pp4.index("decode_gpu_selections=$!") < pp4.index("select-firmus-prefill-megamoe.py")
    take = shell[shell.index('if [[ -n "${decode_gpu_selections:-}" ]]; then'):]
    take = take[:take.index("\nfi\n")]
    assert take.index('wait "${decode_gpu_selections}" || true') < take.index('--aggregate "${ROOT}/results/prefill-moe-input"')
    assert '--env-output "${ROOT}/source/prefill-kernels.env"' in take
    assert shell.index("select-firmus-prefill-fa3.py\" --aggregate") < shell.index(take) < shell.index('bash "${ROOT}/source/worker.sh" prefill-0')
    assert shell.index("select-firmus-prefill-moe-input.py") < shell.index('bash "${ROOT}/source/worker.sh" prefill-0')
    worker = shell.split('cat >"${ROOT}/source/worker.sh" <<\'EOF\'\n', 1)[1].split("\nEOF", 1)[0]
    assert 'if [[ "$role" == prefill ]]; then set -a; source "$(dirname "$0")/prefill-kernels.env"; set +a; fi' in worker
    for line in pp4.splitlines():
        assert not line.rstrip().endswith("\\")
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    assert '"${GITHUB_WORKSPACE}/scripts/select-firmus-prefill-moe-input.py" "${runtime_dir}/scripts/"' in provision


def test_overlay_is_pp4_only_and_defaults_to_pre_dispatch():
    pp4 = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    main = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    # Only the attention-block overlays (select-firmus-prefill-attention.py) follow it.
    assert pp4["sglang_overlay_patches"][-3] == "patches/sglang/prefill-moe-input.patch"
    # The FA3 varlen overlay touches other files; it stays right before this one.
    assert pp4["sglang_overlay_patches"][-4] == "patches/sglang/prefill-fa3-varlen.patch"
    assert "patches/sglang/prefill-moe-input.patch" not in main["sglang_overlay_patches"]
    patch = (ROOT / "patches/sglang/prefill-moe-input.patch").read_text()
    assert '+_PREFILL_MOE_INPUT = os.environ.get("SGLANG_PREFILL_MOE_INPUT", "control")' in patch
    # Prefill EXTEND batches only, on the trivial placement without observers; Decode keeps pre-dispatch.
    for guard in ('_PREFILL_MOE_INPUT == "control"', "forward_batch.forward_mode != ForwardMode.EXTEND", "hidden_states.shape[0] < _INPUT_NORM_QUANT_MIN_ROWS", 'args.disaggregation_mode == "prefill"', "args.ep_dispatch_algorithm is None", "mlp._decode_routing_capture is None", "forward_batch.num_token_non_padded is not None"):
        assert guard in patch, guard
    assert "+    if topk_ids is not None:\n+        deep_gemm.mega_moe_pre_dispatch_sm90(" in patch
