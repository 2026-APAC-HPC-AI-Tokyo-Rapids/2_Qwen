import json
import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BATCHES = (44, 46, 48)


@pytest.fixture
def selector():
    return runpy.run_path(str(ROOT / "scripts/select-firmus-decode-moe-input.py"))


def write_gpus(selector, directory, control=14.0, fused=11.0):
    directory.mkdir(parents=True)
    pairs = selector["PAIRS"]
    for rank in range(4):
        rows = {f"{r}/{live}": {"passed": True, "y_max_relative": 0.0039} for r, live in selector["check_cases"](BATCHES)}
        us = {str(b): [{"control": [control] * pairs, "fused": [fused] * pairs} for _ in range(selector["BLOCKS"])] for b in BATCHES}
        (directory / f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "batches": list(BATCHES), "rows": rows, "us": us}))


def edit(directory, rank, change):
    path = directory / f"rank-{rank}.json"
    entry = json.loads(path.read_text())
    change(entry)
    path.write_text(json.dumps(entry))


def aggregate(selector, tmp_path, directory=None):
    out, env = tmp_path / "moe-input.json", tmp_path / "moe-input.env"
    env.write_text("")
    selector["aggregate"](directory or tmp_path / "results", out, env)
    return json.loads(out.read_text()), env.read_text()


def test_batches_and_checked_rows_follow_the_cap_48_configuration(selector):
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    assert config["decode_max_running_requests"] // config["decode_dp_size"] == 48
    assert selector["batches_from_config"](ROOT / "configs/firmus-prefill-pp4.json") == BATCHES
    # Last request padded at each batch (the graph padding mask), then full cap/4 and one request.
    assert selector["check_cases"](BATCHES) == [(176, 172), (184, 180), (192, 188), (192, 192), (4, 4)]


def test_fused_is_adopted_when_faster_everywhere(selector, tmp_path):
    write_gpus(selector, tmp_path / "results")
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "fused" and len(result["savings"]) == 4 * len(BATCHES) * selector["BLOCKS"]
    assert env == "SGLANG_DECODE_MOE_INPUT=fused\n"


def test_margin_is_the_larger_of_one_us_and_two_percent(selector, tmp_path):
    # 2% of 60 us is 1.2 us: a 1.1 us gain is noise, a 1.3 us gain is adopted.
    write_gpus(selector, tmp_path / "a", control=60.0, fused=58.9)
    assert aggregate(selector, tmp_path, tmp_path / "a")[0]["winner"] == "control"
    write_gpus(selector, tmp_path / "b", control=60.0, fused=58.7)
    assert aggregate(selector, tmp_path, tmp_path / "b")[0]["winner"] == "fused"
    write_gpus(selector, tmp_path / "c", control=14.0, fused=13.1)
    assert aggregate(selector, tmp_path, tmp_path / "c")[0]["reason"] == "below-margin"


def test_one_slow_block_or_batch_on_one_gpu_keeps_pre_dispatch(selector, tmp_path):
    write_gpus(selector, tmp_path / "results")
    edit(tmp_path / "results", 2, lambda e: e["us"]["46"][1].update(fused=[13.5] * selector["PAIRS"]))
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "control" and result["reason"] == "below-margin" and env == ""


def test_one_failed_case_on_one_gpu_keeps_pre_dispatch(selector, tmp_path):
    write_gpus(selector, tmp_path / "results")
    edit(tmp_path / "results", 3, lambda e: e["rows"]["176/172"].update(passed=False))
    result, env = aggregate(selector, tmp_path)
    assert result["winner"] == "control" and result["reason"] == "failed-or-untimed gpu 3" and env == ""


def test_missing_untimed_or_inconsistent_gpu_keeps_pre_dispatch(selector, tmp_path):
    write_gpus(selector, tmp_path / "results")
    (tmp_path / "results/rank-1.json").unlink()
    result, env = aggregate(selector, tmp_path)
    assert result == {**result, "winner": "control", "reason": "missing-results", "env": {}} and env == ""
    write_gpus(selector, tmp_path / "short")
    edit(tmp_path / "short", 0, lambda e: e["us"]["48"][0].update(control=[14.0] * (selector["PAIRS"] - 1)))
    assert aggregate(selector, tmp_path, tmp_path / "short")[0]["winner"] == "control"
    write_gpus(selector, tmp_path / "batches")
    edit(tmp_path / "batches", 0, lambda e: e.update(batches=[40, 42, 44]))
    assert aggregate(selector, tmp_path, tmp_path / "batches")[0]["reason"] == "inconsistent-batches"
    assert aggregate(selector, tmp_path, tmp_path / "absent")[0]["winner"] == "control"


def test_pp4_job_runs_the_selection_on_the_decode_gpus_and_appends_only_a_winner(tmp_path):
    shell = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    pp4 = shell[shell.index('if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then'):shell.index('elif [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]; then')]
    group = pp4[pp4.index("  {\n"):pp4.index("  } &\n")]
    moe = group.index('--container-mounts="$(mounts decode-selection-moe-input)" bash "${ROOT}/scripts/select-firmus-decode-kernels.sh" "${ROOT}" moe-input >"${ART}/decode-selection-moe-input.log" 2>&1 || true')
    assert group.index("check-firmus-pp4-copy-engine.py") < moe and '-w "${TAIL}"' in group[group.rindex("srun", 0, moe):moe]
    script = (ROOT / "scripts/select-firmus-decode-kernels.sh").read_text()
    part = script[script.index('elif [[ "${part}" == moe-input ]]; then'):script.index("else\n", script.index('elif [[ "${part}" == moe-input ]]; then'))]
    assert '"${root}/scripts/select-firmus-decode-moe-input.py" --output "${out}/moe-input"' in part
    assert '--aggregate "${out}/moe-input" --output "${out}/moe-input.json" --env-output "${out}/moe-input.env"' in part
    take = shell[shell.index('if [[ -n "${decode_gpu_selections:-}" ]]; then'):]
    take = take[:take.index("\nfi\n") + 4]
    rows = [row.strip() for row in take.splitlines() if "decode-selection/moe-input.env" in row]
    assert len(rows) == 1
    for moe_env, expected in (("", "A=pinned\n"), ("SGLANG_DECODE_MOE_INPUT=fused\n", "A=pinned\nSGLANG_DECODE_MOE_INPUT=fused\n")):
        (tmp_path / "results/decode-selection").mkdir(parents=True, exist_ok=True)
        (tmp_path / "source").mkdir(exist_ok=True)
        (tmp_path / "source/decode-kernels.env").write_text("A=pinned\n")
        (tmp_path / "results/decode-selection/moe-input.env").write_text(moe_env)
        subprocess.run(["bash", "-c", "set -euo pipefail\n" + "\n".join(rows)], env={"ROOT": str(tmp_path), "PATH": "/usr/bin:/bin"}, check=True)
        assert (tmp_path / "source/decode-kernels.env").read_text() == expected


def test_overlay_extends_the_prefill_path_to_target_verify_only():
    patch = (ROOT / "patches/sglang/decode-moe-input.patch").read_text()
    assert '+_DECODE_MOE_INPUT = os.environ.get("SGLANG_DECODE_MOE_INPUT", "control")' in patch
    assert '+assert _DECODE_MOE_INPUT in ("control", "fused"), _DECODE_MOE_INPUT' in patch
    # TARGET_VERIFY up to the graph cap; the Prefill EXTEND conditions are unchanged.
    for guard in ("+        if forward_batch.forward_mode.is_target_verify():", "+            if not 0 < hidden_states.shape[0] <= 192:", "+            forward_batch.forward_mode == ForwardMode.EXTEND", "+            and hidden_states.shape[0] >= _INPUT_NORM_QUANT_MIN_ROWS", "+            and forward_batch.num_token_non_padded is None", '+                and args.disaggregation_mode in ("prefill", "decode")'):
        assert guard in patch, guard
    # Decode: no pad slots, the router launch of select_experts with the graph-padding mask.
    assert "pad_slots=not decode)" in patch and "+    _post_norm_moe_input[(buf_x.shape[0] if pad_slots else rows,)](x," in patch
    assert "+        FUSE_DECODE_PADDING=num_token_non_padded is not None," in patch
    assert "+            bias = None if num_token_non_padded is not None else self._zero_router_bias" in patch
    # The Prefill call sites keep their defaults: pad slots on, no padding mask.
    assert "def post_norm_moe_input(x, residual, weight, eps, buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights, pad_slots=True):" in patch
    assert "def route_into_buffer(router_logits, zero_bias, buf_topk_idx, buf_topk_weights, num_token_non_padded=None):" in patch
