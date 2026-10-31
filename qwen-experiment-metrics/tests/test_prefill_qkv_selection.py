"""Fail-closed adoption and PP4 job wiring for the default-off QKV gate."""

import copy
import json
import runpy
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/select-firmus-prefill-qkv.py"
GATE = runpy.run_path(str(SCRIPT))
KEY = GATE["KEY"]
PIN = "SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen-pipe\n" + KEY + "=fp32\nDG_SM90_PREFILL_LEAN=1\n"


def numeric():
    metrics = {"passed": True, "fused_max_abs": 0.2, "fp32_max_abs": 0.3, "fused_mean_abs": 0.01, "fp32_mean_abs": 0.02, "fused_fp64_code_mismatch": 0.001, "fp32_fp64_code_mismatch": 0.002, "fused_fp32_code_mismatch": 0.003}
    return {"passed": True, "outputs": {name: dict(metrics) for name in GATE["OUTPUTS"]}}


def entry(worker, rank):
    row = {"initial": numeric(), "after_300_calls": numeric(), "endurance_calls": 300, "samples_us": {order: {"stock": [600.0] * 12, "fused": [590.0] * 12, "gemm": [520.0] * 12, "post": [80.0] * 12} for order in ("forward", "reverse")}}
    return {"schema": GATE["SCHEMA"], "run_id": "r11", "worker": f"prefill-{worker}", "rank": rank, "device": "NVIDIA H200", "uuid": f"GPU-{worker}-{rank}", "finished_at": time.time(), "elapsed_seconds": 45, "passed": True, "norm_weights": {"source": "staged-model"}, "rows": {str(m): copy.deepcopy(row) for m in (3994, 7988)}}


@pytest.fixture
def screen(tmp_path):
    directory = tmp_path / "results"
    for worker in range(3):
        out = directory / f"prefill-{worker}"
        out.mkdir(parents=True)
        for rank in range(4):
            (out / f"rank-{rank}.json").write_text(json.dumps(entry(worker, rank)))
    env = tmp_path / "prefill-kernels.env"
    env.write_text(PIN)
    return directory, env


def aggregate(screen, **kwargs):
    directory, env = screen
    return GATE["aggregate"](directory, env, kwargs.get("run_id", "r11"), kwargs.get("deadline", time.time() + 60), kwargs.get("workers_ok", True))


def edit(screen, change):
    path = screen[0] / "prefill-2/rank-3.json"
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record))


def test_all_twelve_pass_publish_once_and_keep_other_pins(screen):
    result = aggregate(screen)
    assert result["adopted"] and len(result["gpus"]) == 12
    assert screen[1].read_text() == PIN.replace("=fp32", "=fused")
    assert not list(screen[1].parent.glob("*.tmp"))
    assert not aggregate(screen)["adopted"]  # a second publication cannot append duplicate assignments
    summary = result["gpus"][0]["rows"]["7988"]["timing_summary"]
    assert summary["median_us"]["stock"] == 600
    assert summary["noise"]["fused"]["forward_median_us"] == 590
    assert summary["noise"]["fused"]["reverse_median_us"] == 590
    assert summary["noise"]["fused"]["order_delta_fraction"] == 0


@pytest.mark.parametrize("field,value", [("run_id", "old"), ("rank", 0), ("worker", "prefill-0"), ("device", "NVIDIA H100"), ("uuid", "GPU-0-0"), ("passed", False), ("elapsed_seconds", 120), ("finished_at", 1e20)])
def test_bad_identity_failure_or_late_result_keeps_fp32(screen, field, value):
    edit(screen, lambda record: record.update({field: value}))
    assert not aggregate(screen)["adopted"]
    assert screen[1].read_text() == PIN


@pytest.mark.parametrize("failure", ["timeout", "deadline", "missing", "malformed", "array", "missing-shape", "missing-phase", "model", "endurance"])
def test_partial_results_are_never_adopted(screen, failure):
    options = {}
    path = screen[0] / "prefill-2/rank-3.json"
    if failure == "timeout":
        options["workers_ok"] = False
    elif failure == "deadline":
        options["deadline"] = time.time() - 1
    elif failure == "missing":
        path.unlink()
    elif failure in ("malformed", "array"):
        path.write_text("{" if failure == "malformed" else "[]")
    elif failure == "missing-shape":
        edit(screen, lambda r: r["rows"].pop("3994"))
    elif failure == "missing-phase":
        edit(screen, lambda r: r["rows"]["3994"].pop("after_300_calls"))
    elif failure == "model":
        edit(screen, lambda r: r.update(norm_weights={"source": "random"}))
    else:
        edit(screen, lambda r: r["rows"]["7988"].update(endurance_calls=299))
    result = aggregate(screen, **options)
    assert not result["adopted"] and result["reason"]
    assert screen[1].read_text() == PIN


@pytest.mark.parametrize("name", GATE["OUTPUTS"])
@pytest.mark.parametrize("suffix", ["max_abs", "mean_abs", "fp64_code_mismatch"])
@pytest.mark.parametrize("phase", ["initial", "after_300_calls"])
def test_each_output_and_each_metric_must_be_no_worse(screen, name, suffix, phase):
    def worse(record):
        metrics = record["rows"]["3994"][phase]["outputs"][name]
        metrics[f"fused_{suffix}"] = metrics[f"fp32_{suffix}"] + 1e-12
    edit(screen, worse)
    assert not aggregate(screen)["adopted"]
    assert screen[1].read_text() == PIN


@pytest.mark.parametrize("value", [594.000001, 600, float("nan"), float("inf"), 0, -1, True])
def test_one_gpu_missing_margin_or_invalid_time_keeps_fp32(screen, value):
    edit(screen, lambda r: r["rows"]["7988"]["samples_us"]["reverse"].update(fused=[value] * 12))
    result = aggregate(screen)
    assert not result["adopted"]
    assert screen[1].read_text() == PIN


def test_exact_one_percent_boundary_passes(screen):
    edit(screen, lambda r: r["rows"]["7988"]["samples_us"]["reverse"].update(fused=[594.0] * 12))
    result = aggregate(screen)
    assert result["adopted"]
    noise = result["gpus"][-1]["rows"]["7988"]["timing_summary"]["noise"]["fused"]
    assert noise["forward_median_us"] == 590 and noise["reverse_median_us"] == 594
    assert noise["absolute_order_delta_us"] == 4
    assert noise["order_delta_fraction"] == pytest.approx(4 / 592)


def test_incomplete_timing_and_missing_tensor_fail(screen):
    edit(screen, lambda r: r["rows"]["3994"]["samples_us"]["forward"]["stock"].pop())
    assert not aggregate(screen)["adopted"]
    edit(screen, lambda r: r["rows"]["3994"]["initial"]["outputs"].pop("v_cache"))
    assert not aggregate(screen)["adopted"]
    assert screen[1].read_text() == PIN


def test_refusal_keeps_noise_and_all_available_gpu_records(screen):
    edit(screen, lambda r: r["rows"]["7988"]["samples_us"]["reverse"].update(fused=[610.0] * 12))
    result = aggregate(screen)
    assert not result["adopted"] and len(result["gpus"]) == 12
    assert result["gpus"][-1]["rows"]["7988"]["timing_summary"]["noise"]["fused"]["absolute_order_delta_us"] == 20


def test_publication_failure_keeps_pin_and_cleans_temporary(screen, monkeypatch):
    def fail(*args):
        raise OSError("fixture rename error")
    monkeypatch.setattr(Path, "replace", fail)
    assert not aggregate(screen)["adopted"]
    assert screen[1].read_text() == PIN
    assert not list(screen[1].parent.glob("*.tmp"))


def test_pp4_default_off_parallel_timeout_and_archive_wiring():
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    assert not config["prefill"].get("qkv_fused_gate", False)
    assert config["images"]["sglang"]["sha256"].startswith("e7b09add")
    sbatch = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    block = sbatch.split('# Default off. All twelve H200 GPUs', 1)[1].split('\nsrun --exact', 1)[0]
    assert '.prefill.qkv_fused_gate // false' in block and '"${prefill_pp}" == 4' in block
    assert 'test -s "${ROOT}/source/pp4-kernel-settings.json"' in block
    assert 'for worker in 0 1 2; do' in block and '--nproc-per-node=4' in block
    assert 'timeout --kill-after=2 120 srun' in block and 'prefill_qkv_started + 125' in block
    assert 'prefill_qkv_selections+=("$!")' in block
    assert 'wait "${pid}" || prefill_qkv_ok=0' in block
    assert '--workers-ok "${prefill_qkv_ok}"' in block
    assert '--env-output "${ROOT}/source/prefill-kernels.env"' in block
    assert '--model "${MODEL}"' in block and '=fused' not in block
    assert 'unset SGLANG_PREFILL_QKNORM' in sbatch
    assert 'if [[ "$role" == prefill ]]; then set -a; source "$(dirname "$0")/prefill-kernels.env"; set +a; fi' in sbatch
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    for name in ('scripts/select-firmus-prefill-qkv.py', 'kernels/probes/qkv-fused.py'):
        assert f'"${{GITHUB_WORKSPACE}}/{name}"' in provision
    workflow = (ROOT / '.github/workflows/firmus.yml').read_text()
    assert 'for part in logs profile results source; do' in workflow


def test_reference_checks_and_endurance_are_before_timing():
    source = SCRIPT.read_text()
    assert 'runpy.run_path(str(Path(__file__).with_name("qkv-fused.py")))' in source
    assert 'for _ in range(ENDURANCE)' in source
    assert 'data[4:6] = weights' in source
    assert source.index('for phase in ("initial", "after_300_calls")') < source.index('row["samples_us"] = measure')
    assert 'dist.barrier()' in source and 'torch.backends.cuda.matmul.allow_tf32 = False' in source


def test_timed_out_worker_steps_continue_to_fp32_serving(screen, tmp_path):
    import shlex
    import subprocess
    import sys

    root = tmp_path / "runtime"
    (root / "source").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "scripts").mkdir()
    (root / "source/pp4-kernel-settings.json").write_text("{}")
    env = root / "source/prefill-kernels.env"
    env.write_text(PIN)
    (root / "scripts" / SCRIPT.name).write_text(SCRIPT.read_text())
    sbatch = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    block = '# Default off. All twelve H200 GPUs' + sbatch.split('# Default off. All twelve H200 GPUs', 1)[1].split('\nsrun --exact', 1)[0]
    preamble = f'''set -euo pipefail
ROOT={shlex.quote(str(root))}
ART="$ROOT/logs"
prefill_pp=4
HEAD=head
TAIL=tail
SLURM_JOB_ID=fixture
SGLANG=unused
MODEL=unused
jq() {{ printf 'true\\n'; }}
mounts() {{ printf 'unused\\n'; }}
timeout() {{ return 124; }}
uv() {{ shift 5; {shlex.quote(sys.executable)} "$@"; }}
'''
    subprocess.run(["bash", "-c", preamble + block + '\nprintf serving-continues >"$ROOT/continued"\n'], check=True, capture_output=True, text=True)
    assert (root / "continued").read_text() == "serving-continues"
    assert env.read_text() == PIN
    result = json.loads((root / "results/prefill-qkv/selection.json").read_text())
    assert not result["adopted"] and "timed out" in result["reason"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numeric_json_can_still_publish_refusal(screen, value):
    edit(screen, lambda r: r["rows"]["7988"]["initial"]["outputs"]["q"].update(fused_max_abs=value))
    result = aggregate(screen)
    assert not result["adopted"] and len(result["gpus"]) == 12
    json.dumps(result, allow_nan=False)
    assert screen[1].read_text() == PIN
