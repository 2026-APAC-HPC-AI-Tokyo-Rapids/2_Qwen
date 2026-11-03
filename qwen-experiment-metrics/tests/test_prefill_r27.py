"""Reject partial evidence and preserve independent post-greedy selections."""
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("r27", ROOT / "scripts/select-firmus-prefill-r27.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
poll, norm = gate.poll, gate.norm


def samples(control, candidate):
    return {order: {"control": [control] * 12, "candidate": [candidate] * 12} for order in ("forward", "reverse")}


def fixture(tmp_path):
    prior = tmp_path / "pair.json"
    prior.write_text(json.dumps({"adopted": True, "gpus": [{"uuid": f"uuid-{i}", "run_id": "123-100"} for i in range(12)]}))
    env = tmp_path / "prefill.env"
    env.write_text("DG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_LEAN=1\nDG_SM90_PREFILL_SPLIT_ACCUM=1\n" + poll.KEY + "=0\n" + norm.KEY + "=0\n")
    for i in range(12):
        worker, rank = divmod(i, 4)
        common = {"run_id": "123-200", "rank": rank, "worker": f"prefill-{worker}", "uuid": f"uuid-{i}", "device": "NVIDIA H200", "image_sha256": "a" * 64, "healthy": True, "elapsed_seconds": 20, "finished_at": 240}
        a = {**common, "schema": poll.SCHEMA, "header_sha256": "b" * 64, "routing_sha256": "c" * 64, "exact": {case: dict.fromkeys(poll.COMPONENTS, True) for case in poll.CASES}, "timing": {case: samples(2600, 2580 if case == "historical" else 2600) for case in poll.TIMED}, "endurance": {"calls": 1200, "buffers": 2, "input_versions": 2, "outputs": dict.fromkeys(poll.TIMED + ("edges-low", "zero", "wide"), True), "restored": [dict.fromkeys(poll.COMPONENTS, True) for _ in range(2)]}}
        b = {**common, "schema": norm.SCHEMA, "source_sha256": "d" * 64, "accuracy": {case: {"residual_exact": True, "finite": True, "control": {"max_abs_error": .1, "fp8_code_mismatches": 5}, "candidate": {"max_abs_error": .1, "fp8_code_mismatches": 4}} for case in norm.CASES}, "timing": {str(n): samples(60, 58) for n in norm.ROWS}, "endurance": {"calls": 1200, "buffers": 2, "input_versions": 2, "exact": {"0": True, "1": True}}}
        for name, r in (("poll-backoff", a), ("norm-square-fma", b)):
            target = tmp_path / name / f"prefill-{worker}" / f"rank-{rank}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(r))
    return SimpleNamespace(aggregate=tmp_path, output=tmp_path / "selection.json", env_output=env, pair_selection=prior, run_id="123-200", deadline=270, image_sha256="a" * 64, workers_ok=1, poll_backoff=True, norm_square_fma=True)


def mutate(args, name, function):
    p = args.aggregate / name / "prefill-2/rank-3.json"
    r = json.loads(p.read_text())
    function(r)
    p.write_text(json.dumps(r))


def decide(args, monkeypatch):
    monkeypatch.setattr(gate.time, "time", lambda: 250)
    gate.aggregate(args)
    return json.loads(args.output.read_text())


def test_independent_success_and_no_temporary_env(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    result = decide(args, monkeypatch)
    assert all(v["adopted"] for v in result.values())
    assert poll.KEY + "=64" in args.env_output.read_text()
    assert norm.KEY + "=1" in args.env_output.read_text()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("name", ("poll-backoff", "norm-square-fma"))
@pytest.mark.parametrize("failure", ("missing", "wrong-uuid", "wrong-run", "wrong-image", "late", "unhealthy"))
def test_one_candidate_failure_keeps_other(tmp_path, monkeypatch, name, failure):
    args = fixture(tmp_path)
    if failure == "missing":
        (tmp_path / name / "prefill-2/rank-3.json").unlink()
    else:
        key, value = {"wrong-uuid": ("uuid", "uuid-0"), "wrong-run": ("run_id", "old"), "wrong-image": ("image_sha256", "wrong"), "late": ("finished_at", 271), "unhealthy": ("healthy", False)}[failure]
        mutate(args, name, lambda r: r.update({key: value}))
    result = decide(args, monkeypatch)
    assert not result[name]["adopted"]
    assert result["norm-square-fma" if name == "poll-backoff" else "poll-backoff"]["adopted"]


def test_worker_timeout_does_not_discard_other_complete_candidate(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    args.workers_ok = 0
    mutate(args, "poll-backoff", lambda r: r.update(healthy=False))
    result = decide(args, monkeypatch)
    assert not result["poll-backoff"]["adopted"] and result["norm-square-fma"]["adopted"]


@pytest.mark.parametrize("failure", ("byte", "endurance", "historical", "hot", "balanced", "nan", "reverse"))
def test_poll_rejects_accuracy_and_any_gpu_routing_failure(tmp_path, monkeypatch, failure):
    args = fixture(tmp_path)
    def corrupt(r):
        if failure == "byte":
            r["exact"]["saved-35"]["fc2_bf16_partial"] = False
        elif failure == "endurance":
            r["endurance"]["calls"] = 1199
        elif failure == "nan":
            r["timing"]["historical"]["forward"]["candidate"][0] = float("nan")
        else:
            shape = "historical" if failure == "reverse" else failure
            r["timing"][shape]["reverse"]["candidate"] = [2600.01] * 12
    mutate(args, "poll-backoff", corrupt)
    assert not decide(args, monkeypatch)["poll-backoff"]["adopted"]


@pytest.mark.parametrize("failure", ("max_abs_error", "fp8_code_mismatches", "residual", "endurance", "reverse", "missing-shape"))
def test_norm_noninferiority_and_every_shape_order(tmp_path, monkeypatch, failure):
    args = fixture(tmp_path)
    def corrupt(r):
        if failure in ("max_abs_error", "fp8_code_mismatches"):
            r["accuracy"]["fp8-boundary"]["candidate"][failure] = 99
        elif failure == "residual":
            r["accuracy"]["extreme"]["residual_exact"] = False
        elif failure == "endurance":
            r["endurance"]["exact"]["1"] = False
        elif failure == "reverse":
            r["timing"]["7988"]["reverse"] = samples(60, 59.9)["reverse"]
        else:
            del r["timing"]["2049"]
    mutate(args, "norm-square-fma", corrupt)
    assert not decide(args, monkeypatch)["norm-square-fma"]["adopted"]


@pytest.mark.parametrize(("extra", "adopted"), ((0, True), (1, False)))
def test_norm_allows_fp8_tie_noise_but_not_a_larger_code_regression(tmp_path, monkeypatch, extra, adopted):
    args = fixture(tmp_path)
    mutate(args, "norm-square-fma", lambda r: r["accuracy"]["fp8-boundary"]["candidate"].update(fp8_code_mismatches=5 + norm.MISMATCH_SLACK + extra))
    assert decide(args, monkeypatch)["norm-square-fma"]["adopted"] is adopted


def test_expired_budget_leaves_flags_off(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    args.deadline = 249
    before = args.env_output.read_bytes()
    assert all(not v["adopted"] for v in decide(args, monkeypatch).values())
    assert args.env_output.read_bytes() == before


def test_greedy_precedes_r27_and_actual_budget_expressions():
    text = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    assert text.index("# Decode greedy gate end.") < text.index("# r27 begins") < text.index('bash "${ROOT}/source/worker.sh" prefill-0')
    greedy = text[text.index("# Decode greedy gate begin:"):text.index("# Decode greedy gate end.")]
    r27 = text[text.index("# r27 begins"):text.index("# r27 gate end.")]
    assert "1020 - SECONDS >= 90 + 660" in greedy
    assert "SECONDS + 100 + 660 <= 1020" in r27
    assert "prefill_r27_started + 90" in r27 and "prefill_r27_started + 100" in r27
    # Replay measured r25 times and exact boundaries using the real expressions.
    for now, expected in ((208, True), (222, True), (236, True), (260, True), (261, False)):
        expr = r27.split("if (( ", 1)[1].split(" ));", 1)[0]
        result = subprocess.run(["bash", "-c", f"SECONDS={now}; (( {expr} ))"], check=False)
        assert (result.returncode == 0) is expected
    expr = greedy.split("if (( ", 1)[1].split(" ));", 1)[0]
    assert subprocess.run(["bash", "-c", f"SECONDS=208; (( {expr} ))"], check=False).returncode == 0


def test_config_pin_flags_dispatch_and_provision_contracts():
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    assert config["prefill"]["poll_backoff_gate"] is config["prefill"]["norm_square_fma_gate"] is True
    assert config["images"]["sglang"]["sha256"].startswith("a001f901")
    assert config["deep_gemm_overlay_patches"][-1] == "patches/deep-gemm-overlay/sm90-prefill-poll-backoff.patch"
    host = (ROOT / "patches/deep-gemm-host/sm90-prefill-poll-backoff.patch").read_text()
    assert "prefill_split_accum ? get_env<int>" in host and "static bool prefill_poll_logged = false" in host
    patch = (ROOT / "patches/sglang/prefill-norm-square-fma.patch").read_text()
    assert "enable_fp_fusion=False" in patch and "sums = tl.fma(value, value, sums)" in patch
    assert "square_sum = tl.sum(h * h, 0)" in patch and "@lru_cache(maxsize=1)" in patch
    assert "_post_norm_moe_input" not in patch
    sbatch = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    assert "prefill-r27-dispatch-${worker}.txt" in sbatch and '"${ART}/prefill-${worker}.log"' in sbatch
    assert "DeepGEMM sm90 Prefill polling backoff enabled:.*|Prefill norm_quant square FMA enabled:.*" in sbatch
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    for script in ("select-firmus-prefill-r27.py", "firmus_prefill_poll_backoff.py", "firmus_prefill_norm_fma.py", "megamoe-local-ep1-routing.npz"):
        assert script in provision


def test_sass_evidence_covers_all_24_and_spill_free_candidate():
    report = json.loads((ROOT / "research/runtime-performance/prefill-backoff-r27-cuda-validation-20261103.json").read_text())
    assert report["off_sass_identical_instantiations"] == 24
    for i in range(24):
        assert report["builds"][f"before-{i}"]["sass_sha256"] == report["builds"][f"off-{i}"]["sass_sha256"]
    candidate = report["builds"]["candidate-7"]
    assert candidate["spill_store_load"] == [["0", "0"]] and candidate["nanosleep_instructions"] > 0


def test_no_backoff_changes_to_mbarrier_or_ordering():
    patch = (ROOT / "patches/deep-gemm-overlay/sm90-prefill-poll-backoff.patch").read_text()
    added = "\n".join(line for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    assert "dg_poll_failures < 3" in added and "__nanosleep(DG_SM90_PREFILL_POLL_BACKOFF)" in added
    assert "#define DG_PREFILL_POLL_STATE\n" in added
    assert not any(word in added for word in ("mbarrier", "red_add", "red_or", "fence."))
    assert "ld_volatile" in patch and "ld_acq_gpu" in patch and "ld_acq_sys" in patch
    assert poll.CALLS == norm.CALLS == 1200


def test_norm_validator_does_not_mutate_evidence(tmp_path):
    args = fixture(tmp_path)
    r = json.loads((args.aggregate / "norm-square-fma/prefill-0/rank-0.json").read_text())
    before = copy.deepcopy(r)
    norm.validate(r)
    assert r == before


def test_norm_compilation_and_once_per_process_log(monkeypatch, tmp_path):
    import functools
    import os

    report = json.loads((ROOT / "research/runtime-performance/prefill-norm-fma-r27-triton-validation-20261103.json").read_text())
    before, off, candidate = (report["builds"][name] for name in ("before", "off", "candidate"))
    assert before["sass_sha256"] == off["sass_sha256"]
    assert candidate["registers"] == before["registers"] == 64
    assert candidate["spills"] == candidate["local_load_store"] == 0
    assert candidate["fma_instructions"] - before["fma_instructions"] == 15
    assert candidate["fmul_instructions"] - before["fmul_instructions"] == -15
    assert candidate["fadd_instructions"] - before["fadd_instructions"] == -15
    patch = (ROOT / "patches/sglang/prefill-norm-square-fma.patch").read_text()
    start = patch.index("+@lru_cache(maxsize=1)")
    end = patch.index(" @triton.jit", start)
    source = "\n".join(line[1:] for line in patch[start:end].splitlines() if line.startswith("+"))
    entries = []
    class Logger:
        def info(self, value):
            entries.append(value)
    file = tmp_path / "norm_fixture.py"
    file.write_text(source)
    spec = importlib.util.spec_from_file_location("norm_fixture", file)
    module = importlib.util.module_from_spec(spec)
    module.os = os
    module.logging = SimpleNamespace(getLogger=lambda _: Logger())
    module.lru_cache = functools.lru_cache
    spec.loader.exec_module(module)
    function = module._prefill_square_fma
    monkeypatch.delenv(norm.KEY, raising=False)
    assert function() is False and entries == []
    function.cache_clear()
    monkeypatch.setenv(norm.KEY, "1")
    assert function() is function() is True
    assert len(entries) == 1 and entries[0].startswith("Prefill norm_quant square FMA enabled:")


def test_preparation_keeps_prior_hashes_and_prefill_flags_off(tmp_path):
    import runpy

    (tmp_path / "source").mkdir()
    (tmp_path / "results").mkdir()
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    prepare = runpy.run_path(str(ROOT / "scripts/prepare-firmus-pp4-settings.py"))["prepare"]
    prepare(config, ROOT, tmp_path)
    prefill = (tmp_path / "source/prefill-kernels.env").read_text()
    decode = (tmp_path / "source/decode-kernels.env").read_text()
    assert poll.KEY + "=0" in prefill and norm.KEY + "=0" in prefill
    assert poll.KEY not in decode and norm.KEY not in decode
    payload = json.loads((tmp_path / "source/pp4-kernel-settings.json").read_text())
    assert payload["poll_backoff_gate"] is payload["norm_square_fma_gate"] is True
