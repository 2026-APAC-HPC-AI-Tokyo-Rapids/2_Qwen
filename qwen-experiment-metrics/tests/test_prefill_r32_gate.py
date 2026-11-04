"""All-twelve correctness and post-r32 pooled speed selection."""
import json
import runpy
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
G = runpy.run_path(str(ROOT / "scripts/firmus_prefill_fc1_first_k.py"))
PIN = "DG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_LEAN=1\nDG_SM90_PREFILL_SPLIT_ACCUM=1\nDG_SM90_PREFILL_FC1_FIRST_K=0\nKEEP=1\n"


@pytest.fixture
def screen(tmp_path):
    now = time.time()
    pairs = []
    for worker in range(3):
        directory = tmp_path / f"prefill-{worker}"
        directory.mkdir()
        for rank in range(4):
            identity = {"uuid": f"gpu-{worker}-{rank}", "run_id": "123-pair"}
            pairs.append(identity)
            record = {"schema": G["SCHEMA"], "uuid": identity["uuid"], "worker": directory.name, "rank": rank, "device": "NVIDIA H200", "run_id": "123-r32", "healthy": True, "image_sha256": "a" * 64, "header_sha256": "b" * 64, "routing_sha256": "c" * 64, "elapsed_seconds": 60, "finished_at": now, "exact": {name: dict.fromkeys(G["COMPONENTS"], True) for name in G["CASES"]}, "reference": {name: {mode: {"samples": 32, "finite": True, "relative_l2": 0.01, "max_scaled_error": 0.02} for mode in ("control", "candidate")} for name in G["CASES"]}, "timing": {name: {order: {"control": [2700.] * 12, "candidate": [2670.] * 12} for order in ("forward", "reverse")} for name in G["TIMED"]}, "endurance": {"calls": 1200, "buffers": 2, "input_versions": 2, "checks": {str(n): dict.fromkeys(G["ENDURANCE_CELLS"], True) for n in (600, 1200)}, "restored": [dict.fromkeys(G["COMPONENTS"], True) for _ in range(2)]}}
            (directory / f"rank-{rank}.json").write_text(json.dumps(record))
    pair = tmp_path / "pair.json"
    pair.write_text(json.dumps({"adopted": True, "gpus": pairs}))
    env = tmp_path / "prefill.env"
    env.write_text(PIN)
    return tmp_path, env, pair, now + 100


def aggregate(screen, workers_ok=True):
    directory, env, pair, deadline = screen
    return G["aggregate"](directory, env, pair, "123-r32", deadline, workers_ok, "a" * 64)


def test_complete_twelve_gpu_evidence_adopts(screen):
    assert aggregate(screen)["adopted"]
    assert screen[1].read_text() == PIN.replace("DG_SM90_PREFILL_FC1_FIRST_K=0\n", "") + "DG_SM90_PREFILL_FC1_FIRST_K=1\n"


@pytest.mark.parametrize("bad", ("missing", "uuid", "duplicate", "image", "source", "stale", "late", "healthy", "case", "component", "fp8", "scale", "partial", "reference", "l2", "scaled", "samples", "finite", "calls", "buffer", "input", "checkpoint", "cell", "midway", "final", "restore", "short", "order", "mode", "nan"))
def test_every_missing_or_failing_cell_rejects(screen, bad):
    path = screen[0] / "prefill-2/rank-3.json"
    value = json.loads(path.read_text())
    if bad == "missing":
        path.unlink()
    else:
        mutations = {
            "uuid": lambda r: r.update(uuid="other"),
            "duplicate": lambda r: r.update(uuid="gpu-0-0"),
            "image": lambda r: r.update(image_sha256="d" * 64),
            "source": lambda r: r.update(header_sha256="d" * 64),
            "stale": lambda r: r.update(run_id="122-r32"),
            "late": lambda r: r.update(finished_at=time.time() + 1000),
            "healthy": lambda r: r.update(healthy=False),
            "case": lambda r: r["exact"].pop("edges-high"),
            "component": lambda r: r["exact"]["historical"].pop("fc1_fp8"),
            "fp8": lambda r: r["exact"]["historical"].update(fc1_fp8=False),
            "scale": lambda r: r["exact"]["historical"].update(fc1_scale=False),
            "partial": lambda r: r["exact"]["historical"].update(fc2_bf16_partial=False),
            "reference": lambda r: r["reference"].pop("zero"),
            "l2": lambda r: r["reference"]["wide"]["control"].update(relative_l2=0.025),
            "scaled": lambda r: r["reference"]["zero"]["candidate"].update(max_scaled_error=0.08),
            "samples": lambda r: r["reference"]["hot"]["candidate"].update(samples=31),
            "finite": lambda r: r["reference"]["balanced"]["candidate"].update(finite=False),
            "calls": lambda r: r["endurance"].update(calls=1199),
            "buffer": lambda r: r["endurance"].update(buffers=1),
            "input": lambda r: r["endurance"].update(input_versions=1),
            "checkpoint": lambda r: r["endurance"]["checks"].pop("600"),
            "cell": lambda r: r["endurance"]["checks"]["1200"].pop(G["ENDURANCE_CELLS"][0]),
            "midway": lambda r: r["endurance"]["checks"]["600"].update({G["ENDURANCE_CELLS"][0]: False}),
            "final": lambda r: r["endurance"]["checks"]["1200"].update({G["ENDURANCE_CELLS"][-1]: False}),
            "restore": lambda r: r["endurance"]["restored"][1].update(fc2_bf16_partial=False),
            "short": lambda r: r["timing"]["balanced"]["forward"]["control"].pop(),
            "order": lambda r: r["timing"]["hot"].pop("reverse"),
            "mode": lambda r: r["timing"]["hot"]["reverse"].pop("candidate"),
            "nan": lambda r: r["timing"]["hot"]["reverse"].update(candidate=[float("nan")] * 12),
        }
        mutations[bad](value)
        path.write_text(json.dumps(value))
    assert not aggregate(screen)["adopted"]
    assert screen[1].read_text() == PIN


def test_worker_timeout_and_missing_p_reject(screen):
    assert not aggregate(screen, False)["adopted"]
    value = json.loads(screen[2].read_text())
    value["gpus"].pop()
    screen[2].write_text(json.dumps(value))
    assert not aggregate(screen)["adopted"]
    assert screen[1].read_text() == PIN


def test_post_greedy_budget_and_staging():
    batch = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    assert batch.index("# Decode greedy gate end.") < batch.index("# Round 32 candidate / round 33 pooled gate;") < batch.index('bash "${ROOT}/source/worker.sh" prefill-0')
    assert "SECONDS + 105 + 660 <= 1020" in batch
    assert 'env DG_JIT_CACHE_DIR="/tmp/sglang-r32-' in batch and 'TRITON_CACHE_DIR="/tmp/sglang-r32-' in batch
    assert 'TVM_FFI_CACHE_DIR="/tmp/sglang-r32-' in batch
    assert '--env-output "${ROOT}/source/prefill-kernels.r32.env"' in batch
    assert 'mv "${ROOT}/source/prefill-kernels.r32.env" "${ROOT}/source/prefill-kernels.env"' in batch
    assert 'prefill-r32-dispatch-${worker}.txt' in batch
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    assert '"${GITHUB_WORKSPACE}/scripts/firmus_prefill_fc1_first_k.py" "${runtime_dir}/scripts/"' in provision
    assert '"${GITHUB_WORKSPACE}/kernels/probes/megamoe-local-ep1-routing.npz" "${runtime_dir}/source/"' in provision


def test_round32_saved_measurements_pass_pooled_rule(screen):
    fixture = json.loads((ROOT / "qwen-experiment-metrics/tests/fixtures/prefill-r32-timing.json").read_text())
    for gpu in fixture["gpus"]:
        path = screen[0] / gpu["worker"] / f"rank-{gpu['rank']}.json"
        record = json.loads(path.read_text())
        record["timing"] = gpu["timing"]
        path.write_text(json.dumps(record))
    result = aggregate(screen)
    assert result["adopted"]
    expected = {"balanced": (24.4559645652771, 19.831955432891846, 24.559974670410156, 12, 8.43191146850586), "hot": (16.911983489990234, 18.016040325164795, 17.416059970855713, 12, 8.880019187927246), "historical": (16.552090644836426, 16.575992107391357, 17.216026782989502, 11, -3.056049346923828)}
    for name, (pooled, forward, reverse, positive, worst) in expected.items():
        summary = result["timing_summary"][name]
        assert summary["saving_median_us"] == pytest.approx({"pooled": pooled, "forward": forward, "reverse": reverse})
        assert summary["positive_gpus"] == positive
        assert summary["minimum_saving_us"] == pytest.approx(worst)
    # Replaying timings does not turn the original failed worker exits into success.
    assert not aggregate(screen, workers_ok=False)["adopted"]


def samples(control, forward, reverse=None):
    return {order: {"control": [control] * 12, "candidate": [control - saving] * 12} for order, saving in (("forward", forward), ("reverse", forward if reverse is None else reverse))}


@pytest.mark.parametrize("control,saving,passes", ((100., 5., True), (100., 4.999, False), (2700., 13.5, True), (2700., 13.499, False)))
def test_pooled_absolute_and_relative_thresholds(control, saving, passes):
    rows = [samples(control, saving) for _ in range(12)]
    if passes:
        G["pooled_margin"](rows)
    else:
        with pytest.raises(AssertionError, match="below pooled"):
            G["pooled_margin"](rows)


@pytest.mark.parametrize("positive,loss,passes", ((9, -27., True), (8, -27., False), (9, -27.001, False), (9, 0., True)))
def test_positive_count_and_one_percent_boundary(positive, loss, passes):
    rows = [samples(2700., 30. if i < positive else loss) for i in range(12)]
    if passes:
        G["pooled_margin"](rows)
    else:
        with pytest.raises(AssertionError):
            G["pooled_margin"](rows)


@pytest.mark.parametrize("order", ("forward", "reverse"))
def test_pooled_gain_cannot_hide_nonpositive_order(order):
    rows = [samples(2700., 0., 60.) if order == "forward" else samples(2700., 60., 0.) for _ in range(12)]
    with pytest.raises(AssertionError, match="nonpositive order"):
        G["pooled_margin"](rows)


def test_pool_samples_within_each_gpu_before_cross_gpu_median():
    row = {"forward": {"control": [100.] * 12, "candidate": [40.] * 12}, "reverse": {"control": [300.] * 12, "candidate": [280.] * 12}}
    assert G["timing_medians"](row)["pooled"] == {"control": 200., "candidate": 160.}
    assert G["pooled_margin"]([row] * 12)["saving_median_us"]["pooled"] == 40.


@pytest.mark.parametrize("fail", (False, True))
def test_cleanup_releases_all_views_and_ep1_before_world_and_publication(tmp_path, fail):
    calls = []
    group = object()
    fields = ("topk_idx", "topk_weights", "l1_acts", "l1_acts_sf", "l2_acts", "l2_acts_sf")

    class Buffer:
        def __init__(self):
            self.group = group
            for field in fields:
                setattr(self, field, object())

        def destroy(self):
            assert all(getattr(self, field) is None for field in fields)
            calls.append("buffer")
            self.group = None

    def destroy(pg=None):
        assert not (tmp_path / "rank-0.json").exists()
        calls.append("WORLD" if pg is None else "EP1")
        if pg is None and fail:
            raise RuntimeError("cleanup failed")

    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: calls.append("sync")))
    dist = SimpleNamespace(barrier=lambda: calls.append("barrier"), destroy_process_group=destroy)
    record = {"healthy": False}
    if fail:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            G["finish"](torch, dist, (Buffer(), Buffer()), record, tmp_path, 0, time.monotonic())
        assert record["healthy"] is False and not (tmp_path / "rank-0.json").exists()
    else:
        G["finish"](torch, dist, (Buffer(), Buffer()), record, tmp_path, 0, time.monotonic())
        assert json.loads((tmp_path / "rank-0.json").read_text())["healthy"] is True
    assert calls == ["sync", "barrier", "buffer", "buffer", "EP1", "barrier", "WORLD"]


def test_round32_budget_with_head_container_delay():
    fixture = json.loads((ROOT / "qwen-experiment-metrics/tests/fixtures/prefill-r32-timing.json").read_text())
    # First Slurm log 07:42:11+08, gate run_id 07:46:21+08.
    elapsed = fixture["gate_started_at"] - fixture["budget"]["slurm_first_log_at"]
    assert G["SECONDS"] == 95
    assert elapsed + 105 + 660 == 1015
    assert elapsed + 105 + 660 <= 1020
    # READY follows router startup on the same HEAD node. Allow one second
    # each for timestamp truncation and the shell SECONDS counter.
    upper = fixture["gate_started_at"] + 1 - fixture["budget"]["router_started_at"] + fixture["budget"]["ready_seconds"] + 2
    assert upper == 255 and upper + 105 + 660 <= 1020
    # HEAD torchrun appears after 33 s; rank body starts after about 36 s.
    for r in fixture["gpus"]:
        assert r["finished_at"] - fixture["gate_started_at"] < 56
        assert r["finished_at"] - fixture["gate_started_at"] < G["SECONDS"]
    batch = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    block = batch[batch.index("# Round 32 candidate / round 33 pooled gate;"):batch.index('bash "${ROOT}/source/worker.sh" prefill-0')]
    assert block.count("prefill_fc1_first_k_started + 95") == 2
    assert block.count("prefill_fc1_first_k_started + 105") == 2


def test_round32_pin_is_consistent_with_archived_provenance():
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    image = "3a47b757bb88f6bc14ad5f151e8405f645007f25f381f631d88046d9bd1b7b04"
    assert config["images"]["sglang"] == {"path": "/scratch/home/ACCOUNT/qwen/containers/41729881283.sqsh", "sha256": image, "size_bytes": 12296638464, "provenance": "/scratch/home/ACCOUNT/qwen/containers/41729881283.container-sidecar.json"}
    assert config["container_build_run_id"] == 41729881283
    assert config["container_sha256"] == image and config["container_size_bytes"] == 12296638464
    assert config["container_source_commit"] == "3f9124fd8afb28e34c71288b6dc31e5c490c7f60"
    assert runpy.run_path(str(ROOT / "scripts/check-firmus-pp4-cpu.py"))["PINNED_IMAGE"] == image


def test_relative_threshold_uses_per_gpu_saving_fractions():
    # A ratio of cross-GPU medians would pass despite every GPU losing its
    # required relative margin. Normalize by each GPU's own control median.
    rows = [samples(10000., 49.) for _ in range(4)] + [samples(100000., 499.) for _ in range(3)] + [samples(100., 90.) for _ in range(5)]
    with pytest.raises(AssertionError, match="below pooled"):
        G["pooled_margin"](rows)
