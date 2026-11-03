"""CPU geometry and refusal tests for the opt-in r26 in-job gate."""
import json
import runpy
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CPU = runpy.run_path(str(ROOT / "scripts/check-prefill-megamoe-swap-tail.py"))
GATE = runpy.run_path(str(ROOT / "scripts/firmus_prefill_swap_tail.py"))
PIN = "DG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_LEAN=1\nDG_SM90_PREFILL_SPLIT_ACCUM=1\nOTHER=retained\n"


@pytest.mark.parametrize("valid", range(1, 129))
def test_tail_layout_and_boundaries(valid):
    assert CPU["width"](valid) == (16 if valid <= 16 else 32 if valid <= 32 else 0)
    assert CPU["check_coordinates"](valid) == valid * 384


def test_saved_and_boundary_schedules():
    saved = CPU["load_schedule"]().archived_distributions()
    assert len(saved) == 36
    result = CPU["check_schedule"](saved)
    assert result["eligible_pool_blocks"] == 2090
    assert result["saved_rows"] == 92144
    edges = [[0, r, r + 64, r + 128] * 32 for r in range(1, 65)]
    assert CPU["check_schedule"](edges)["eligible_pool_blocks"] == 2048


def test_independent_scale_swizzle_and_fp32_reference():
    pytest.importorskip("numpy")
    assert CPU["check_scales_and_swizzle"]() == 425984
    result = CPU["numerical"]()
    assert result["cases"] == 384 and result["max_abs_error_to_independent_fp32_numpy"] < 3e-6
    assert result["k_numeric"] == [256, 1536, 4096]
    assert result["exact_zero_input_included"] is True
    assert result["tensor_core_internal_order_proved"] is False


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
            record = {"schema": GATE["SCHEMA"], "uuid": identity["uuid"], "worker": directory.name, "rank": rank, "device": "NVIDIA H200", "run_id": "123-swap", "healthy": True, "image_sha256": "a" * 64, "header_sha256": "b" * 64, "routing_sha256": "c" * 64, "elapsed_seconds": 90, "finished_at": now, "exact": {name: dict.fromkeys(GATE["COMPONENTS"], True) for name in GATE["CASES"]}, "timing": {name: {order: {"control": [2700.] * 12, "candidate": [2670.] * 12} for order in ("forward", "reverse")} for name in GATE["TIMED"]}, "endurance": {"calls": 1200, "buffers": 2, "input_versions": 2, "outputs": dict.fromkeys(GATE["TIMED"] + ("edges-low", "zero", "wide"), True), "restored": [dict.fromkeys(GATE["COMPONENTS"], True) for _ in range(2)]}}
            (directory / f"rank-{rank}.json").write_text(json.dumps(record))
    pair = tmp_path / "pair.json"
    pair.write_text(json.dumps({"adopted": True, "gpus": pairs}))
    env = tmp_path / "prefill.env"
    env.write_text(PIN)
    return tmp_path, env, pair, now + 130


def aggregate(screen):
    directory, env, pair, deadline = screen
    return GATE["aggregate"](directory, env, pair, "123-swap", deadline, True, "a" * 64)


def test_adopts_complete_twelve_gpu_gate(screen):
    assert aggregate(screen)["adopted"] is True
    assert screen[1].read_text() == PIN + "DG_SM90_PREFILL_SWAP_TAIL=1\n"


@pytest.mark.parametrize("bad", ["missing-rank", "uuid", "duplicate", "image", "header", "routing", "stale", "late", "unhealthy", "missing-case", "component", "fp8", "scale", "partial", "endurance", "restore", "input", "buffer", "short-timing", "reverse", "fraction-margin", "nan", "missing-order", "missing-kind"])
def test_every_incomplete_or_regressing_gate_retains_control(screen, bad):
    directory, env, _, _ = screen
    path = directory / "prefill-2/rank-3.json"
    value = json.loads(path.read_text())
    if bad == "missing-rank":
        path.unlink()
    else:
        mutations = {
            "uuid": lambda r: r.update(uuid="other"),
            "duplicate": lambda r: r.update(uuid="gpu-0-0"),
            "image": lambda r: r.update(image_sha256="d" * 64),
            "header": lambda r: r.update(header_sha256="d" * 64),
            "routing": lambda r: r.update(routing_sha256="d" * 64),
            "stale": lambda r: r.update(run_id="122-swap"),
            "late": lambda r: r.update(finished_at=time.time() + 1000),
            "unhealthy": lambda r: r.update(healthy=False),
            "missing-case": lambda r: r["exact"].pop("edges-high"),
            "component": lambda r: r["exact"]["historical"].pop("fc1_fp8"),
            "fp8": lambda r: r["exact"]["historical"].update(fc1_fp8=False),
            "scale": lambda r: r["exact"]["historical"].update(fc1_scale=False),
            "partial": lambda r: r["exact"]["historical"].update(fc2_bf16_partial=False),
            "endurance": lambda r: r["endurance"].update(calls=1199),
            "restore": lambda r: r["endurance"]["restored"][1].update(fc2_bf16_partial=False),
            "input": lambda r: r["endurance"].update(input_versions=1),
            "buffer": lambda r: r["endurance"].update(buffers=1),
            "short-timing": lambda r: r["timing"]["balanced"]["forward"]["control"].pop(),
            "reverse": lambda r: r["timing"]["balanced"]["reverse"].update(candidate=[2710.] * 12),
            "fraction-margin": lambda r: r["timing"]["balanced"]["reverse"].update(candidate=[2694.] * 12),
            "nan": lambda r: r["timing"]["balanced"]["reverse"].update(candidate=[float("nan")] * 12),
            "missing-order": lambda r: r["timing"]["hot"].pop("reverse"),
            "missing-kind": lambda r: r["timing"].pop("historical"),
        }
        mutations[bad](value)
        path.write_text(json.dumps(value))
    assert aggregate(screen)["adopted"] is False
    assert env.read_text() == PIN


def test_uuid_sets_allow_worker_reassignment(screen):
    directory = screen[0]
    paths = [directory / f"prefill-{w}/rank-{r}.json" for w in range(3) for r in range(4)]
    rows = [json.loads(p.read_text()) for p in paths]
    uuids = [r["uuid"] for r in rows]
    for i, path in enumerate(paths):
        rows[i]["uuid"] = uuids[(i + 4) % 12]
        path.write_text(json.dumps(rows[i]))
    assert aggregate(screen)["adopted"] is True


def test_missing_p_prerequisite_refuses(screen):
    value = json.loads(screen[2].read_text())
    value["gpus"].pop()
    screen[2].write_text(json.dumps(value))
    assert aggregate(screen)["adopted"] is False
    assert screen[1].read_text() == PIN


def test_budget_and_order_before_the_greedy_gate():
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    assert config["prefill"]["megamoe_swap_tail_gate"] is True
    batch = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    assert "1020 - SECONDS >= 125 + 660" in batch
    # Round 25 reached the greedy gate's end at about 236 s, past this gate's 235 s limit, and the greedy
    # gate holds all eight g4 devices, so this gate runs between prepack and the greedy gate.
    assert batch.index("# r26:") < batch.index("# Decode greedy gate begin") < batch.index('bash "${ROOT}/source/worker.sh" prefill-0')


def test_serving_swap_tail_dispatch_is_logged_and_recorded():
    host = (ROOT / "patches/deep-gemm-host/sm90-prefill-swap-tail.patch").read_text()
    assert 'fprintf(stderr, "DeepGEMM sm90 Prefill MegaMoE swap tail enabled:' in host
    assert "prefill_swap_tail_logged = true;" in host
    batch = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    record = batch.index("results/prefill-megamoe-swap-tail-dispatch.txt")
    assert batch.index('test -s "${ROOT}/results/score.json"') < record < batch.index("collect-eagle-acceptance.py")
    assert "'DeepGEMM sm90 Prefill MegaMoE swap tail enabled.*'" in batch
