"""All-or-nothing r32 GPU evidence and the post-greedy serving budget."""
import json
import runpy
import time
from pathlib import Path

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


@pytest.mark.parametrize("bad", ("missing", "uuid", "duplicate", "image", "source", "stale", "late", "healthy", "case", "component", "fp8", "scale", "partial", "reference", "l2", "scaled", "samples", "finite", "calls", "buffer", "input", "checkpoint", "cell", "midway", "final", "restore", "short", "order", "mode", "relative-margin", "absolute-margin", "nan"))
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
            "relative-margin": lambda r: r["timing"]["historical"]["reverse"].update(candidate=[2690.] * 12),
            "absolute-margin": lambda r: r["timing"]["balanced"]["forward"].update(control=[100.] * 12, candidate=[96.] * 12),
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
    assert batch.index("# Decode greedy gate end.") < batch.index("# Round 32:") < batch.index('bash "${ROOT}/source/worker.sh" prefill-0')
    assert "SECONDS + 100 + 660 <= 1020" in batch
    assert 'env DG_JIT_CACHE_DIR="/tmp/sglang-r32-' in batch and 'TRITON_CACHE_DIR="/tmp/sglang-r32-' in batch
    assert 'TVM_FFI_CACHE_DIR="/tmp/sglang-r32-' in batch
    assert '--env-output "${ROOT}/source/prefill-kernels.r32.env"' in batch
    assert 'mv "${ROOT}/source/prefill-kernels.r32.env" "${ROOT}/source/prefill-kernels.env"' in batch
    assert 'prefill-r32-dispatch-${worker}.txt' in batch
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    assert '"${GITHUB_WORKSPACE}/scripts/firmus_prefill_fc1_first_k.py" "${runtime_dir}/scripts/"' in provision
    assert '"${GITHUB_WORKSPACE}/kernels/probes/megamoe-local-ep1-routing.npz" "${runtime_dir}/source/"' in provision
