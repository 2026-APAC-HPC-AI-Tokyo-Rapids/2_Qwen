"""GPU-free checks of the multi-candidate Firmus kernel bench staging and summary."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import runpy
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEADER = "opt/sglang/venv/lib/python3.12/site-packages/deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"
JIT = "opt/sglang/source/python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh"
UNSQUASHFS = """#!/usr/bin/env bash
set -euo pipefail
while [[ $1 != -d ]]; do shift; done
dest=$2 image=$3
shift 3
for path in "$@"; do mkdir -p "${dest}/$(dirname "${path}")"; cp -a "$(dirname "${image}")/rootfs/${path}" "${dest}/${path}"; done
"""


def make_patch(tmp_path: Path, relative: str, before: str, after: str) -> str:
    repo = tmp_path / "patch-repo"
    shutil.rmtree(repo, ignore_errors=True)
    (repo / relative).parent.mkdir(parents=True)
    (repo / relative).write_text(before)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "base"], check=True)
    (repo / relative).write_text(after)
    return subprocess.run([*git, "diff"], check=True, capture_output=True, text=True).stdout


def workspace(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    header = "".join(f"line {index}\n" for index in range(1, 12))
    jit = "".join(f"jit {index}\n" for index in range(1, 8))
    rootfs = tmp_path / "containers/rootfs"
    for relative, text in ((HEADER, header), (JIT, jit)):
        (rootfs / relative).parent.mkdir(parents=True, exist_ok=True)
        (rootfs / relative).write_text(text)
    (rootfs / "opt/apac").mkdir(parents=True)
    (rootfs / "opt/apac/sm90-megamoe-build.json").write_text(json.dumps({"header_sha256": hashlib.sha256(header.encode()).hexdigest()}))
    # The real image also supplies this extracted record. These fixture
    # candidates do not enable the independent-WG host switch.
    (rootfs / "opt/apac/deep-gemm-host-build.json").write_text(json.dumps({"abi": "tvm-ffi", "patches": {}}))
    image = tmp_path / "containers/image.sqsh"
    image.write_bytes(b"fake squashfs")
    ws = tmp_path / "ws"
    (ws / "configs").mkdir(parents=True)
    (ws / "configs/firmus-framework.json").write_text(json.dumps({"images": {"sglang": {"path": str(image), "sha256": hashlib.sha256(b"fake squashfs").hexdigest(), "size_bytes": image.stat().st_size}}}))
    (ws / "scripts").mkdir()
    for script in ("provision-firmus-kernel-bench.sh", "bench-firmus-megamoe.py", "check-firmus-megamoe.py", "check-firmus-decode-kernels.py", "firmus_megamoe_replicas.py"):
        shutil.copy(ROOT / "scripts" / script, ws / "scripts" / script)
    (ws / "platforms/firmus/kernel-bench").mkdir(parents=True)
    shutil.copy(ROOT / "platforms/firmus/kernel-bench/run.sbatch", ws / "platforms/firmus/kernel-bench/run.sbatch")
    relative = "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"
    timeline = ws / "patches/deep-gemm-timeline/sm90-mega-moe-timeline.patch"
    timeline.parent.mkdir(parents=True)
    timeline.write_text(make_patch(tmp_path, relative, header, header.replace("line 3\n", "line 3 timeline\n")))
    patches = {
        "comment": make_patch(tmp_path, relative, header, header + "// comment only\n"),
        "conflict": make_patch(tmp_path, relative, header, header.replace("line 3\n", "line 3 candidate\n")),
        "sglang": make_patch(tmp_path, "python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh", jit, jit + "// comment only\n"),
    }
    shims = tmp_path / "bin"
    shims.mkdir()
    (shims / "unsquashfs").write_text(UNSQUASHFS)
    (shims / "unsquashfs").chmod(0o755)
    return ws, patches


def provision(tmp_path: Path, ws: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "GITHUB_WORKSPACE": str(ws), "GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "abc", "GITHUB_OUTPUT": str(tmp_path / "output"), "RUNTIME_ROOT": str(tmp_path / "runs")}
    return subprocess.run(["bash", str(ws / "scripts/provision-firmus-kernel-bench.sh")], cwd=tmp_path, env=env, capture_output=True, text=True, check=False)


def add_candidate(ws: Path, name: str, patch: str, timeline: bool | str = False, env: str | None = None) -> None:
    directory = ws / "patches/deep-gemm-bench" / name
    directory.mkdir(parents=True)
    (directory / "000.patch").write_text(patch)
    if isinstance(timeline, str):
        (directory / "timeline").write_text(timeline)
    elif timeline:
        (directory / "timeline").touch()
    if env is not None:
        (directory / "env").write_text(env)


@pytest.mark.skipif(shutil.which("jq") is None, reason="provisioning uses jq like the Firmus login node")
def test_provision_stages_each_candidate_and_records_timeline_failures(tmp_path: Path) -> None:
    ws, patches = workspace(tmp_path)
    add_candidate(ws, "a-comment", patches["comment"], timeline=True)
    add_candidate(ws, "b-conflict", patches["conflict"], timeline=True)
    (ws / "patches/deep-gemm-bench/timeline").touch()
    (ws / "patches/sglang-bench").mkdir(parents=True)
    (ws / "patches/sglang-bench/000.patch").write_text(patches["sglang"])

    result = provision(tmp_path, ws)

    assert result.returncode == 0, result.stderr
    runtime = Path((tmp_path / "output").read_text().strip().split("=", 1)[1])
    variants = json.loads((runtime / "results/variants.json").read_text())
    assert [(v["name"], v["timeline"]) for v in variants["deep_gemm"]] == [("image", "ready"), ("a-comment", "ready"), ("b-conflict", "apply-failed")]
    assert variants["deep_gemm"][1]["patches"][0]["path"] == "patches/deep-gemm-bench/a-comment/000.patch"
    assert variants["sglang"]["bench_script"] is False
    header = "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"
    assert (runtime / "overlay/candidates/a-comment" / header).read_text().endswith("// comment only\n")
    assert "line 3 candidate" in (runtime / "overlay/candidates/b-conflict" / header).read_text()
    assert "line 3 timeline" in (runtime / "overlay/timeline/image" / header).read_text()
    assert (runtime / "overlay/timeline/a-comment" / header).read_text().endswith("// comment only\n")
    assert not (runtime / "overlay/timeline/b-conflict").exists()
    assert (runtime / "logs/timeline-apply-b-conflict.log").read_text()
    assert not (runtime / "logs/timeline-apply-a-comment.log").exists()
    assert (runtime / "overlay/sglang/python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh").read_text().endswith("// comment only\n")
    assert (runtime / "source/patches/deep-gemm-bench/a-comment/timeline").exists()
    assert json.loads((runtime / "source/provenance.json").read_text())["schema_version"] == "qwen3-megamoe-kernel-bench/v2"


@pytest.mark.skipif(shutil.which("jq") is None, reason="provisioning uses jq like the Firmus login node")
def test_candidate_timeline_marker_with_content_is_that_candidates_timeline_patch(tmp_path: Path) -> None:
    ws, patches = workspace(tmp_path)
    relative = "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"
    header = "".join(f"line {index}\n" for index in range(1, 12))
    # The stock timeline patch edits line 3, which this candidate changed; its own timeline patch edits line 9.
    own = make_patch(tmp_path, relative, header.replace("line 3\n", "line 3 candidate\n"), header.replace("line 3\n", "line 3 candidate\n").replace("line 9\n", "line 9 own timeline\n"))
    add_candidate(ws, "own", patches["conflict"], timeline=own)

    result = provision(tmp_path, ws)

    assert result.returncode == 0, result.stderr
    runtime = Path((tmp_path / "output").read_text().strip().split("=", 1)[1])
    variants = json.loads((runtime / "results/variants.json").read_text())
    assert [(v["name"], v["timeline"]) for v in variants["deep_gemm"]] == [("image", "not-requested"), ("own", "ready")]
    assert "line 9 own timeline" in (runtime / "overlay/timeline/own" / relative).read_text()


@pytest.mark.skipif(shutil.which("jq") is None, reason="provisioning uses jq like the Firmus login node")
def test_control_and_candidate_share_ordered_serving_overlays(tmp_path: Path) -> None:
    ws, _ = workspace(tmp_path)
    relative = "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"
    original = "".join(f"line {index}\n" for index in range(1, 12))
    first = original.replace("line 7\n", "serving stage one\n")
    second = first.replace("serving stage one\n", "serving stage two\n")
    overlay = ws / "patches/deep-gemm-overlay"
    overlay.mkdir(parents=True)
    (overlay / "first.patch").write_text(make_patch(tmp_path, relative, original, first))
    (overlay / "second.patch").write_text(make_patch(tmp_path, relative, first, second))
    config_path = ws / "configs/firmus-framework.json"
    config = json.loads(config_path.read_text())
    config["deep_gemm_overlay_patches"] = ["patches/deep-gemm-overlay/first.patch", "patches/deep-gemm-overlay/second.patch"]
    config_path.write_text(json.dumps(config))
    add_candidate(ws, "change", make_patch(tmp_path, relative, second, second.replace("serving stage two\n", "candidate after serving\n")))

    result = provision(tmp_path, ws)

    assert result.returncode == 0, result.stderr
    runtime = Path((tmp_path / "output").read_text().strip().split("=", 1)[1])
    assert (runtime / "image" / HEADER).read_text() == second
    assert "candidate after serving" in (runtime / "overlay/candidates/change" / relative).read_text()
    variants = json.loads((runtime / "results/variants.json").read_text())
    assert [p["path"] for p in variants["deep_gemm"][0]["patches"]] == config["deep_gemm_overlay_patches"]
    for path in config["deep_gemm_overlay_patches"]:
        assert (runtime / "source" / path).read_bytes() == (ws / path).read_bytes()


@pytest.mark.skipif(shutil.which("jq") is None, reason="provisioning uses jq like the Firmus login node")
def test_candidate_env_reaches_only_that_candidates_steps(tmp_path: Path) -> None:
    ws, patches = workspace(tmp_path)
    add_candidate(ws, "a-host", patches["comment"], timeline=True, env="DG_SM90_PREFILL_2WG=1\nDG_EXTRA=a.b/c,d:e-f\n")
    add_candidate(ws, "b-plain", patches["comment"])
    (ws / "patches/deep-gemm-bench/timeline").touch()

    result = provision(tmp_path, ws)

    assert result.returncode == 0, result.stderr
    runtime = Path((tmp_path / "output").read_text().strip().split("=", 1)[1])
    variants = json.loads((runtime / "results/variants.json").read_text())
    assert [(v["name"], v["env"]) for v in variants["deep_gemm"]] == [("image", []), ("a-host", ["DG_SM90_PREFILL_2WG=1", "DG_EXTRA=a.b/c,d:e-f"]), ("b-plain", [])]
    assert (runtime / "overlay/env/a-host").read_text() == "DG_SM90_PREFILL_2WG=1\nDG_EXTRA=a.b/c,d:e-f\n"
    assert not (runtime / "overlay/env/b-plain").exists()
    assert (runtime / "source/patches/deep-gemm-bench/a-host/env").exists()

    # run.sbatch with an srun stub that records each step's command line.
    shims = tmp_path / "bin"
    (shims / "srun").write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >>"${SRUN_LOG}"\n')
    (shims / "srun").chmod(0o755)
    srun_log = tmp_path / "srun.log"
    env = {**os.environ, "PATH": f"{shims}:{os.environ['PATH']}", "SLURM_SUBMIT_DIR": str(runtime), "SLURM_JOB_ID": "1", "SRUN_LOG": str(srun_log)}
    run = subprocess.run(["bash", str(runtime / "scripts/run.sbatch")], cwd=runtime, env=env, capture_output=True, text=True, check=False)

    assert run.returncode == 0, run.stdout + run.stderr
    lines = srun_log.read_text().splitlines()
    steps = [line for line in lines if "--variant" in line or "check-firmus-megamoe.py" in line]
    assert len(steps) == 7  # image bench + timeline, a-host gate + bench + timeline, b-plain gate + bench
    assert f"PYTHONPATH={runtime}/image/opt/sglang/venv/lib/python3.12/site-packages" in steps[0]
    for line in steps:
        assert "env DG_SM90_PREFILL_2WG=1 SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY=1 DG_JIT_CACHE_DIR=" in line
        candidate = f"{runtime}/overlay/candidates/a-host" in line or f"{runtime}/overlay/timeline/a-host" in line
        assert ("DG_SM90_PREFILL_2WG=1 DG_EXTRA=a.b/c,d:e-f " in line) is candidate, line
    assert (runtime / "run-success").exists()


@pytest.mark.skipif(shutil.which("jq") is None, reason="provisioning uses jq like the Firmus login node")
@pytest.mark.parametrize("env", ["", "dg_lower=1\n", "DG_EMPTY=\n", "DG_SPACE=1 2\n", "DG_QUOTE='1'\n", "# comment\nDG_A=1\n", "DG_A=1\n\n", "DG_A=1\nDG_A=2\n", "PYTHONPATH=/x\n", "DG_JIT_CACHE_DIR=/x\n"])
def test_provision_rejects_malformed_candidate_env_before_staging(tmp_path: Path, env: str) -> None:
    ws, patches = workspace(tmp_path)
    add_candidate(ws, "c0", patches["comment"], env=env)

    result = provision(tmp_path, ws)

    assert result.returncode != 0
    assert "env" in result.stderr
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("layout", ["flat", "too-many", "too-many-timelines", "bad-name"])
def test_provision_rejects_layouts_before_staging(tmp_path: Path, layout: str) -> None:
    ws, patches = workspace(tmp_path)
    if layout == "flat":
        (ws / "patches/deep-gemm-bench").mkdir(parents=True)
        (ws / "patches/deep-gemm-bench/000.patch").write_text(patches["comment"])
    elif layout == "too-many":
        # 1 image bench + 6 x (gate + bench) = 13 steps > 12.
        for index in range(6):
            add_candidate(ws, f"c{index}", patches["comment"])
    elif layout == "too-many-timelines":
        # 1 + 4 x (gate + bench + timeline) = 13 steps > 12.
        for index in range(4):
            add_candidate(ws, f"c{index}", patches["comment"], timeline=True)
    else:
        add_candidate(ws, "image", patches["comment"])

    result = provision(tmp_path, ws)

    assert result.returncode != 0
    assert not (tmp_path / "runs").exists()


def bench_row(case: str, median: float, digest: str, clock: float = 1980) -> dict:
    return {"case": case, "variant": "x", "median_ms": median, "p10_ms": median * 0.99, "p90_ms": median * 1.01, "rank_load_max_over_mean": 1.0, "deterministic": True, "output_sha256": digest, "sm_clock_mhz_before": [clock] * 4, "sm_clock_mhz_after": [clock] * 4, "power_w_before": [500] * 4, "power_w_after": [600] * 4}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def load_summary(monkeypatch: pytest.MonkeyPatch):
    # Timeline .npy reports need numpy; these fixtures carry no dumps.
    if importlib.util.find_spec("numpy") is None:
        monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    return runpy.run_path(str(ROOT / "scripts/summarize-megamoe-kernel-bench.py"))["summarize"]


def test_summary_compares_candidates_and_flags_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summarize = load_summary(monkeypatch)
    patch = [{"path": "patches/deep-gemm-bench/x/000.patch", "sha256": "0"}]
    (tmp_path / "variants.json").write_text(json.dumps({"deep_gemm": [{"name": "image", "patches": [], "env": [], "timeline": "ready"}, {"name": "fast", "patches": patch, "env": ["DG_SM90_PREFILL_2WG=1"], "timeline": "not-requested"}, {"name": "wrong", "patches": patch, "timeline": "apply-failed"}, {"name": "broken", "patches": patch, "timeline": "ready"}, {"name": "reordered", "patches": patch, "timeline": "not-requested"}], "sglang": {"patches": [{"path": "patches/sglang-bench/000.patch", "sha256": "1"}], "bench_script": True}}))
    (tmp_path / "steps.tsv").write_text("image\tbench\t0\t70\nimage\ttimeline\t0\t35\nsglang-image\tcheck\t0\t30\nsglang-image\tbench\t0\t20\nsglang-candidate\tcheck\t0\t60\nsglang-candidate\tbench\t0\t20\nfast\tgate\t0\t70\nfast\tbench\t0\t40\nwrong\tgate\t0\t70\nwrong\tbench\t0\t40\nbroken\tgate\t1\t50\nreordered\tgate\t0\t70\nreordered\tbench\t0\t40\n")
    write_rows(tmp_path / "image/bench.jsonl", [bench_row("decode-balanced-128", 0.2, "d"), bench_row("prefill-balanced-4096", 1.7, "p", clock=1500)])
    write_rows(tmp_path / "image/timeline.jsonl", [{**bench_row("decode-balanced-128", 0.23, "d"), "delta_median_vs_reference": 0.15, "bitwise_equal_to_reference": True}])
    write_rows(tmp_path / "fast/bench.jsonl", [bench_row("decode-balanced-128", 0.18, "d"), bench_row("prefill-balanced-4096", 1.6, "p", clock=1600)])
    write_rows(tmp_path / "wrong/bench.jsonl", [bench_row("decode-balanced-128", 0.19, "other"), bench_row("prefill-balanced-4096", 1.7, "p")])
    # A different accumulation order changes the bytes; the passing gate decides.
    write_rows(tmp_path / "reordered/bench.jsonl", [bench_row("decode-balanced-128", 0.19, "d"), bench_row("prefill-balanced-4096", 1.5, "reordered")])
    write_rows(tmp_path / "sglang/image.jsonl", [{"case": "silu-quant-128", "median_us": 10.0, "p10_us": 9.9, "p90_us": 10.1, "output_sha256": "s"}])
    write_rows(tmp_path / "sglang/candidate.jsonl", [{"case": "silu-quant-128", "median_us": 8.0, "p10_us": 7.9, "p90_us": 8.1, "output_sha256": "s"}])

    text, failed = summarize(tmp_path)

    assert failed == ["wrong", "broken"]
    assert "| fast | 000.patch; env DG_SM90_PREFILL_2WG=1 | pass | pass | 2/2 | not-requested | OK |" in text
    assert "| reordered | 000.patch | pass | pass | 1/2 (rest differ) | not-requested | OK |" in text
    assert "| wrong | 000.patch | pass | pass | 1/2 (rest differ) | timeline patch did not apply | FAILED |" in text
    assert "| broken | 000.patch | FAIL (exit 1) | not run | 0/2 (rest differ) | not run | FAILED |" in text
    assert "| decode-balanced-128 | 0.2000 | 0.1800 (-10.0%) | 0.1900 (-5.0%)* | - |" in text
    assert "| prefill-balanced-4096 | fast | 1.6000 | 1.5840 | 1.6160 | -5.9% | 1600 / 1600 | 600 |" in text
    assert "| decode-balanced-128 | image | 0.2300 | +15.0% | True | 1980 / 1980 |" in text
    assert "| silu-quant-128 | 10.00 | 8.00 | -20.0% | 7.90 / 8.10 | True |" in text


def test_summary_passes_clean_run_and_reports_differing_hashes_without_failing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summarize = load_summary(monkeypatch)
    (tmp_path / "variants.json").write_text(json.dumps({"deep_gemm": [{"name": "image", "patches": [], "timeline": "not-requested"}], "sglang": None}))
    (tmp_path / "steps.tsv").write_text("image\tbench\t0\t70\n")
    write_rows(tmp_path / "image/bench.jsonl", [bench_row("decode-balanced-128", 0.2, "d")])
    assert summarize(tmp_path)[1] == []

    (tmp_path / "variants.json").write_text(json.dumps({"deep_gemm": [{"name": "image", "patches": [], "timeline": "not-requested"}], "sglang": {"patches": [], "bench_script": True}}))
    (tmp_path / "steps.tsv").write_text("image\tbench\t0\t70\nsglang-image\tcheck\t0\t1\nsglang-image\tbench\t0\t1\nsglang-candidate\tcheck\t0\t1\nsglang-candidate\tbench\t0\t1\n")
    write_rows(tmp_path / "sglang/image.jsonl", [{"case": "rope", "median_us": 5.0, "p10_us": 5.0, "p90_us": 5.0, "output_sha256": "a"}])
    write_rows(tmp_path / "sglang/candidate.jsonl", [{"case": "rope", "median_us": 5.0, "p10_us": 5.0, "p90_us": 5.0, "output_sha256": "b"}])
    text, failed = summarize(tmp_path)
    assert failed == []
    assert "| rope | 5.00 | 5.00 | +0.0% | 5.00 / 5.00 | False |" in text


@pytest.mark.parametrize("patch_name", ["sm90-decode-sparse-dispatch.patch", "sm90-decode-fc1-register.patch", "sm90-decode-independent-wg.patch", "stream-combine.patch"])
@pytest.mark.parametrize("same", [False, True])
def test_arithmetic_preserving_candidates_require_equal_outputs(tmp_path, monkeypatch, patch_name, same):
    summarize = load_summary(monkeypatch)
    variants = [{"name": "image", "patches": [], "timeline": "not-requested"}, {"name": "renamed", "patches": [{"path": f"patches/deep-gemm-bench/renamed/{patch_name}"}], "timeline": "not-requested"}]
    (tmp_path / "variants.json").write_text(json.dumps({"deep_gemm": variants, "sglang": None}))
    (tmp_path / "steps.tsv").write_text("image\tbench\t0\t1\nrenamed\tgate\t0\t1\nrenamed\tbench\t0\t1\n")
    write_rows(tmp_path / "image/bench.jsonl", [bench_row("decode", 0.2, "control")])
    write_rows(tmp_path / "renamed/bench.jsonl", [bench_row("decode", 0.1, "control" if same else "changed")])
    assert summarize(tmp_path)[1] == ([] if same else ["renamed"])


@pytest.mark.parametrize("failure", ["image-replay", "image-missing-replay", "image-bench", "image-missing-bench", "candidate-replay", "missing-replay", "wrong-case", "empty"])
def test_summary_rejects_unverified_replay_and_case_coverage(tmp_path, monkeypatch, failure):
    summarize = load_summary(monkeypatch)
    variants = [{"name": name, "patches": [], "timeline": "not-requested"} for name in ("image", "candidate")]
    (tmp_path / "variants.json").write_text(json.dumps({"deep_gemm": variants, "sglang": None}))
    (tmp_path / "steps.tsv").write_text("image\tbench\t0\t1\ncandidate\tgate\t0\t1\ncandidate\tbench\t0\t1\n")
    image_row, candidate_row = bench_row("decode", 0.2, "same"), bench_row("decode", 0.1, "same")
    if failure == "image-replay":
        image_row["deterministic"] = False
    elif failure == "image-missing-replay":
        del image_row["deterministic"]
    elif failure in ("image-bench", "image-missing-bench"):
        (tmp_path / "steps.tsv").write_text(("image\tbench\t1\t1\n" if failure == "image-bench" else "") + "candidate\tgate\t0\t1\ncandidate\tbench\t0\t1\n")
    elif failure == "candidate-replay":
        candidate_row["deterministic"] = False
    elif failure == "missing-replay":
        del candidate_row["deterministic"]
    elif failure == "wrong-case":
        candidate_row["case"] = "different"
    write_rows(tmp_path / "image/bench.jsonl", [] if failure == "empty" else [image_row])
    write_rows(tmp_path / "candidate/bench.jsonl", [] if failure == "empty" else [candidate_row])
    failed = summarize(tmp_path)[1]
    assert failed == (["image", "candidate"] if failure.startswith("image-") or failure == "empty" else ["candidate"])


@pytest.mark.parametrize("kind,percent", [("mixed50", 50), ("mixed64", 64)])
def test_mixed_routing_is_local_unique_and_rank_balanced(kind: str, percent: int) -> None:
    torch = pytest.importorskip("torch")
    # Load only routing: the benchmark's module initialization requires CUDA,
    # a distributed process group and DeepGEMM, none needed for this CPU check.
    source = ast.parse((ROOT / "scripts/bench-firmus-megamoe.py").read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "routing")
    context = {"torch": torch, "NUM_EXPERTS": 128, "NUM_RANKS": 4, "TOPK": 8}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "routing", "exec"), context)  # noqa: S102 - trusted repository function, without CUDA initialization
    loads = torch.zeros(4, dtype=torch.int64)
    for rank in range(4):
        context["dist"] = types.SimpleNamespace(get_rank=lambda rank=rank: rank)
        generator = torch.Generator().manual_seed(104 + rank)
        ids, weights = context["routing"](kind, 4096, generator, "cpu")
        assert ids.shape == weights.shape == (4096, 8)
        assert ids.dtype == torch.int32 and ids.is_contiguous() and weights.is_contiguous()
        assert ids.min() >= 0 and ids.max() < 128
        assert (ids.sort(dim=1).values.diff(dim=1) > 0).all()
        local = ((ids // 32) == rank).sum().item()
        assert local == (4096 * 8 * percent) // 100
        assert torch.allclose(weights.sum(dim=1), torch.ones(4096))
        loads += torch.bincount((ids // 32).flatten().long(), minlength=4)
    assert loads.tolist() == [32768] * 4


@pytest.mark.parametrize("case,counts,expected_rows,expected_groups", [
    ("mixed-geometry-graph", [7, 0, 145, 256], [7, 1, 145, 256], [64, 64, 64, 128]),
    ("boundary", [255, 256, 257, 0], [255, 256, 257, 1], [64, 128, 128, 64]),
    ("idle-prefill-rows", [4096, 4096, 4096, 0], [4096, 4096, 4096, 256], [128] * 4),
    ("decode-all-idle", [0] * 4, [1] * 4, [64] * 4),
    ("decode-graph-swapab-144", [144, 140, 136, 132], [144, 140, 136, 132], [64] * 4),
])
def test_numeric_gate_reference_matches_each_expert_owner_geometry(case, counts, expected_rows, expected_groups):
    # Execute selection, the real empty-expert reference loop, and allocation.
    # The loop reassigns `rows` to index tensors; allocation must keep its size.
    # A global max(counts) incorrectly makes the first three mixed ranks SF128.
    source = ast.parse((ROOT / "scripts/check-firmus-megamoe.py").read_text())
    main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    loop = next(node for node in main.body if isinstance(node, ast.For) and ast.unparse(node.target) == "(case, counts)")
    names = {"count", "rows", "output_rows", "prefill_geometry", "l2_group", "actual"}
    statements = [node for node in loop.body if (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)) or (isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "local_expert")]
    selection = compile(ast.Module(body=statements, type_ignores=[]), "gate_geometry", "exec")
    def empty(row_count, columns, **kwargs):
        assert isinstance(row_count, int), "Output row count was overwritten by reference indices"
        return row_count, columns
    torch = types.SimpleNamespace(empty=empty, bfloat16="bf16", where=lambda _: (types.SimpleNamespace(numel=lambda: 0), None))
    for rank in range(4):
        context = {"case": case, "counts": counts, "rank": rank, "torch": torch, "device": "cpu", "all_ids": None}
        exec(selection, context)  # noqa: S102 - trusted repository selection, without CUDA initialization
        assert context["actual"] == (expected_rows[rank], 4096)
        assert context["l2_group"] == expected_groups[rank]


def test_summary_displays_mixed_locality_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summarize = load_summary(monkeypatch)
    (tmp_path / "variants.json").write_text(json.dumps({"deep_gemm": [{"name": "image", "patches": [], "timeline": "not-requested"}, {"name": "control-sf128-swizzle", "patches": [], "timeline": "not-requested"}], "sglang": None}))
    (tmp_path / "steps.tsv").write_text("image\tbench\t0\t65\ncontrol-sf128-swizzle\tgate\t0\t80\ncontrol-sf128-swizzle\tbench\t0\t65\n")
    rows = [bench_row(f"prefill-{kind}-4096", 1.5, kind) for kind in ("mixed50", "mixed64", "replica8")]
    write_rows(tmp_path / "image/bench.jsonl", rows)
    write_rows(tmp_path / "control-sf128-swizzle/bench.jsonl", rows)
    text, failed = summarize(tmp_path)
    assert failed == []
    for row in rows:
        assert f"| {row['case']} | 1.5000 | 1.5000 (+0.0%) |" in text


def test_replica_mapping_preserves_logical_experts_and_masks() -> None:
    torch = pytest.importorskip("torch")
    replicas = runpy.run_path(str(ROOT / "scripts/firmus_megamoe_replicas.py"))
    mapping = replicas["physical_to_logical"]()
    assert len(mapping) == 140
    assert [mapping.count(e) for e in range(128)] == [4] * 4 + [1] * 124
    loads = torch.zeros(4, dtype=torch.int64)
    for rank in range(4):
        ids = torch.arange(-1, 128, dtype=torch.int32)
        physical = replicas["remap"](ids, rank)
        assert physical[0] == -1
        assert torch.tensor(mapping)[physical[1:].long()].tolist() == list(range(128))
        assert (physical[1:5] // 35 == rank).all()
        assert (physical[1:] // 35 == rank).sum().item() == 35
        loads += torch.bincount((physical[1:] // 35).long(), minlength=4)
    assert loads.tolist() == [128] * 4
    source = ast.parse((ROOT / "scripts/bench-firmus-megamoe.py").read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "routing")
    context = {"torch": torch, "NUM_EXPERTS": 128, "NUM_RANKS": 4, "TOPK": 8, "dist": types.SimpleNamespace(get_rank=lambda: 3)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "routing", "exec"), context)  # noqa: S102 - trusted repository function, without CUDA initialization
    logical, _ = context["routing"]("balanced", 4096, torch.Generator().manual_seed(9), "cpu")
    physical = replicas["remap"](logical, 3)
    assert torch.equal(torch.tensor(mapping)[physical.long()], logical.long())


@pytest.mark.skipif(shutil.which("jq") is None, reason="provisioning uses jq like the Firmus login node")
def test_provision_accepts_five_plain_candidates(tmp_path: Path) -> None:
    ws, patches = workspace(tmp_path)
    for index in range(5):
        add_candidate(ws, f"c{index}", patches["comment"])
    result = provision(tmp_path, ws)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("bench_script,expected_steps", [(False, 13), (True, 15)])
def test_provision_counts_sglang_and_image_timeline_before_staging(tmp_path: Path, bench_script: bool, expected_steps: int) -> None:
    ws, patches = workspace(tmp_path)
    for index in range(3):
        add_candidate(ws, f"c{index}", patches["comment"], timeline=True)
    (ws / "patches/deep-gemm-bench/timeline").touch()
    (ws / "patches/sglang-bench").mkdir(parents=True)
    (ws / "patches/sglang-bench/000.patch").write_text(patches["sglang"])
    if bench_script:
        (ws / "scripts/bench-firmus-sglang-kernels.py").touch()

    result = provision(tmp_path, ws)

    assert result.returncode != 0
    assert f"{expected_steps} Slurm steps" in result.stderr
    assert not (tmp_path / "runs").exists()
