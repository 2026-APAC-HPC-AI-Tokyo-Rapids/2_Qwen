"""CPU regression gates for Prefill queue selection, staging and image guard."""
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(path):
    spec = importlib.util.spec_from_file_location("candidate_module", ROOT / path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.mark.parametrize("mode", ["a", "b"])
@pytest.mark.parametrize("corruption", [None, "host", "extension", "abi"])
def test_prefill_provision_guard_checks_actual_requested_host(tmp_path, mode, corruption):
    source = (ROOT / "scripts/provision-firmus-kernel-bench.sh").read_text()
    start = source.index("check_candidate_host() {\n")
    function = source[start:source.index("\n}\n", start) + 3]
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "env").write_text(f"DG_SM90_PREFILL_INDEPENDENT_WG={mode}\n")
    extension = tmp_path / "_C.so"
    extension.write_bytes(b"test extension, not loaded")
    patch = ROOT / "patches/deep-gemm-host/sm90-prefill-independent-wg.patch"
    record = tmp_path / "host.json"
    record.write_text(json.dumps({"abi": "pybind" if corruption == "abi" else "tvm-ffi", "patches": {patch.name: "0" * 64 if corruption == "host" else hashlib.sha256(patch.read_bytes()).hexdigest()}, "extension_sha256": "0" * 64 if corruption == "extension" else hashlib.sha256(extension.read_bytes()).hexdigest()}))
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", function + '\ncheck_candidate_host "$1" "$2" "$3"\n', "guard", str(candidate) + "/", str(record), str(extension)], env={**os.environ, "GITHUB_WORKSPACE": str(ROOT)}, capture_output=True, text=True, check=False)
    assert (result.returncode == 0) == (corruption is None), result.stderr


def test_bench_candidates_and_image_compile_use_same_patch():
    patch = ROOT / "patches/deep-gemm-overlay/sm90-prefill-independent-wg.patch"
    candidate = ROOT / "patches/deep-gemm-bench/prefill-iwg-b"
    assert (candidate / "sm90-prefill-iwg-b.patch").read_bytes() == patch.read_bytes()
    assert not (candidate / "timeline").exists()
    assert not (ROOT / "patches/deep-gemm-bench/prefill-iwg-a/timeline").exists()
    assert (ROOT / "patches/deep-gemm-bench/timeline").is_file()
    assert "check-sm90-prefill-independent-wg.py" in (ROOT / "containers/Dockerfile").read_text()
    assert "check-sm90-prefill-independent-wg.py" in (ROOT / "scripts/build-sglang-container.sh").read_text()
    assert (ROOT / "patches/deep-gemm-host/series").read_text().split()[-1] == patch.name


def test_summary_enforces_both_prefill_hash_identities():
    pytest.importorskip("numpy")
    summary = module("scripts/summarize-megamoe-kernel-bench.py")
    assert {"sm90-prefill-iwg-a.patch", "sm90-prefill-iwg-b.patch"} <= summary.EXACT_OUTPUT_PATCHES


def test_gate_modes_do_not_filter_existing_case_or_replay_paths():
    source = (ROOT / "scripts/check-firmus-megamoe.py").read_text()
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assignments = [n for n in function.body if isinstance(n, ast.Assign)]
    cases = next(n.value for n in assignments if any(isinstance(t, ast.Name) and t.id == "cases" for t in n.targets))
    assert {ast.literal_eval(case.elts[0]) for case in cases.elts} >= {"capacity-hot-experts", "mixed-geometry-graph", "decode-graph", "cross-rank-empty-tail"}
    assert 'cases.append(("idle-prefill-rows"' in source
    assert 'check_replicas(quantized, dequantized)' in source
    assert "prefill_iwg" not in source[source.index("    cases ="):]


def test_static_virtual_fc1_tasks_cover_expert_tiles_and_halves_once():
    counts = (0, 1, 64, 65, 128, 129, 1025)
    expected = [(e, m, n, half) for e, count in enumerate(counts) for m in range((count + 127) // 128) for half in range(2) for n in range(12)]
    pool_experts = [(e, m) for e, count in enumerate(counts) for m in range((count + 127) // 128)]
    seen = []
    for worker in range(264):
        previous = -1
        for task in range(worker, len(expected), 264):
            parent = task // 24 * 12 + task % 12
            assert parent > previous
            previous = parent
            expert, m = pool_experts[parent // 12]
            seen.append((expert, m, parent % 12, task // 12 % 2))
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen))


def test_ring_and_compact_cd_do_not_overlap_live_regions():
    assert 140 * 4 + 128 <= 768
    ring_end = 768 + 4 * 8 + 4 * 2 * 8
    assert ring_end == 864 and ring_end <= 1024
    assert 4 * 16 * 144 <= 16384
    for valid in range(1, 129):
        rows = []
        for half in range(2):
            for pass_index in range(2):
                cells = set()
                for warp in range(4):
                    if warp // 2 != pass_index:
                        continue
                    for row in range(warp * 16, warp * 16 + 16):
                        if half * 64 + row < valid:
                            assert row % 32 not in cells
                            cells.add(row % 32)
                            rows.append(half * 64 + row)
        assert sorted(rows) == list(range(valid))


@pytest.mark.parametrize("mode,spill,expected", [(1, [32, 52], True), (1, [128, 128], True), (1, [129, 0], False), (1, [0, 129], False), (2, [0, 0], True), (2, [1, 0], False)])
def test_compile_resource_policy(mode, spill, expected):
    checker = module("containers/check-sm90-prefill-independent-wg.py")
    row = {"mode": mode, "compile_exit": 0, "spill_bytes": [spill], "registers": [168], "serialization_warning": False, "smem_bytes": 208080 if mode == 1 else 199360}
    assert checker.resource_ok(row) == expected
