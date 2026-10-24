"""WG-separated timeline decoding and control-only staging, without GPU imports."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
@pytest.fixture
def summary():
    pytest.importorskip("numpy", reason="timeline dumps and the production summary use numpy")
    spec = importlib.util.spec_from_file_location("timeline_summary", ROOT / "scripts/summarize-megamoe-kernel-bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump_v2(summary, groups=2):
    np = summary.np
    events = np.zeros((2, 46, 2), dtype=np.uint64)
    counters = np.zeros((2, 32), dtype=np.uint64)
    for cta in range(2):
        start = 1_000_000 + cta * 1000
        for name, elapsed in [("Entry", 0), ("DispExit", 100_000)]:
            events[cta, summary.V2_EVENTS.index(name)] = start + elapsed, 2 * (start + elapsed)
        for wg in range(groups):
            for name, elapsed in zip(summary.WG_EVENTS, [1000, 10_000, 30_000, 32_000, 60_000, 61_000, 70_000, 90_000], strict=True):
                events[cta, summary.V2_EVENTS.index(f"WG{wg}{name}")] = start + elapsed, 2 * (start + elapsed)
            counters[cta, wg * 8:(wg + 1) * 8] = [3, 4, 2000, 4000, 20_000 * (wg + 1), 30_000, 6000, 8000]
    header = np.array([summary.MAGIC, 2, 2, 46, 32, 4096, 0, 128, 256, groups * 128, 3, 64, 32, groups, 0, 0], dtype=np.uint64)
    return np.concatenate([header, events.ravel(), counters.ravel()])


@pytest.mark.parametrize("groups", [2, 4])
def test_timeline_v2_reports_each_wg_without_merging_counters(tmp_path, groups, summary):
    np = summary.np
    words = dump_v2(summary, groups)
    parsed = summary.parse(words)
    assert parsed["end"] == 101
    assert np.all(parsed["ghz"] == 2)
    assert parsed["counters"][0, summary.V2_COUNTERS.index("WG1FC1Loop")] == 20
    path = tmp_path / "prefill-real43-replica4-4096.npy"
    np.save(path, np.tile(words, (3, 4, 1)))
    text = summary.report(path, path.stem)
    assert "| 0 | FC1 mainloop | 10.0 / 10.0" in text
    assert "| 1 | FC1 mainloop | 20.0 / 20.0" in text
    assert "FC1 epilogue (SwiGLU + FP8 + store) | 3.0 / 3.0" in text
    assert "FC2 epilogue (BF16 + NVLink scatter) | 4.0 / 4.0" in text
    assert "pre-combine wait (grid + NVLink) | 9.0 / 9.0" in text
    assert "combine (incl. dispatch cleanup handshake) | 20.0 / 20.0" in text
    assert ("WG3Start" in text) == (groups == 4)


def test_wg_without_combine_work_has_missing_end_not_zero_duration(tmp_path, summary):
    np = summary.np
    words = dump_v2(summary)
    events = words[16:16 + 2 * 46 * 2].reshape(2, 46, 2)
    events[:, summary.V2_EVENTS.index("WG1CombineEnd")] = 0
    path = tmp_path / "decode-balanced-128.npy"
    np.save(path, np.tile(words, (3, 4, 1)))
    text = summary.report(path, path.stem)
    assert "| 1 | combine (incl. dispatch cleanup handshake) | - | - | - | - |" in text


def test_unknown_timeline_schema_is_rejected(summary):
    words = dump_v2(summary)
    words[1] = 3
    with pytest.raises(AssertionError, match="Unknown timeline version"):
        summary.parse(words)


def test_adopted_header_is_only_in_serving_overlay_and_build_context():
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    assert config["deep_gemm_overlay_patches"][-1] == "patches/deep-gemm-overlay/sm90-decode-independent-wg.patch"
    bench = ROOT / "patches/deep-gemm-bench"
    assert sorted(p.name for p in bench.iterdir()) == ["prefill-iwg-a", "prefill-iwg-b", "timeline"]
    assert (bench / "prefill-iwg-b/env").read_text() == "DG_SM90_PREFILL_INDEPENDENT_WG=b\n"
    assert (bench / "timeline").stat().st_size == 0
    for name in ("containers/Dockerfile", "scripts/build-sglang-container.sh", "containers/check-sm90-independent-wg.py", "scripts/check-sm90-independent-wg.py", "scripts/check-sm90-independent-wg-host.py"):
        assert "deep-gemm-bench" not in (ROOT / name).read_text()
    checker = (ROOT / "containers/check-sm90-independent-wg.py").read_text()
    assert "patches[:-1]" in checker
    assert 'assert patches[-1] == CANDIDATE' in checker


def test_patch_layout_matches_reader_and_only_wraps_existing_stage_waits():
    patch = (ROOT / "patches/deep-gemm-timeline/sm90-mega-moe-timeline.patch").read_text()
    removed = [line[1:] for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")]
    # Every removed operation is either counter initialization (same SMEM
    # allocation) or one of the exact existing waits now wrapped in timing.
    assert len(removed) == 12
    for line in removed:
        assert "i < kNumExperts;" in line or re.fullmatch(r"\s*full_barriers\[(stage_idx|next_stage)\]->wait\((phase|next_phase)\);", line)
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    assert "kCountersPerWG = 8" in added
    assert "kWGStart = 14" in added
    assert "kEventsPerWG = 8" in added
    assert "kNumCounters * sizeof(uint32_t) == 128" in added
    assert "ptx::sync_" not in added
    assert "comm::grid_sync" not in added
    assert "comm::nvlink_barrier" not in added
    assert added.count("tl_finish();") == 2  # inactive-M and normal exits


def test_build_checker_applies_adopted_overlay_once(tmp_path, monkeypatch):
    import difflib
    import sys

    spec = importlib.util.spec_from_file_location("build_checker", ROOT / "containers/check-sm90-independent-wg.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    package = tmp_path / "package"
    header = package / "include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"
    header.parent.mkdir(parents=True)
    header.write_text("image\n")
    patches = tmp_path / "patches"
    patches.mkdir()
    previous = "image\n"
    for name, text in [("prior.patch", "serving before adoption\n"), (checker.CANDIDATE, "serving after adoption\n")]:
        patch = "".join(difflib.unified_diff(previous.splitlines(True), text.splitlines(True), fromfile="a/" + str(checker.HEADER), tofile="b/" + str(checker.HEADER)))
        (patches / name).write_text(patch)
        previous = text
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"deep_gemm_overlay_patches": ["patches/deep-gemm-overlay/prior.patch", "patches/deep-gemm-overlay/" + checker.CANDIDATE]}))
    output = tmp_path / "result.json"
    seen = []

    def compile_fixture(nvcc, package, overlay, directory, name, geometry):
        seen.append((name, (overlay / checker.HEADER).read_text()))
        return {"spill_bytes": [["0", "0"]], "sass_sha256": "same opt-out SASS", "serialization_warning": False}

    monkeypatch.setattr(checker, "compile_one", compile_fixture)
    monkeypatch.setattr(sys, "argv", ["check", "--package", str(package), "--patch-dir", str(patches), "--config", str(config), "--output", str(output)])
    checker.main()
    assert seen == [("base-decode", "serving before adoption\n"), ("base-prefill", "serving before adoption\n"), ("control-decode", "serving after adoption\n"), ("control-prefill", "serving after adoption\n"), ("independent-decode", "serving after adoption\n")]
    assert set(json.loads(output.read_text())["patches"]) == {"prior.patch", checker.CANDIDATE}


def test_legacy_v1_timeline_is_still_readable(tmp_path, summary):
    np = summary.np
    events = np.zeros((2, 40, 2), dtype=np.uint64)
    for cta in range(2):
        for event in range(len(summary.EVENTS)):
            time = 1_000_000 + cta * 1000 + event * 1000
            events[cta, event] = time, time * 2
    header = np.array([summary.MAGIC, 1, 2, 40, 24, 4096, 0, 128, 256, 512, 3, 64, 32, 0, 0, 0], dtype=np.uint64)
    words = np.concatenate([header, events.ravel(), np.zeros(2 * 24, dtype=np.uint64)])
    assert summary.parse(words)["header"]["version"] == 1
    path = tmp_path / "legacy.npy"
    np.save(path, np.tile(words, (3, 4, 1)))
    assert "math: FC1 span (first ready -> last end)" in summary.report(path, "legacy")


def test_timeline_switches_tail_when_replica_routing_switches_buffer(summary):
    import ast
    import types

    # Execute the real capture prologue with two differently laid-out buffers;
    # importing the benchmark itself would import CUDA dependencies.
    tree = ast.parse((ROOT / "scripts/bench-firmus-megamoe.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    cases = next(node for node in main.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "case")
    capture = next(node for node in cases.body if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "timeline")
    prologue = compile(ast.Module(body=capture.body[:3], type_ignores=[]), "capture-prologue", "exec")
    namespace = {"torch": types.SimpleNamespace(int64=summary.np.int64), "locate_timeline": lambda buf, size: (size, 2), "row": {}}
    arrays = [summary.np.full(64, value, dtype=summary.np.int8) for value in (1, 2)]
    for array, size in zip(arrays, (8, 16), strict=True):
        namespace.update(buf=types.SimpleNamespace(buffer=array), num_bytes=size)
        exec(prologue, namespace)  # noqa: S102 - fixed repository AST with CPU fixtures
        region = namespace["region"]
        assert summary.np.shares_memory(region, array)
        assert region.tobytes() == bytes([int(array[0])]) * 16
        assert namespace["row"]["timeline_offset_minus_layout_bytes"] == 0
