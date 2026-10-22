"""CPU checks of paired dense bench inputs, capture switches and numeric gates."""

import ast
import json
import os
import sys
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_dense():
    source = ast.parse("\n".join((ROOT / "scripts" / name).read_text() for name in ("bench-firmus-sglang-kernels.py", "check-firmus-decode-kernels.py")))
    names = {"dense_gemm_mode", "require_dense_host", "check_dense_errors", "bench_dense_gemm"}
    selected = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    selected += [node for node in source.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"DENSE_ROWS", "DENSE_PROJECTIONS"} for t in node.targets)]
    namespace = {"os": os, "json": json, "Path": Path, "contextmanager": contextmanager, "__file__": str(ROOT / "scripts/bench-firmus-sglang-kernels.py")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "dense-bench", "exec"), namespace)  # noqa: S102 - only selected definitions from our tracked bench
    return namespace


@pytest.mark.parametrize("previous", [None, "0", "1"])
def test_capture_switch_restored_after_nested_modes_and_failure(monkeypatch, previous):
    if previous is None:
        monkeypatch.delenv("DG_SM90_1D2D_SMALL_M", raising=False)
    else:
        monkeypatch.setenv("DG_SM90_1D2D_SMALL_M", previous)
    mode = load_dense()["dense_gemm_mode"]
    with pytest.raises(RuntimeError), mode(True):
        assert os.environ["DG_SM90_1D2D_SMALL_M"] == "1"
        with mode(False):
            assert os.environ["DG_SM90_1D2D_SMALL_M"] == "0"
        assert os.environ["DG_SM90_1D2D_SMALL_M"] == "1"
        raise RuntimeError("capture failed")
    assert os.environ.get("DG_SM90_1D2D_SMALL_M") == previous


def test_old_image_cannot_silently_benchmark_same_geometry_twice(tmp_path):
    record = {"source_commit": "1c8138a59b3df5dac50b76b730834ed57982fdfb", "patches": {"sm90-swap-ab-144.patch": "a" * 64}}
    path = tmp_path / "host.json"
    path.write_text(json.dumps(record))
    require = load_dense()["require_dense_host"]
    with pytest.raises(AssertionError, match="rebuilt small-M"):
        require(path)
    record["patches"]["sm90-1d2d-small-m.patch"] = "b" * 64
    path.write_text(json.dumps(record))
    assert require(path) == "b" * 64


@pytest.mark.parametrize("bad", [{"finite": False}, {"relative_l2": 0.006}, {"max_scaled_error": 0.02}, {"relative_l2": float("nan")}, {"max_scaled_error": float("inf")}])
def test_numeric_gate_rejects_missing_work_scale_errors_and_nonfinite(bad):
    metrics = {"finite": True, "relative_l2": 0.002, "max_scaled_error": 0.004}
    check = load_dense()["check_dense_errors"]
    check(metrics)
    with pytest.raises(AssertionError):
        check({**metrics, **bad})


def test_bench_pairs_same_four_weight_sets_with_switch_live_during_capture(monkeypatch):
    ns = load_dense()
    monkeypatch.delenv("DG_SM90_1D2D_SMALL_M", raising=False)
    made, calls, rows = [], [], []

    def inputs(m, n, k, seed):
        result = (object(), object(), object())
        made.append((m, n, k, seed, result))
        return result

    def gemm(*args):
        calls.append((os.environ["DG_SM90_1D2D_SMALL_M"], args))

    def timing(functions):
        # The real timing helper executes these while warming up and capturing.
        assert len(functions) == 4
        for call in functions:
            call()
        assert len({id(args[0]) for _, args in calls[-4:]}) == 4
        return {"median_us": 16.0 if os.environ["DG_SM90_1D2D_SMALL_M"] == "0" else 12.0, "p10_us": 11.0, "p90_us": 17.0}

    monkeypatch.setitem(sys.modules, "deep_gemm", types.SimpleNamespace(fp8_gemm_nt=gemm))
    cuda = types.SimpleNamespace(current_device=lambda: 0, get_device_properties=lambda _: types.SimpleNamespace(multi_processor_count=132))
    ns.update(torch=types.SimpleNamespace(cuda=cuda), require_dense_host=lambda: "a" * 64, dense_fp32_reference=nullcontext, dense_inputs=inputs, dense_reference=lambda *_: object(), dense_errors=lambda *_: {}, sha256=lambda _: "hash", time_graph=timing, run_path=lambda _: ns)
    ns["bench_dense_gemm"](lambda case, digest, timing, **extra: rows.append((case, extra)))
    assert len(made) == 32 and len(rows) == 16
    for index, ((op, n, k), m) in enumerate((projection, m) for projection in ns["DENSE_PROJECTIONS"] for m in ns["DENSE_ROWS"]):
        stock, candidate = calls[index * 10:index * 10 + 5], calls[index * 10 + 5:index * 10 + 10]
        assert [mode for mode, _ in stock] == ["0"] * 5
        assert [mode for mode, _ in candidate] == ["1"] * 5
        assert [args for _, args in stock] == [args for _, args in candidate]
        assert rows[index * 2][0] == f"dense-fp8-{op}-m{m}-stock"
        assert rows[index * 2 + 1][0] == f"dense-fp8-{op}-m{m}-small-m"
        assert rows[index * 2 + 1][1]["speedup_vs_stock"] == 16 / 12
        assert rows[index * 2][1]["weight_bytes"] == 4 * n * k > 50 * 1024**2
    assert "DG_SM90_1D2D_SMALL_M" not in os.environ
