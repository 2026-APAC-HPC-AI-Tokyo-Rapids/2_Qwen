"""CPU numerical and layout contracts for the isolated round-11 overlay."""

import ast
import json
import runpy
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
REF = runpy.run_path(str(ROOT / "kernels/probes/qkv-fused-reference.py"))
PATCH = ROOT / "patches/sglang/prefill-qkv-fused.patch"


@pytest.fixture
def added(tmp_path):
    subprocess.run(["git", "apply", "--include=python/sglang/kernels/*", "--include=python/sglang/srt/layers/attention/prefill_qkv_fused.py", str(PATCH)], cwd=tmp_path, check=True)
    return tmp_path / "python/sglang"


def test_outputs_and_no_new_quantization(added):
    code = (added / "kernels/jit/csrc/gemm/qkv_fused_prefill.cuh").read_text()
    assert REF["OUTPUTS"] == ("q", "k", "v", "k_cache", "v_cache")
    for shape in ("{M,64,128}", "{M,4,128}", "{S,4,128}"):
        assert "TensorMatcher(" + shape + ").with_dtype<fp8_e4m3_t>()" in code
    assert "amax" not in code
    assert "__NV_SATFINITE" in code and "0x43e80000" in code
    assert "slot!=0" in code and "const bool valid=row<rows" in code
    assert "cudaGetDriverEntryPointByVersion" in code
    assert "tensor_map_d" not in code


def test_overlay_order_and_default_pin():
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    assert config["sglang_overlay_patches"][-2:] == ["patches/sglang/decode-moe-input.patch", "patches/sglang/prefill-qkv-fused.patch"]
    patch = PATCH.read_text()
    assert '+        if _PREFILL_QKNORM == "fused":' in patch
    assert '+                if _PREFILL_QKNORM in ("fp32", "fused")' in patch
    settings = (ROOT / "scripts/prepare-firmus-pp4-settings.py").read_text()
    assert '"SGLANG_PREFILL_QKNORM": "fp32"' in settings


@pytest.mark.parametrize("env", ["control", "fp32", "fused"])
def test_only_exact_fused_routes(env):
    # Execute the actual added routing statements from the qwen method.
    lines = PATCH.read_text().splitlines()
    start = next(i for i, x in enumerate(lines) if x == '+        if _PREFILL_QKNORM == "fused":')
    block = "\n".join(x[1:] for x in lines[start:start + 6])
    tree = ast.parse("def route():\n" + block)
    check = tree.body[0].body[0].test
    assert eval(compile(ast.Expression(check), "routing", "eval"), {"_PREFILL_QKNORM": env}) == (env == "fused")


def test_wgmma_ownership_and_neox_pairs():
    seen = set()
    for t, wave, r, row, col, index in REF["ownership"]():
        assert (row, col) not in seen
        seen.add((row, col))
        assert index // 64 == wave
        assert index % 4 // 2 == r
        assert ((col + 64) % 128) % 8 == col % 8
        assert t % 4 == col % 8 // 2
    assert len(seen) == 256 * 128


@pytest.mark.parametrize("m", [1, 52, 129, 255, 256, 3994, 7988])
def test_m_tail_exact_coverage(m):
    rows = [row for _, _, _, row, col, _ in REF["ownership"]() if col == 0]
    valid = [tile * 256 + row for tile in range((m + 255) // 256) for row in rows if tile * 256 + row < m]
    assert sorted(valid) == list(range(m))


def test_quad_register_transpose():
    words = np.arange(16).reshape(4, 4)
    original = words.copy()
    for bit in (1, 2):
        before = words.copy()
        for lane in range(4):
            for reg in range(4):
                if (lane & bit) != (reg & bit):
                    words[lane, reg] = before[lane ^ bit, reg ^ bit]
    np.testing.assert_array_equal(words, original.T)


def test_random_fp32_tiles_against_fp64():
    rng = np.random.default_rng(20261031)
    acc = rng.normal(size=(4096, 128)).astype(np.float32) * 1.7
    weight = REF["bf16_round"](rng.uniform(0.3, 3, 128).astype(np.float32))
    angle = rng.uniform(-20, 20, (4096, 64))
    trig = np.concatenate((np.cos(angle), np.sin(angle)), -1).astype(np.float32)
    fused, codes = REF["epilogue"](acc, weight, trig)
    exact, reference = REF["epilogue"](acc, weight, trig, fp64=True)
    control, control_codes = REF["epilogue"](acc, weight, trig, bf16_input=True)
    assert np.max(np.abs(fused - exact)) < 4e-6
    assert np.mean(np.abs(fused - exact)) < np.mean(np.abs(control - exact))
    assert np.mean(codes != reference) <= np.mean(control_codes != reference)
    assert np.mean(codes != reference) < 1e-5


def test_nosat_special_values_and_ties():
    values = np.array([0.0, -0.0, 448, -448, 464, -464, 465, -465, np.inf, -np.inf, np.nan, -np.nan, 1.0625, 1.1875])
    np.testing.assert_array_equal(REF["fp8_codes"](values), [0, 128, 126, 254, 126, 254, 127, 255, 127, 255, 127, 127, 56, 58])


def test_full_overlay_chain_from_prepared_image(tmp_path):
    import os

    image = os.environ.get("QWEN_SGLANG_IMAGE_SOURCE")
    if not image:
        pytest.skip("requires the prepared image source, checked explicitly before dispatch")
    tree = tmp_path / "sglang"
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", "--branch", "image", image, str(tree)], check=True)
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    for overlay in config["sglang_overlay_patches"]:
        subprocess.run(["git", "apply", "--check", str(ROOT / overlay)], cwd=tree, check=True)
        subprocess.run(["git", "apply", str(ROOT / overlay)], cwd=tree, check=True)
    assert (tree / "python/sglang/kernels/jit/csrc/gemm/qkv_fused_prefill.cuh").is_file()
    ast.parse((tree / "python/sglang/srt/models/qwen3_moe.py").read_text())
