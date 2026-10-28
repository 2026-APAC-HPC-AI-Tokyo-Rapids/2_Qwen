"""CPU fixtures for stage attribution from standalone Decode probe dumps."""
import ast
import copy
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "kernels/probes/fa3-decode-merge.py"
API = runpy.run_path(str(DRIVER))


def dumps():
    batch, heads, splits, tq = 2, 4, 3, 8
    rng = np.random.default_rng(93141)
    partial = rng.normal(size=(splits, 64, tq, 128)).astype(np.float32)
    plse = rng.normal(size=(splits, 64, tq)).astype(np.float32)
    merged = API["fp32_merge"](partial, plse)
    codes, scales, amax = API["fp8_quantize"](merged)
    records = np.zeros((batch, heads, splits, 12), dtype=np.int32)
    for b in range(batch):
        for h in range(heads):
            for s in range(splits):
                old = (s + h) % splits
                records[b, h, s] = [b + h, 0, 0, 0, h, b, (splits << 16) + s, splits, s, old, int(old == splits - 1), 1]
    metadata = np.zeros(4 * batch + 1 + batch * heads, dtype=np.int32)
    metadata[:batch] = splits
    metadata[batch:2 * batch] = 1
    metadata[2 * batch:3 * batch] = np.arange(batch)[::-1]
    metadata[3 * batch:4 * batch] = 4
    abi = np.arange(28, dtype=np.int64)
    data = {"records": records, "abi": np.broadcast_to(abi, (batch, heads, splits, 28)).copy(), "expected_abi": abi, "prepare": metadata.copy(), "metadata": metadata.copy(), "partial": partial, "partial_lse": plse, "loaded": partial.copy(), "loaded_lse": np.stack((plse, plse)), "merged": merged, "codes": codes, "quant_scales": scales, "amax": amax, "output": codes.copy(), "scales": scales.copy()}
    control, merge = copy.deepcopy(data), copy.deepcopy(data)
    control["records"][..., :3] += 7  # CTA assignment is intentionally unrelated.
    control["records"][..., 9:11] = -1  # No arrivals in separate combine.
    return control, merge


def test_matching_dumps_allow_different_ctas_and_arrival_order():
    result = API["analyze"](*dumps())
    assert result["first_divergence"] is None, result["issues"]
    assert not any(result["issues"].values())


@pytest.mark.parametrize("key,stage", [
    ("records", "records"), ("abi", "records"), ("prepare", "records"), ("metadata", "records"),
    ("partial", "workspace"), ("partial_lse", "workspace"),
    ("loaded", "merge_snapshot"), ("loaded_lse", "merge_snapshot"),
    ("merged", "merged_fp32"), ("amax", "fp8_bytes_scales"),
    ("output", "fp8_bytes_scales"), ("scales", "fp8_bytes_scales"),
    ("codes", "fp8_bytes_scales"), ("quant_scales", "fp8_bytes_scales"),
])
def test_planted_divergence_is_attributed_to_first_stage(key, stage):
    control, merge = dumps()
    if key == "records":
        merge[key][0, 0, 0, 3] += 1  # Corrupt m; a different CTA is valid.
    else:
        merge[key].flat[0] += 1
    result = API["analyze"](control, merge)
    assert result["first_divergence"] == stage, result["issues"]
    assert result["issues"][stage][0]["first"]


@pytest.mark.parametrize("field", [6, 7, 8, 9, 10, 11])
def test_invalid_coordinate_or_arrival_record(field):
    control, merge = dumps()
    merge["records"][0, 0, 0, field] += 1
    assert API["analyze"](control, merge)["first_divergence"] == "records"


def test_poisoned_workspace_is_never_equal_even_when_both_paths_are_nan():
    control, merge = dumps()
    control["partial"].flat[0] = merge["partial"].flat[0] = np.nan
    assert API["analyze"](control, merge)["first_divergence"] == "workspace"


def test_quantization_uses_e4m3fn_codes_and_fp32_scales():
    values = np.array([[[0., -0., 1., -1., 448., -448., 2.**-9, 1.0625, 1.1875]]], dtype=np.float32)
    codes, scales, amax = API["fp8_quantize"](values)
    assert codes.tolist() == [[[0, 128, 56, 184, 126, 254, 1, 56, 58]]]
    assert scales.item() == 1 and amax.item() == 448


def test_dump_can_be_reanalyzed_without_torch(tmp_path):
    control, merge = dumps()
    merge["loaded"][0, 0, 0, 0] += 1
    dump = tmp_path / "dump.npz"
    np.savez(dump, **{prefix + "_" + key: value for prefix, data in (("control", control), ("merge", merge)) for key, value in data.items()})
    result = subprocess.run([sys.executable, str(DRIVER), "--analyze", str(dump), "--results", str(tmp_path)], capture_output=True, text=True, check=True)
    assert "diagnostic_completed=true first_divergence=merge_snapshot" in result.stdout
    assert json.loads((tmp_path / "fa3-decode-merge.json").read_text())["first_divergence"] == "merge_snapshot"
    assert "loaded_O_vs_own_workspace" in (tmp_path / "fa3-decode-merge.log").read_text()
    assert "torch" not in API


def test_login_build_is_python36_syntax_and_uses_serving_build_api():
    for path in (ROOT / "kernels/probes/fa3-decode-merge-build.py", ROOT / "kernels/fa3-verify/build.py"):
        ast.parse(path.read_text(), feature_version=(3, 6))
    source = (ROOT / "kernels/probes/fa3-decode-merge-build.py").read_text()
    for function in ("prepare_sources", "generate_dispatch", "cuda_flags"):
        assert 'api["' + function + '"]' in source
    assert '"-DFA3_DECODE_MERGE_PROBE=1"' in source
    assert "FA3_DECODE_MERGE_PROBE" not in (ROOT / "kernels/fa3-verify/build.py").read_text()
