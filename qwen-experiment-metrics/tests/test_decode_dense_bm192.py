import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_host_series_ends_with_the_bm192_override_and_keeps_the_pinned_config_patch():
    series = (ROOT / "patches/deep-gemm-host/series").read_text().split()
    assert series.index("sm90-1d2d-decode-config.patch") < series.index("sm90-1d2d-decode-bm192.patch") < series.index("sm90-prefill-fc1-first-k.patch") < series.index("sm90-prefill-dense-first-k-r37.patch")
    patch = (ROOT / "patches/deep-gemm-host/sm90-1d2d-decode-bm192.patch").read_text()
    # Only "QKV>128"/"O>128" entries may use BLOCK_M 192, only above 128 rows, with three math warpgroups.
    for line in ('+        DG_HOST_ASSERT(name == "QKV" or name == "O" or name == "QKV>128" or name == "O>128");', "+        DG_HOST_ASSERT(bm == 64 or bm == 128 or (bm == 192 and large));", "+            launch.num_math_threads = 384, launch.num_threads = launch.num_tma_threads + 384;", "+        if (qkv == (desc.n == 9216) and large and desc.m > 128)"):
        assert line in patch, line
    # The dense selection pins the bytes of the original override patch, which stays unchanged.
    checks = (ROOT / "scripts/check-firmus-decode-kernels.py").read_text()
    assert 'record["patches"].get("sm90-1d2d-decode-config.patch", "")' in checks


def test_header_overlay_changes_only_the_three_warpgroup_geometry():
    pp4 = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    patches = pp4["deep_gemm_overlay_patches"]
    assert patches.index("patches/deep-gemm-bench/decode-dense-bm192/sm90-1d2d-decode-bm192.patch") < patches.index("patches/deep-gemm-overlay/sm90-prefill-fc1-first-k.patch") < patches.index("patches/deep-gemm-overlay/sm90-prefill-dense-first-k-r37.patch")
    patch = (ROOT / "patches/deep-gemm-bench/decode-dense-bm192/sm90-1d2d-decode-bm192.patch").read_text()
    assert "kNumMathThreads == 128 ? 248 : (kNumMathThreads == 384 ? 152 : 232)" in patch
    assert "WGMMA::M * (kNumMathThreads == 384 ? 3 : 2)" in patch
    # 3 x 128 x 152 + 128 x 40 registers fit the 64K register file.
    assert 3 * 128 * 152 + 128 * 40 <= 65536
    assert sum(line.startswith("+") and not line.startswith("+++") for line in patch.splitlines()) == 5


def test_bm192_candidates_cover_one_wave_of_qkv_tiles():
    checks_text = (ROOT / "scripts/check-firmus-decode-kernels.py").read_text()
    # N80: 116 tiles, one wave on 132 SMs. N72 (128 tiles) failed the host's bn % 16 == 0 assertion in
    # 40964714831 before any timing; every candidate must satisfy the host checks of sm90-1d2d-decode-config.
    assert 'DENSE_LARGE = {"qkv": ((192, 80, 5, 1, 0),), "o": ()}' in checks_text
    host = (ROOT / "patches/deep-gemm-host/sm90-1d2d-decode-config.patch").read_text()
    assert "DG_HOST_ASSERT(bn >= 16 and bn <= 192 and bn % 16 == 0);" in host
    for bm, bn, *_ in ast.literal_eval(checks_text.split("DENSE_LARGE = ", 1)[1].split("\n", 1)[0])["qkv"]:
        assert 16 <= bn <= 192 and bn % 16 == 0 and bm == 192
        assert -(-9216 // bn) <= 132
