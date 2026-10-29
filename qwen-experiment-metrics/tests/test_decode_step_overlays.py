"""GPU-free checks of the Decode-step overlays: SwapAB dense GEMM index logic,
the 512-lane input norm/quant order change, and the verify penalty broadcast."""
import json
import re
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
SWAPAB_PATCH = ROOT / "patches/sglang/decode-dense-swapab.patch"
NORM_PATCH = ROOT / "patches/sglang/decode-norm-quant-wide.patch"
PENALTY_PATCH = ROOT / "patches/sglang/verify-penalty-broadcast.patch"


def added(patch, path):
    """Lines a patch adds to one file."""
    text = patch.read_text()
    section = text.split(f"+++ b/{path}\n", 1)[1].split("\ndiff --git", 1)[0]
    return "\n".join(line[1:] for line in section.splitlines() if line.startswith("+"))


def bf16(x):
    """Round FP32 to BF16 (round to nearest even), returned as FP32."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16 << 16
    return bits.astype(np.uint32).view(np.float32)


def swapab_emulation(w, x, sw, sx, bn, split_k):
    """The kernel's data movement for one call, per the PTX layouts it relies on:
    WGMMA m64nNk32 accumulator fragments (warp w rows 16w.., lane l row l/4 (+8),
    columns 8i + 2(l%4) (+1)), the FP32 promotion, the cluster-rank-ordered
    split-K sum and group ownership, stmatrix.x4.trans into the 128B-swizzled
    token-major halves, and the TMA store that undoes the swizzle."""
    n, k = w.shape
    m = x.shape[0]
    xp = np.zeros((bn, k), np.float32)
    xp[:m] = x
    sxp = np.zeros((bn, k // 128), np.float32)
    sxp[:m] = sx
    groups = bn // 8
    lane = np.arange(32)
    d = np.arange(bn // 2)
    rows = (lane[:, None] // 4) + 8 * ((d[None, :] // 2) % 2)  # + 16 * warp
    cols = (d[None, :] // 4) * 8 + 2 * (lane[:, None] % 4) + d[None, :] % 2
    out = np.full((m, n), np.nan, np.float32)
    kb_per = k // 128 // split_k
    for tile in range(n // 128):
        finals = []
        for split in range(split_k):
            final = np.zeros((2, 4, 32, bn // 2), np.float32)
            for kb in range(split * kb_per, (split + 1) * kb_per):
                ks = slice(kb * 128, kb * 128 + 128)
                for wg in range(2):
                    partial = (w[tile * 128 + wg * 64:tile * 128 + wg * 64 + 64, ks].astype(np.float64) @ xp[:, ks].T.astype(np.float64)).astype(np.float32)
                    for warp in range(4):
                        acc = partial[16 * warp + rows, cols]
                        scale = (np.float32(sw[tile, kb]) * sxp[cols, kb]).astype(np.float32)
                        final[wg, warp] = (final[wg, warp].astype(np.float64) + scale.astype(np.float64) * acc).astype(np.float32)
            finals.append(final)
        summed = finals[0].copy()
        for r in range(1, split_k):
            summed = (summed + finals[r]).astype(np.float32)
        for split in range(split_k):
            pairs = groups // 2
            begin, end = (2 * (split * pairs // split_k), 2 * ((split + 1) * pairs // split_k)) if split_k > 1 else (0, groups)
            smem = np.full((2, bn, 64), np.nan, np.float32)
            for wg in range(2):
                for warp in range(4):
                    for i in range(begin, end, 2):
                        for lane_addr in range(32):
                            mat, j = lane_addr // 8, lane_addr % 8
                            token = 8 * (i + mat // 2) + j
                            chunk = 2 * warp + mat % 2
                            for a in range(8):
                                holder, e = 4 * a + j // 2, j % 2
                                smem[wg, token, (chunk ^ j) * 8 + a] = bf16(summed[wg, warp, holder, 4 * (i + mat // 2) + 2 * (mat % 2) + e])
            for wg in range(2):
                for i in range(begin, end):
                    for token in range(8 * i, min(8 * i + 8, m)):
                        for chunk in range(8):
                            position = chunk ^ (token % 8)
                            out[token, tile * 128 + wg * 64 + 8 * chunk:tile * 128 + wg * 64 + 8 * chunk + 8] = smem[wg, token, position * 8:position * 8 + 8]
    return out


@pytest.mark.parametrize("rows,split_k", [(1, 1), (7, 1), (129, 1), (176, 1), (9, 4), (144, 4), (176, 2)])
def test_swapab_fragment_epilogue_and_split_order_reproduce_the_gemm(rows, split_k):
    rng = np.random.default_rng(rows * 7 + split_k)
    n, k, bn = 256, 128 * 4, 176
    w = rng.integers(-8, 9, (n, k)).astype(np.float32)
    x = rng.integers(-8, 9, (rows, k)).astype(np.float32)
    sw = (2.0 ** rng.integers(-6, 0, (n // 128, k // 128))).astype(np.float32)
    sx = (2.0 ** rng.integers(-6, 0, (rows, k // 128))).astype(np.float32)
    out = swapab_emulation(w, x, sw, sx, bn, split_k)
    reference = sum((x[:, kb * 128:kb * 128 + 128] * sx[:, kb:kb + 1]) @ (w[:, kb * 128:kb * 128 + 128] * np.repeat(sw[:, kb], 128)[:, None]).T for kb in range(k // 128))
    assert not np.isnan(out).any()
    # Power-of-two scales and small integers keep every FP32 step exact: only the BF16 store rounds.
    assert np.array_equal(out, bf16(reference))


def test_swapab_kernel_source_matches_the_emulated_layout():
    source = added(SWAPAB_PATCH, "python/sglang/kernels/jit/csrc/gemm/decode_swapab_fp8.cuh")
    assert "const uint32_t token = 8 * (i + m / 2) + j;" in source
    assert "const uint32_t chunk = 2 * (warp_idx % 4) + m % 2;" in source
    assert "smem_d + token * 128 + ((chunk ^ j) * 16)" in source
    assert "group_begin = 2 * (split * kPairs / kSplitK);" in source and "group_end = 2 * ((split + 1) * kPairs / kSplitK);" in source
    assert "const float s0 = scale_w * s.x, s1 = scale_w * s.y;" in source
    # Scales are read before the stage is released to the producer.
    loop = source.split("ptx::warpgroup_wait<0>();", 1)[1]
    assert loop.index("smem_sx(stage)") < loop.index("empty_barriers[stage].arrive")
    # Weight scales may be read before the grid dependency; activations only after.
    assert source.index("__ldg(sw +") < source.index("cudaGridDependencySynchronize();") < source.index("tma::copy<kSwapBlockK, BN, 128>")


def test_swapab_backends_are_selection_candidates_with_a_serving_row_guard():
    python = added(SWAPAB_PATCH, "python/sglang/srt/layers/quantization/decode_dense_gemm.py")
    assert '"QKV": {"swapab-176-s5": (176, 5, 1, 1), "swapab-176-s5-mc2": (176, 5, 1, 2)}' in python
    assert '"O": {"swapab-176-s5-k4": (176, 5, 4, 1)}' in python
    assert "SWAPAB_MIN_ROWS = 129" in python
    checks = (ROOT / "scripts/check-firmus-decode-kernels.py").read_text()
    names = re.search(r"DENSE_SWAPAB = (\{.*\})", checks).group(1)
    assert names == '{"qkv": ("swapab-176-s5", "swapab-176-s5-mc2"), "o": ("swapab-176-s5-k4",)}'


def fp8_round(values):
    """Round FP32 magnitudes to E4M3 (nearest even, finite range), keeping signs."""
    grid = np.array(sorted({(1 + mant / 8) * 2.0 ** (exp - 7) for exp in range(1, 16) for mant in range(8)} | {mant * 2.0 ** -9 for mant in range(8)}))
    grid = grid[grid <= 448]
    magnitude = np.abs(values)
    index = np.clip(np.searchsorted(grid, magnitude), 1, len(grid) - 1)
    low, high = grid[index - 1], grid[index]
    choose_high = (high - magnitude < magnitude - low) | ((high - magnitude == magnitude - low) & (index % 2 == 0))
    return np.sign(values) * np.where(choose_high, high, low)


def norm_quant(h, weight, order):
    """Per-row FP32 squares (FMA rounding), then the kernel's summation order."""
    squares = h.astype(np.float64) ** 2
    if order == "stock":
        # 64 lanes x 64 values: lane-sequential FMA chain, 5-step XOR tree, 2 warps.
        lanes = h.reshape(8, 64, 8).transpose(1, 0, 2).reshape(64, 64)
        partial = np.zeros(64, np.float32)
        for value in lanes.T:
            partial = (partial.astype(np.float64) + value.astype(np.float64) ** 2).astype(np.float32)
        warps = partial.reshape(2, 32)
    else:
        # 512 lanes x 8 values, 16 warps summed in index order.
        partial = np.zeros(512, np.float32)
        for value in h.reshape(512, 8).T:
            partial = (partial.astype(np.float64) + value.astype(np.float64) ** 2).astype(np.float32)
        warps = partial.reshape(16, 32)
    for mask in (1, 2, 4, 8, 16):
        warps = (warps + warps[:, np.arange(32) ^ mask]).astype(np.float32)
    total = warps[0, 0]
    for w in warps[1:, 0]:
        total = np.float32(total + w)
    assert np.isfinite(squares.sum())
    rstd = np.float32(1 / np.sqrt(np.float64(np.float32(total / 4096 + 1e-6))))
    y = bf16((h * rstd).astype(np.float32) * weight).reshape(32, 128)
    amax = np.maximum(np.abs(y).max(axis=1), 1e-10).astype(np.float32)
    q = fp8_round(np.minimum(y * (np.float32(448) / amax)[:, None], 448))
    return q, (amax / np.float32(448)).astype(np.float32)


def test_wide_norm_order_stays_within_the_decode_gate_bounds():
    rng = np.random.default_rng(1764096)
    weight = bf16(rng.standard_normal(4096).astype(np.float32) * 0.1 + 1)
    differing = 0
    for _ in range(64):
        h = (bf16(rng.standard_normal(4096).astype(np.float32)) + bf16(rng.standard_normal(4096).astype(np.float32) * 8)).astype(np.float32)
        q0, s0 = norm_quant(h, weight, "stock")
        q1, s1 = norm_quant(h, weight, "wide")
        scale_rel = np.max(np.abs(s1 - s0) / s0)
        a, b = (q1 * s1[:, None]), (q0 * s0[:, None])
        bound = np.maximum(np.abs(a), np.abs(b)) * (2.0 ** -3 + scale_rel) + 2.0 ** -9 * np.maximum(s0, s1)[:, None]
        assert scale_rel <= 2.0 ** -7
        assert (np.abs(a - b) <= bound).all()
        differing += int((q0 != q1).sum())
    assert differing < 64 * 4096 // 100


def test_wide_norm_kernel_keeps_the_stock_element_arithmetic_and_selection_wiring():
    header = added(NORM_PATCH, "python/sglang/kernels/jit/csrc/elementwise/add_rmsnorm_group_quant.cuh")
    for line in ("h[2 * j] = __fadd_rn(xf.x, rf.x);", "__fmul_rn(__fmul_rn(h[2 * j], rstd), __fadd_rn(wf.x, 0.0f)),", "const float quant_scale = kFP8Max / amax;", "warp::reduce_max<16>(amax2)", "if (t % 16 == 0) params.scale[(t / 16) * params.scale_group_stride + row] = amax * kFP8MaxInv;"):
        assert line in header
    # Only the weight load precedes the grid dependency wait.
    before_wait = header.split("PDLWaitPrimary<kUsePDL>();", 1)[0]
    assert "load_as<Vec>(params.weight, t)" in before_wait and "params.x" not in before_wait and "params.residual" not in before_wait
    python = added(NORM_PATCH, "python/sglang/kernels/ops/layernorm/add_rmsnorm_group_quant.py")
    assert 'DECODE_NORM_QUANT_KERNELS = {"": decode_add_rmsnorm_group_quant_fp8_out, "wide": decode_add_rmsnorm_group_quant_wide_fp8_out}' in python
    assert 'os.environ.get("SGLANG_DECODE_NORM_QUANT", "")' in python
    checks = (ROOT / "scripts/check-firmus-decode-kernels.py").read_text()
    assert 'NORM_QUANT_CANDIDATES = ("stock", "wide")' in checks


def test_selection_writes_the_norm_choice_only_when_it_wins():
    import runpy

    selector = runpy.run_path(str(ROOT / "scripts/select-firmus-decode-dense.py"))
    stock = {"name": "stock", "backend": "deepgemm", "config": None}
    candidates = {"qkv": [stock], "o": [stock], "norm": [{"name": "stock"}, {"name": "wide"}]}
    for winner, expected in (("stock", ""), ("wide", "wide")):
        selections = {"qkv": {"winner": "stock"}, "o": {"winner": "stock"}, "norm": {"winner": winner}}
        env = selector["worker_environment"](selections, candidates)
        assert env["SGLANG_DECODE_NORM_QUANT"] == expected
        assert env["SGLANG_DECODE_DENSE_GEMM_QKV"] == env["SGLANG_DECODE_DENSE_GEMM_O"] == env["DG_SM90_1D2D_DECODE_CONFIG"] == ""


def test_penalty_broadcast_equals_the_repeated_copy_bitwise():
    rng = np.random.default_rng(5)
    bs, draft, vocab = 5, 4, 97
    logits = rng.standard_normal((bs * draft, vocab)).astype(np.float32)
    logits[0, 0] = -0.0
    penalties = np.where(rng.random((bs, vocab)) < 0.05, -np.inf, 0).astype(np.float32)
    penalties[1] = 0.0
    repeated = logits + np.repeat(penalties, draft, axis=0)
    broadcast = (logits.reshape(bs, draft, vocab) + penalties[:, None, :]).reshape(bs * draft, vocab)
    assert np.array_equal(repeated.view(np.uint32), broadcast.view(np.uint32))
    python = added(PENALTY_PATCH, "python/sglang/srt/speculative/eagle_utils.py")
    assert "next_token_logits.unflatten(0, (bs, verify_input.draft_token_num)).add_(" in python
    assert "sampling_info.acc_additive_penalties.unsqueeze(1)" in python


@pytest.mark.parametrize("config", ["configs/firmus-framework.json", "configs/firmus-prefill-pp4.json"])
def test_decode_step_overlays_follow_the_patches_they_build_on(config):
    overlays = json.loads((ROOT / config).read_text())["sglang_overlay_patches"]
    for patch, base in (("decode-norm-quant-wide", "decode-add-rmsnorm-group-quant"), ("decode-dense-swapab", "decode-dense-gemm"), ("verify-penalty-broadcast", None)):
        path = f"patches/sglang/{patch}.patch"
        assert overlays.count(path) == 1
        if base:
            assert overlays.index(path) > overlays.index(f"patches/sglang/{base}.patch")
