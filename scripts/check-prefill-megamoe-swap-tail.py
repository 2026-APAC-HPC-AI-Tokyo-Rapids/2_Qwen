"""CPU accounting for the r26 N16/N32 experiment; not a Tensor Core bit proof."""
import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_schedule():
    spec = importlib.util.spec_from_file_location("schedule", ROOT / "qwen-experiment-metrics/tests/test_prefill_megamoe_tail_pair.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def width(valid):
    return 16 if 0 < valid <= 16 else 32 if valid <= 32 and valid > 0 else 0


def fragment(n):
    """Production forward map: (fragment, thread, register) -> (token, feature)."""
    for f in range(4):
        for thread in range(128):
            warp, lane = divmod(thread, 32)
            for i in range(n // 8):
                for a in range(4):
                    yield (f, thread, i * 4 + a), (i * 8 + lane % 4 * 2 + a % 2, f * 64 + warp * 16 + lane // 4 + a // 2 * 8)


def inverse(token, feature, n):
    """Independent coordinate-first inverse of the PTX m64nN accumulator layout."""
    f, row = divmod(feature, 64)
    warp, row = divmod(row, 16)
    hi, lane_row = divmod(row, 8)
    group, col = divmod(token, 8)
    lane_col, parity = divmod(col, 2)
    assert token < n
    return f, warp * 32 + lane_row * 4 + lane_col, group * 4 + hi * 2 + parity


def normal_fragment(valid):
    """Unchanged split m64n128 x2, expressed in token-major coordinates."""
    for wg in range(2):
        for thread in range(128):
            warp, lane = divmod(thread, 32)
            for chunk in range(32):
                for a in range(4):
                    token = wg * 64 + warp * 16 + lane // 4 + a // 2 * 8
                    if token < valid:
                        yield (wg, thread, chunk * 4 + a), (token, chunk * 8 + lane % 4 * 2 + a % 2)


@lru_cache(None)
def check_coordinates(valid):
    n = width(valid)
    if not n:
        coords = [coord for _, coord in normal_fragment(valid)]
        assert Counter(coords) == Counter({(t, c): 1 for t in range(valid) for c in range(256)})
        outputs = Counter((t, c // 16 * 8 + c % 8) for t, c in coords if c % 16 < 8)
        assert outputs == Counter({(t, c): 1 for t in range(valid) for c in range(128)})
        return valid * 384
    observed = {}
    for key, coord in fragment(n):
        assert inverse(*coord, n) == key
        assert coord not in observed
        observed[coord] = key
    assert set(observed) == {(t, c) for t in range(n) for c in range(256)}
    fc1 = Counter()
    for (token, feature), (f, thread, reg) in observed.items():
        if token >= valid or feature % 16 >= 8:
            continue
        # Packed gate/up: eight gate then eight up, in each 16 features.
        logical_gate = feature // 16 * 8 + feature % 8
        up = observed[token, feature + 8]
        assert up == (f, thread, reg + 2)
        assert logical_gate == f * 32 + thread // 32 * 8 + thread % 32 // 4
        fc1[token, logical_gate] += 1
    assert fc1 == Counter({(t, c): 1 for t in range(valid) for c in range(128)})
    assert Counter(c for (t, c) in observed if t < valid) == Counter({c: valid for c in range(256)})
    return valid * (256 + 128)


def check_scales_and_swizzle():
    # Byte-coordinate model: 128B swizzle, rows of 128 FP8 K elements.
    def physical(row, k):
        return row * 128 + (k ^ ((row % 8) * 16))

    checked = 0
    for row in range(256):
        for k in range(128):
            # Independent 16-byte sector model, also used for swapped descriptors.
            sector, byte = divmod(k, 16)
            assert physical(row, k) == (row * 8 + (sector ^ (row & 7))) * 16 + byte
            f, r = divmod(row, 64)
            assert physical(row, k) == f * 8192 + physical(r, k)
            checked += 1
    for n in (16, 32):
        for (_, _, a), (token, feature) in fragment(n):
            for kb in range(32):
                # Logical feature owns the weight scale even after A/B exchange.
                gate = feature % 16 < 8
                logical = feature // 16 * 8 + feature % 8
                original_fc1 = ((0 if gate else 1536) + logical) // 128, kb
                swapped_fc1 = (0 if a % 4 < 2 else 12), kb
                assert original_fc1 == swapped_fc1
                assert (feature // 128, kb) == ((feature // 64) // 2, kb)
                # Decode's per-64 activation coordinate would differ in FC2.
                for k32 in range(4):
                    global_k = kb * 128 + k32 * 32
                    assert kb * 128 + token == (global_k // 128) * 128 + token
                checked += 1
    return checked


def check_schedule(distributions):
    m = load_schedule()
    total_items, total_stages, tails, saved_rows = 0, 0, 0, 0
    for seed, rows in enumerate(distributions):
        blocks = m.pool_blocks(rows)
        arrivals = Counter()
        for stream in m.streams(blocks, seed):
            traces = [m.role_uses(stream, blocks, False, role) for role in "abm"]
            assert [(p, k) for p, k, _ in traces[0]] == [(p, k) for p, k, _ in traces[1]] == [(p, k) for p, k, _ in traces[2]]
            position = 0
            pending = [None] * 8
            published = Counter()
            for phase, (b, _) in stream:
                valid = blocks[b][2]
                ks = 32 if phase == "fc1" else 12
                # First FC2 fetch publishes the last FC1 tile, including empty WG.
                if phase == "fc2":
                    for w in range(8):
                        if pending[w] is not None:
                            published[pending[w]] += 1
                            pending[w] = None
                consumer_uses = []
                for k in range(ks):
                    stage, parity = position % 4, position // 4 % 2
                    consumer_uses.append((stage, parity))
                    if phase == "fc1" and k == 2:
                        for w in range(8):
                            if pending[w] is not None:
                                published[pending[w]] += 1
                                pending[w] = None
                    position += 1
                # The unchanged producer advances one ring slot per K128. Swap
                # WG0 and padding-only WG1 must consume exactly that sequence.
                start = position - ks
                producer_uses = [((start + k) % 4, ((start + k) // 4) & 1) for k in range(ks)]
                assert consumer_uses == producer_uses
                releases = Counter((stage, parity, w) for stage, parity in consumer_uses for w in range(8))
                expected_releases = Counter((stage, parity, w) for stage, parity in producer_uses for w in range(8))
                assert releases == expected_releases
                if phase == "fc1":
                    for w in range(8):
                        if (w // 4) * 64 < valid:
                            assert pending[w] is None
                            pending[w] = b
                total_items += 1
                total_stages += ks
            # Dynamic driver invokes the FC2-fetch hook even with no FC2 items.
            for p in pending:
                if p is not None:
                    published[p] += 1
            arrivals.update(published)
        for b, (_, _, valid) in enumerate(blocks):
            assert arrivals[b] == 12 * ((valid + 63) // 64) * 4
            check_coordinates(valid)
            if width(valid):
                tails += 1
                saved_rows += 64 - width(valid)
    return {"distributions": len(distributions), "items": total_items, "stage_uses": total_stages, "eligible_pool_blocks": tails, "saved_rows": saved_rows}


def numerical():
    import numpy as np

    rng = np.random.default_rng(26)
    table = np.array([((v & 7) / 8 + (1 if v >> 3 else 0)) * 2.0 ** ((v >> 3) - 7 if v >> 3 else -6) for v in range(127)], dtype=np.float32)

    def fp8(x):
        distances = np.abs(np.abs(x)[..., None] - table)
        # RNE: even mantissa wins a tie, then take the unsigned code.
        pick = np.argmin(distances, -1)
        tie = (pick % 2 == 1) & (pick < 126) & (np.take_along_axis(distances, pick[..., None], -1)[..., 0] == np.take_along_axis(distances, np.minimum(pick + 1, 126)[..., None], -1)[..., 0])
        return (pick + tie + np.signbit(x) * 128).astype(np.uint8)

    def quantize(x):
        amax = np.maximum(np.max(np.abs(x), -1), np.float32(1e-10))
        sf = np.multiply(amax, np.float32(1 / 448), dtype=np.float32)
        inv = np.divide(np.float32(1), sf, dtype=np.float32)
        return fp8(np.multiply(x, inv[:, None], dtype=np.float32)), sf

    worst = 0.0
    for valid in range(1, 33):
        for magnitude, fc1, full in ((m, f, full) for m in (0.0, 0.25, 10.0) for f in (False, True) for full in (False, True)):
            k = (4096 if fc1 else 1536) if full else 256
            blocks = k // 128
            x = rng.integers(-4, 5, (valid, k)).astype(np.float32)
            if magnitude == 0.0:
                x.fill(0)
            w = rng.integers(-4, 5, (256, k)).astype(np.float32)
            sx = rng.uniform(0.002, 0.01, (valid, blocks)).astype(np.float32) * np.float32(magnitude)
            sx = np.maximum(sx, np.float32(1e-10))
            sw = rng.uniform(0.002, 0.01, (2, blocks)).astype(np.float32)
            weight_groups = (np.arange(256) // 8) % 2 if fc1 else np.arange(256) // 128
            # Explicit FP32 carried-scale algorithm, K128 then four K32 groups.
            acc = np.zeros((valid, 256), dtype=np.float32)
            prev = None
            for kb in range(blocks):
                scales = sx[:, kb, None] * sw[weight_groups, kb][None, :]
                if prev is not None:
                    acc = acc * (prev * np.divide(np.float32(1), scales, dtype=np.float32))
                for k32 in range(4):
                    start = kb * 128 + k32 * 32
                    acc = np.add(acc, x[:, start:start + 32] @ w[:, start:start + 32].T, dtype=np.float32)
                prev = scales
            acc *= prev
            # Independent NumPy reference: dequantize whole blocks before matmul.
            reference = np.zeros_like(acc)
            for kb in range(blocks):
                reference += (x[:, kb * 128:(kb + 1) * 128] * sx[:, kb, None]) @ (w[:, kb * 128:(kb + 1) * 128] * sw[weight_groups, kb][:, None]).T
            error = float(np.max(np.abs(acc - reference)))
            worst = max(worst, error)
            np.testing.assert_allclose(acc, reference, atol=3e-6, rtol=3e-5)
            # Separate weight-as-A calculation: four m64 fragments, scales still
            # owned by the original token and logical gate/up or FC2 feature.
            swapped = np.zeros_like(acc)
            for f in range(4):
                tile = np.zeros((64, valid), dtype=np.float32)
                previous = None
                rows = slice(f * 64, (f + 1) * 64)
                for kb in range(blocks):
                    scale = sx[:, kb][None, :] * sw[weight_groups[rows], kb][:, None]
                    if previous is not None:
                        tile *= previous * np.divide(np.float32(1), scale, dtype=np.float32)
                    for k32 in range(4):
                        start = kb * 128 + k32 * 32
                        tile += w[rows, start:start + 32] @ x[:, start:start + 32].T
                    previous = scale
                tile *= previous
                swapped[:, rows] = tile.T
            assert acc.tobytes() == swapped.tobytes()
            # Scatter/gather the actual swapped fragments, then gate/up pair.
            regs = {key: swapped[coord] if coord[0] < valid else np.float32(0) for key, coord in fragment(width(valid))}
            restored = np.array([[regs[inverse(t, c, width(valid))] for c in range(256)] for t in range(valid)], dtype=np.float32)
            assert acc.tobytes() == restored.tobytes()
            gate = acc.reshape(valid, 16, 16)[:, :, :8].reshape(valid, 128)
            up = restored.reshape(valid, 16, 16)[:, :, 8:].reshape(valid, 128)
            out = ((gate * np.divide(np.float32(1), np.float32(1) + np.exp(-gate), dtype=np.float32)) * up).astype(np.float32)
            q, sf = quantize(out)
            # Four fragments, four warps, eight features each: 128-wide reduction.
            per_warp = np.max(np.abs(out.reshape(valid, 4, 4, 8)), axis=(1, 3))
            amax = np.max(per_warp, -1)
            expected_sf = np.maximum(amax, np.float32(1e-10)) * np.float32(1 / 448)
            assert sf.tobytes() == expected_sf.tobytes()
            assert q.tobytes() == fp8(out * np.divide(np.float32(1), expected_sf)[:, None]).tobytes()
            # BF16 RNE at the postscale point; no rounding of partial sums.
            bits = acc.view(np.uint32)
            bf16 = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            rbits = restored.view(np.uint32)
            assert bf16.tobytes() == ((rbits + 0x7fff + ((rbits >> 16) & 1)) >> 16).astype(np.uint16).tobytes()
    return {"cases": 384, "k_numeric": [256, 1536, 4096], "exact_zero_input_included": True, "max_abs_error_to_independent_fp32_numpy": worst, "tensor_core_internal_order_proved": False, "gpu_approx_ex2_rcp_proved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schedule = load_schedule()
    archived = schedule.archived_distributions()
    assert len(archived) == 36
    # Every remainder, on either side of the 64/128 boundary, plus empties.
    edges = [[0, r, 64 + r, 128 + r] * 32 for r in range(1, 65)]
    report = {"schema": "prefill-megamoe-swap-tail-cpu/v1", "archived": check_schedule(archived), "all_remainders": check_schedule(edges), "scale_swizzle_coordinates": check_scales_and_swizzle(), "numerical": numerical(), "patch_sha256": hashlib.sha256((ROOT / "patches/deep-gemm/sm90-prefill-swap-tail.patch").read_bytes()).hexdigest(), "gpu_executed": False}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
