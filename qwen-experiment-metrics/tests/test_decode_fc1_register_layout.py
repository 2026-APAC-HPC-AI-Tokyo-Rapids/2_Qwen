"""CPU topology checks for the SwapAB register-retained FC1 candidate.

These prove lane ownership, amax grouping and byte packing, not GPU arithmetic
or memory ordering. GPU bitwise/replay and compiler resource gates remain required.
"""
import math
import random


def cute_max(a, b):
    return b if a < b else a  # noqa: FURB136 - model cute::max NaN semantics explicitly


def amax_index(warp, token):
    return warp * 64 + (token ^ (warp * 4))


def warp_amax(values):
    values = [cute_max(0.0, abs(v)) for v in values]
    for mask in (4, 8, 16):
        values = [cute_max(v, values[lane ^ mask]) for lane, v in enumerate(values)]
    return values


def test_fc1_register_layout_all_partial_tiles():
    rng = random.Random(20261023)
    tile = [[rng.uniform(-128.0, 128.0) for _ in range(64)] for _ in range(64)]
    # NaN ignores, infinity, signed zero, and maxima in the other warpgroup.
    tile[0] = [math.nan] * 64
    tile[1] = [-0.0] * 64
    tile[2][63] = math.inf
    tile[3][47] = 1e20
    for valid_m in range(1, 65):
        scratch = {}
        owners = set()
        for warp in range(8):
            for chunk in range((valid_m + 7) // 8):
                for parity in (0, 1):
                    tokens = [chunk * 8 + (lane % 4) * 2 + parity for lane in range(32)]
                    values = [tile[token][warp * 8 + lane // 4] if token < valid_m else 0.0 for lane, token in enumerate(tokens)]
                    maxima = warp_amax(values)
                    for lane in range(4):
                        index = amax_index(warp, tokens[lane])
                        assert index not in scratch
                        scratch[index] = maxima[lane]
                    for lane, token in enumerate(tokens):
                        if token < valid_m:
                            owner = (token, warp * 8 + lane // 4)
                            assert owner not in owners
                            owners.add(owner)
        assert len(owners) == valid_m * 64
        assert max(scratch) < 512  # 2 KiB, disjoint from FP8 at byte offset 16384.
        for token in range(valid_m):
            expected = 0.0
            for value in tile[token]:
                expected = cute_max(expected, abs(value))
            quad = [cute_max(scratch[amax_index(2 * q, token)], scratch[amax_index(2 * q + 1, token)]) for q in range(4)]
            for mask in (1, 2):
                quad = [cute_max(value, quad[q ^ mask]) for q, value in enumerate(quad)]
            assert all(actual == expected for actual in quad)
        # Use arbitrary encoded FP8 bytes: the test checks routing/packing and
        # does not substitute a CPU quantizer for the unchanged CUDA converter.
        encoded = [[rng.randrange(256) for _ in range(64)] for _ in range(64)]
        stores = {}
        for warp in range(8):
            for chunk in range((valid_m + 7) // 8):
                for parity in (0, 1):
                    tokens = [chunk * 8 + (lane % 4) * 2 + parity for lane in range(32)]
                    values = [encoded[token][warp * 8 + lane // 4] for lane, token in enumerate(tokens)]
                    pairs = [value | (values[lane ^ 4] << 8) for lane, value in enumerate(values)]
                    for lane in range(32):
                        col = warp * 8 + lane // 4
                        token = tokens[lane]
                        if lane // 4 % 4 == 0 and token < valid_m:
                            packed = pairs[lane] | (pairs[lane ^ 8] << 16)
                            offset = token * 64 + col
                            assert offset % 4 == 0
                            for byte in range(4):
                                assert offset + byte not in stores
                                stores[offset + byte] = (packed >> (8 * byte)) & 255
        assert len(stores) == valid_m * 64
        assert all(stores[token * 64 + col] == encoded[token][col] for token in range(valid_m) for col in range(64))


def test_fc1_amax_scratch_bank_mapping():
    for warp in range(8):
        assert {amax_index(warp, token) for token in range(64)} == set(range(warp * 64, (warp + 1) * 64))
        for chunk in range(8):
            for parity in (0, 1):
                indices = [amax_index(warp, chunk * 8 + lane * 2 + parity) for lane in range(4)]
                assert len({index % 32 for index in indices}) == 4
    for warp in range(8):
        for partial_parity in (0, 1):
            indices = [amax_index((lane % 4) * 2 + partial_parity, warp * 8 + lane // 4) for lane in range(32)]
            assert len({index % 32 for index in indices}) == 32
