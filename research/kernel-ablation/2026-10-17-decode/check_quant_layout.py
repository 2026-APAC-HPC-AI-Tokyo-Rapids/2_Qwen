"""CPU checks of the two quantizers with no extra CTA barrier.

All 0..64 tails, zero/tiny rows, unique value/scale ownership, immutable
partial-max scratch and identical FP32 inputs to the FP8 conversion.
GPU numerical and race checks are still required before adoption.
"""
import random
import struct


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def reduce_columns(values):
    for offset in (4, 8, 16):
        values = [max(values[lane], values[lane ^ offset]) for lane in range(32)]
    return values


def check():
    rng = random.Random(713)
    for valid in range(65):
        rows = [[f32(rng.uniform(-500, 500)) for _ in range(64)] for _ in range(valid)]
        if valid:
            rows[0] = [0.0] * 64
        if valid > 1:
            rows[1] = [f32(1e-12)] * 64
        chunks = (valid + 7) // 8
        shared = [None] * 512
        swizzled = {}
        for i in range(chunks):
            for warp in range(8):
                for half in (0, 1):
                    values = []
                    for lane in range(32):
                        token, col = i * 8 + (lane % 4) * 2 + half, warp * 8 + lane // 4
                        value = rows[token][col] if token < valid else 0.0
                        values.append(abs(value))
                        if token < valid:
                            address = token * 64 + (col ^ ((token & 6) << 2))
                            assert address not in swizzled
                            swizzled[address] = value
                    maxima = reduce_columns(values)
                    for lane in range(4):
                        address = i * 64 + half * 32 + lane * 8 + warp
                        assert shared[address] is None
                        shared[address] = maxima[lane]
        snapshot = shared.copy()
        scales, output = {}, {}
        for i in range(chunks):
            for warp in range(8):
                for half in (0, 1):
                    addresses = [i * 64 + half * 32 + (lane % 4) * 8 + lane // 4 for lane in range(32)]
                    assert len({a % 32 for a in addresses}) == 32
                    maxima = reduce_columns([shared[a] for a in addresses])
                    for lane in range(32):
                        token, col = i * 8 + (lane % 4) * 2 + half, warp * 8 + lane // 4
                        if token >= valid:
                            continue
                        sf = f32(max(maxima[lane], f32(1e-10)) * f32(1.0 / 448.0))
                        if warp == 0 and lane // 4 == 0:
                            assert token not in scales
                            scales[token] = sf
                        assert (token, col) not in output
                        output[token, col] = f32(rows[token][col] * f32(1.0 / sf))
        assert shared == snapshot  # No overwrite; no second scratch-publication barrier.
        assert len(scales) == valid and len(output) == valid * 64
        for token, row in enumerate(rows):
            sf = f32(max(max(map(abs, row)), f32(1e-10)) * f32(1.0 / 448.0))
            assert scales[token] == sf
            loaded = [swizzled[token * 64 + (col ^ ((token & 6) << 2))] for col in range(64)]
            assert loaded == row
            assert max(map(abs, loaded)) == max(map(abs, row))
            for col, value in enumerate(row):
                expected = f32(value * f32(1.0 / sf))
                assert struct.pack("f", output[token, col]) == struct.pack("f", expected)
    print("PASS: register no-extra-barrier and swizzled warp quantizers, all 0..64 tails")


if __name__ == "__main__":
    check()
