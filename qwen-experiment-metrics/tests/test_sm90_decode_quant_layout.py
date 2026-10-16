"""CPU ownership/reduction model for the register-resident Decode quantizer.

Checks the two warp reductions and reuse of partial-max storage for inverse
scales. This does not replace the distributed GPU numerical/race check.
"""
import random
import struct
from collections import Counter


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def reduce_columns(values):
    for offset in (4, 8, 16):
        values = [max(values[lane], values[lane ^ offset]) for lane in range(32)]
    return values


def test_all_tail_sizes_preserve_quantizer_inputs_and_storage_ownership():
    rng = random.Random(713)
    for valid in range(65):
        rows = [[f32(rng.uniform(-500, 500)) for _ in range(64)] for _ in range(valid)]
        if valid:
            rows[0] = [0.0] * 64
        if valid > 1:
            rows[1] = [f32(1e-12)] * 64
        chunks = (valid + 7) // 8
        shared = [None] * 512
        written = set()
        for i in range(chunks):
            for warp in range(8):
                for half in (0, 1):
                    values = []
                    for lane in range(32):
                        token = i * 8 + (lane % 4) * 2 + half
                        col = warp * 8 + lane // 4
                        values.append(abs(rows[token][col]) if token < valid else 0.0)
                    maxima = reduce_columns(values)
                    addresses = []
                    for lane in range(4):
                        address = i * 64 + half * 32 + lane * 8 + warp
                        assert address not in written
                        written.add(address)
                        shared[address] = maxima[lane]
                        addresses.append(address)
                    assert max(Counter(a % 32 for a in addresses).values()) == 1
        scales = {}
        for i in range(chunks):
            # Read BOTH halves before reusing the first 64 words. These
            # reductions model the data dependencies of the warp shuffles.
            maxima = []
            for half in (0, 1):
                addresses = [i * 64 + half * 32 + (lane % 4) * 8 + lane // 4 for lane in range(32)]
                assert max(Counter(a % 32 for a in addresses).values()) == 1
                maxima.append(reduce_columns([shared[a] for a in addresses]))
            for lane in range(4):
                for half in (0, 1):
                    token = i * 8 + lane * 2 + half
                    amax = maxima[half][lane]
                    sf = f32(max(amax, f32(1e-10)) * f32(1.0 / 448.0))
                    shared[token] = f32(1.0 / sf)
                    if token < valid:
                        assert token not in scales
                        scales[token] = sf
                        assert amax == max(map(abs, rows[token]))
        output = {}
        for i in range(chunks):
            for warp in range(8):
                for lane in range(32):
                    for half in (0, 1):
                        token = i * 8 + (lane % 4) * 2 + half
                        col = warp * 8 + lane // 4
                        if token < valid:
                            assert (token, col) not in output
                            output[token, col] = f32(rows[token][col] * shared[token])
        assert len(scales) == valid
        assert len(output) == valid * 64
        for token, row in enumerate(rows):
            sf = f32(max(max(map(abs, row)), f32(1e-10)) * f32(1.0 / 448.0))
            assert scales[token] == sf
            for col, value in enumerate(row):
                reference = f32(value * f32(1.0 / sf))
                assert struct.pack("f", output[token, col]) == struct.pack("f", reference)
