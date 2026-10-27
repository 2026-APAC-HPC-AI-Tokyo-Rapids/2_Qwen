"""CPU address/ownership check; does not validate GPU ordering or arithmetic."""
from collections import Counter

HIDDEN_BYTES = 4096 * 2
CHUNKS = 4
CHUNK_BYTES = HIDDEN_BYTES // CHUNKS
VECTOR_BYTES = 16
WARPS = 8
SMS = 132
STRIDE = SMS * WARPS


def check(tokens):
    owners = Counter()
    for sm in range(SMS):
        for warp in range(WARPS):
            for token in range(sm * WARPS + warp, tokens, STRIDE):
                for chunk in range(CHUNKS):
                    for j in range(CHUNK_BYTES // VECTOR_BYTES // 32):
                        for lane in range(32):
                            offset = token * HIDDEN_BYTES + chunk * CHUNK_BYTES
                            address = offset + (j * 32 + lane) * VECTOR_BYTES
                            assert address % VECTOR_BYTES == 0
                            assert offset <= address < offset + CHUNK_BYTES
                            owners[address // VECTOR_BYTES] += 1
    expected = tokens * HIDDEN_BYTES // VECTOR_BYTES
    assert len(owners) == expected
    assert all(count == 1 for count in owners.values())
    assert not owners or (min(owners) == 0 and max(owners) == expected - 1)
    return expected


if __name__ == '__main__':
    cases = (0, 1, 2, 7, 8, 9, 131, 132, 133, 1055, 1056, 1057, 2048, 4096)
    total = sum(check(tokens) for tokens in cases)
    print(f'{len(cases)} token-count cases: {total:,} unique 16-byte output vectors verified')
