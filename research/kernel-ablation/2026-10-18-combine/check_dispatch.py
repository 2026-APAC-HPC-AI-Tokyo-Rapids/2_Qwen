"""CPU model of sparse reservations and post-barrier completion tags.

Checks ownership/count invariants, not CUDA memory ordering or performance.
Run with Python 3; no GPU or third-party dependency is needed.
"""

import random
from collections import Counter


def reservations(routes, order, sparse):
    counts = [Counter() for _ in range(132)]
    for token, experts in enumerate(routes):
        # Two dispatch warps, four token rows per warp, grid-stride loop.
        counts[(token // 8) % 132].update(e for e in experts if e >= 0)
    total = [0] * 128
    offsets = {}
    atomics = 0
    for sm in order:
        for expert in range(128):
            count = counts[sm][expert]
            if not sparse or count:
                atomics += 1
                offsets[sm, expert] = total[expert] & 0xffffffff
                total[expert] += count if sparse else (1 << 32) | count
    # This publication occurs only after all CTAs finish writing indices and
    # cross the unchanged release/acquire grid barrier.
    published = [(132 << 32) | n if sparse else n for n in total]
    slots = [set() for _ in range(128)]
    for (sm, expert), offset in offsets.items():
        owned = set(range(offset, offset + counts[sm][expert]))
        assert not (slots[expert] & owned)
        slots[expert].update(owned)
    for expert, owned in enumerate(slots):
        assert owned == set(range(published[expert] & 0xffffffff))
        assert published[expert] >> 32 == 132
    return published, atomics


def check():
    rng = random.Random(0x1838E3A7)
    for tokens in (0, 1, 7, 8, 9, 16, 64, 116, 128, 129, 1056, 4096, 4224):
        for pattern in ("random", "hot", "masked"):
            order = list(range(132))
            rng.shuffle(order)
            routes = [rng.sample(range(128), 8) if pattern != "hot" else list(range(8)) for _ in range(tokens)]
            if pattern == "masked":
                routes = [[e if rng.randrange(3) else -1 for e in row] for row in routes]
            old, old_ops = reservations(routes, order, False)
            new, new_ops = reservations(routes, order, True)
            assert old == new
            assert new_ops <= old_ops == 132 * 128
            assert new_ops <= 8 * tokens
            # Four ranks retain the exact receive scheduler's completion word.
            assert all((n * 4) >> 32 == 132 * 4 for n in new)
            if tokens == 128 and pattern == "random":
                print(f"128-token random routing: {old_ops} -> {new_ops} counter atomics per rank")
    print("PASS: disjoint reservations, exact expert counts and completion words, empty/masked/hot routes and capacity boundaries")


if __name__ == "__main__":
    check()
