"""CPU ownership/order checks for the Qwen SwapAB combine partition.

This checks disjoint output coverage and the existing two-stage load protocol,
not CUDA memory ordering, device numerics, latency, or an official score.
"""

from collections import Counter


def work_items(tokens, chunks, warps):
    for sm in range(132):
        for warp in range(warps):
            for work in range(sm * warps + warp, tokens * chunks, 132 * warps):
                yield sm, warp, work // chunks, work % chunks


def check():
    # All captured Decode sizes, idle/tail ranks, multi-iteration warps, and
    # the small-Prefill fallback also using the SwapAB specialization.
    counts = list(range(130)) + [132, 133, 264, 1024, 4095, 4096]
    for tokens in counts:
        counts_by_chunk = Counter((token, chunk) for _, _, token, chunk in work_items(tokens, 8, 8))
        assert counts_by_chunk == Counter((token, chunk) for token in range(tokens) for chunk in range(8))
        assert all(count == 1 for count in counts_by_chunk.values())

    # 8 KiB of BF16 output: each 16-byte uint4 has exactly one owner. The
    # original and new loops read the same source vector for every top-k slot.
    vectors = []
    for chunk in range(8):
        for lane in range(32):
            for j in range(2):
                address = chunk * 1024 + (j * 32 + lane) * 16
                vectors.append(address)
                old_j, old_lane = divmod(address // 16, 32)
                assert (old_j * 32 + old_lane) * 16 == address
    assert sorted(vectors) == list(range(0, 8192, 16))

    # Preserve the ascending live-slot order, including all-masked tokens.
    # Simulate consecutive work items so stage/phase reuse crosses token and
    # chunk boundaries. The FP32 fmaf expression and BF16 cast are unchanged.
    stage, phase = 0, 0
    generations = [0, 0]
    for mask in list(range(256)) * 3:
        expected = [slot for slot in range(8) if mask & (1 << slot)]
        loaded = {}

        def load(i, loaded=loaded):
            nonlocal mask
            if not mask:
                return False
            slot = (mask & -mask).bit_length() - 1
            mask &= mask - 1
            assert i not in loaded
            loaded[i] = slot
            return True

        do_reduce = load(stage)
        observed = []
        while do_reduce:
            do_reduce = load(stage ^ 1)
            assert phase == generations[stage] % 2
            observed.append(loaded.pop(stage))
            generations[stage] += 1
            phase ^= stage
            stage ^= 1
        assert observed == expected and not loaded

    old_sms = {sm for sm, _, _, _ in work_items(128, 1, 8)}
    new_sms = {sm for sm, _, _, _ in work_items(128, 8, 8)}
    assert len(old_sms) == 16 and len(new_sms) == 128
    print("PASS: unique output coverage, all 256 route masks, unchanged slot order, two-stage phase reuse; 16 -> 128 active combine CTAs at 128 rows")


if __name__ == "__main__":
    check()
