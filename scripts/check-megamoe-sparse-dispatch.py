"""CPU model of Decode sparse dispatch reservations and the unchanged wire protocol.

This checks integer routing, not CUDA memory ordering or numerical GPU output.
Pass the reconstructed production header and candidate header to also verify
that the dispatch boundary, token/SF pull and final cleanup remain unchanged.
"""

import argparse
import itertools
import random
from collections import Counter
from pathlib import Path

SMS, EXPERTS, RANKS, TOPK, DISPATCH_WARPS = 132, 128, 4, 8, 2
WORD = 1 << 32


def reserve(routes, sparse, seed):
    rng = random.Random(seed)
    by_cta = [[] for _ in range(SMS)]
    for token, experts in enumerate(routes):
        sm = (token // (32 // TOPK) // DISPATCH_WARPS) % SMS
        by_cta[sm].extend((expert, token * TOPK + slot) for slot, expert in enumerate(experts) if expert >= 0)
    counts = [Counter(expert for expert, _ in entries) for entries in by_cta]
    counters = [0] * EXPERTS
    offsets = [[0] * EXPERTS for _ in range(SMS)]
    atomics = 0
    # Independent CTA order per expert models arbitrary reservation interleaving.
    for expert in range(EXPERTS):
        order = list(range(SMS))
        rng.shuffle(order)
        for sm in order:
            count = counts[sm][expert]
            if not sparse or count:
                offsets[sm][expert] = counters[expert] % WORD
                counters[expert] += count + (0 if sparse else WORD)
                atomics += 1
    lists = [[None] * (count % WORD) for count in counters]
    for sm, entries in enumerate(by_cta):
        rng.shuffle(entries)
        for expert, source in entries:
            slot = offsets[sm][expert]
            assert lists[expert][slot] is None, "overlapping CTA reservation"
            lists[expert][slot] = source
            offsets[sm][expert] += 1
    assert all(source is not None for sources in lists for source in sources)
    # Publication is after the existing grid sync, including all source writes.
    published = [(SMS * WORD + count % WORD) if sparse else count for count in counters]
    assert all(status // WORD == SMS for status in published)
    return published, lists, atomics


def routing(count, kind, rng):
    if kind == "padding":
        return [[-1] * TOPK for _ in range(count)]
    if kind == "concentrated":
        return [list(range(TOPK)) for _ in range(count)]
    if kind == "duplicate":
        return [[7] * TOPK for _ in range(count)]
    if kind == "balanced":
        return [[(token * TOPK + slot) % EXPERTS for slot in range(TOPK)] for token in range(count)]
    return [[expert if rng.randrange(5) else -1 for expert in rng.sample(range(EXPERTS), TOPK)] for _ in range(count)]


def check_headers(baseline, candidate):
    old, new = baseline.read_text(), candidate.read_text()
    begin = "        // Stake out per-expert SM offsets via global atomic"
    end = "        // Write source token-topk indices to remote ranks"
    read = "                const auto expert_status = *workspace.get_expert_send_count_ptr(i);"
    publish = "                *sym_buffer.map(\n                    workspace.get_expert_recv_count_ptr"
    # Outside the reservation and the SM0 status normalization, bytes must match.
    before, old_tail = old.split(begin, 1)
    new_before, new_tail = new.split(begin, 1)
    assert before == new_before
    old_reserve, old_tail = old_tail.split(end, 1)
    new_reserve, new_tail = new_tail.split(end, 1)
    assert "if constexpr (kSwapABActive)" in new_reserve
    assert "if (count != 0)" in new_reserve
    assert old_reserve.strip().endswith("ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);")
    assert new_reserve.strip().endswith("ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);")
    old_middle, old_tail = old_tail.split(read, 1)
    new_middle, new_tail = new_tail.split("                auto expert_status = *workspace.get_expert_send_count_ptr(i);", 1)
    assert old_middle == new_middle and "comm::grid_sync<" in old_middle
    _, old_tail = old_tail.split(publish, 1)
    normalization, new_tail = new_tail.split(publish, 1)
    assert "(static_cast<uint64_t>(kNumSMs) << 32) | (expert_status & 0xffffffffull)" in normalization
    assert old_tail == new_tail, "pull, FP arithmetic, barriers or cleanup changed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    check_headers(args.baseline, args.candidate)
    rng = random.Random(20261023)
    cases = 0
    for count, kind, seed in itertools.product((0, 1, 3, 4, 7, 8, 9, 127, 128, 144, 145, 257, 1057, 4096), ("padding", "concentrated", "duplicate", "balanced", "random"), range(3)):
        routes = routing(count, kind, rng)
        old = reserve(routes, False, seed)
        new = reserve(routes, True, seed)
        assert old[:2] == new[:2], (count, kind, seed)
        assert old[2] == SMS * EXPERTS
        assert new[2] <= sum(expert >= 0 for experts in routes for expert in experts)
        cases += 1
    # Same receive-count ABI for every mixture of old and sparse local geometry.
    # Reuse the receive counters across changed-input replays and explicit reset.
    recv = [0] * EXPERTS
    for sparse_mask in itertools.product((False, True), repeat=RANKS):
        for sizes in ((0, 0, 0, 0), (0, 128, 145, 7), (144, 1, 129, 0), (128, 128, 128, 128)):
            assert not any(recv)
            expected = Counter()
            for rank in range(RANKS):
                routes = routing(sizes[rank], "random", rng)
                expected.update(expert for experts in routes for expert in experts if expert >= 0)
                published, _, _ = reserve(routes, sparse_mask[rank], rank)
                recv = [left + right for left, right in zip(recv, published)]
                assert all(status // WORD == SMS * (rank + 1) for status in recv)
                assert all((status // WORD == SMS * RANKS) == (rank == RANKS - 1) for status in recv)
            assert [status % WORD for status in recv] == [expected[e] for e in range(EXPERTS)]
            recv[:] = [0] * EXPERTS  # existing cleanup after all readers finish
    print(f"PASS: {cases} reservation cases; 64 mixed/idle-rank publications and changed-input replays; unchanged pull/arithmetic/cleanup")
    print("Decode 128 tokens/rank: 16896 baseline count atomics; candidate at most 1024 (zero for all padding).")


if __name__ == "__main__":
    main()
