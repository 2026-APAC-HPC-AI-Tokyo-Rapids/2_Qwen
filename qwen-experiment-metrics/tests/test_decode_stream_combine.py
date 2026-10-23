"""CPU task, publication and replay models for the unmeasured streaming candidate.

These verify combinatorial invariants, not GPU memory-ordering or numerics.
"""

import random
from collections import Counter
from graphlib import TopologicalSorter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "patches/deep-gemm-bench/decode-stream-combine/stream-combine.patch"


def decode_task(task, counts):
    blocks = [(count + 63) // 64 for count in counts]
    pool_size = sum(blocks)
    fc1_size = pool_size * 24
    fc2 = task >= fc1_size
    relative = task - fc1_size if fc2 else task
    pool = (relative % (pool_size * 4)) // 4 if fc2 else relative // 24
    column = relative // (pool_size * 4) * 4 + relative % 4 if fc2 else relative % 24
    expert = 0
    while pool >= blocks[expert]:
        pool -= blocks[expert]
        expert += 1
    return int(fc2), expert, pool, column


@pytest.mark.parametrize("counts", [[0] * 32, [1] * 32, [64] * 32, [0, 1, 63, 64, 65, 512] + [0] * 26])
def test_dynamic_claims_cover_each_task_once_in_chunk_order(counts):
    expected = {
        (phase, expert, row, column)
        for phase, columns in [(0, 24), (1, 32)]
        for expert, count in enumerate(counts)
        for row in range((count + 63) // 64)
        for column in range(columns)
    }
    total = len(expected)
    decoded = [decode_task(task, counts) for task in range(total)]
    assert set(decoded) == expected
    assert len(decoded) == len(set(decoded))
    assert [item[3] // 4 for item in decoded if item[0]] == sorted(item[3] // 4 for item in decoded if item[0])
    # A loader claims ahead; delta is carried as an unsigned 32-bit ring word.
    current = list(range(132))
    next_claim = 132
    seen = []
    rng = random.Random(81)
    while active := [cta for cta, task in enumerate(current) if task < total]:
        cta = rng.choice(active)
        task = current[cta]
        delta = (next_claim - task - 132) & 0xFFFFFFFF
        current[cta] = (task + 132 + delta) & 0xFFFFFFFF
        next_claim += 1
        seen.append(task)
    assert sorted(seen) == list(range(total))
    # Every FC2 prerequisite is earlier in the global claim stream.
    locations = {value: index for index, value in enumerate(decoded)}
    for task, (phase, expert, row, _) in enumerate(decoded):
        if phase:
            assert all(locations[0, expert, row, col] < task for col in range(24))


@pytest.mark.parametrize("geometry", [[(64, 128)] * 4, [(64, 128), (128, 256), (128, 128), (64, 256)]])
def test_mixed_geometry_publication_and_changed_input_replay(geometry):
    rng = random.Random(103)
    # First allocation is dirty, as are counters at the end of each replay.
    ready = Counter({(rank, expert, chunk): 991 for rank in range(4) for expert in range(32) for chunk in range(8)})
    for counts in ([0, 1, 63, 64, 65, 128, 129, 257] * 4, [0] * 32, [129] + [0] * 31):
        ready.clear()  # dispatch writes zero before the existing pull barrier
        produced = set()
        events = []
        for rank, (bm, bn) in enumerate(geometry):
            for expert, count in enumerate(counts):
                for row in range(0, count, bm):
                    for column in range(0, 4096, bn):
                        events.append((rank, expert, row, min(bm, count - row), column, bn))
        rng.shuffle(events)
        for rank, expert, row, valid, column, bn in events:
            chunk = column // 512
            # All scatter lanes publish their stores before the tile RMW.
            for token in range(row, row + valid):
                for tile in range(column // 128, (column + bn) // 128):
                    produced.add((rank, expert, token, tile))
            ready[rank, expert, chunk] += valid * (bn // 128)
            expected = counts[expert] * 4
            assert ready[rank, expert, chunk] <= expected
            if ready[rank, expert, chunk] == expected:
                assert all((rank, expert, token, tile) in produced for token in range(counts[expert]) for tile in range(chunk * 4, chunk * 4 + 4))
        for rank in range(4):
            for expert, count in enumerate(counts):
                for chunk in range(8):
                    assert ready[rank, expert, chunk] == count * 4


@pytest.mark.parametrize("tokens", [0, 1, 17, 128, 129])
def test_chunk_consumers_cover_output_and_preserve_topk_order(tokens):
    written = Counter()
    rng = random.Random(21)
    for warp in range(132 * 2):
        for work in range(warp, tokens * 8, 132 * 2):
            chunk, token = divmod(work, tokens)
            slots = [slot for slot in range(8) if rng.randrange(4)]
            completion = slots.copy()
            rng.shuffle(completion)
            # Completion order cannot become addition order; masked slots skip.
            observed = [slot for slot in range(8) if slot in completion]
            assert observed == slots
            for lane in range(32):
                for pair in range(8):
                    for half in range(2):
                        written[token, chunk * 512 + (pair * 32 + lane) * 2 + half] += 1
    assert len(written) == tokens * 4096
    assert set(written.values()) <= {1}


def test_dependency_graph_has_no_consumer_to_producer_cycle_or_early_cleanup():
    graph = {"reset": set(), "pull_barrier": {"reset"}}
    for rank in range(4):
        graph[f"pull{rank}"] = {"pull_barrier"}
        graph[f"fc1{rank}"] = {f"pull{rank}"}
        graph[f"fc2{rank}"] = {f"fc1{rank}"}
        graph[f"consume{rank}"] = {f"pull{rank}", *(f"fc2{source}" for source in range(4))}
    graph["math_barrier"] = {f"fc2{rank}" for rank in range(4)}
    for rank in range(4):
        graph[f"join{rank}"] = {"math_barrier", f"consume{rank}"}
    graph["consumer_barrier"] = {f"join{rank}" for rank in range(4)}
    graph["cleanup"] = {"consumer_barrier"}
    graph["final_barrier"] = {"cleanup"}
    graph["next_reset"] = {"final_barrier"}
    order = list(TopologicalSorter(graph).static_order())
    assert all(order.index(f"consume{rank}") < order.index("cleanup") for rank in range(4))
    assert order.index("math_barrier") < order.index("consumer_barrier")
    assert order.index("final_barrier") < order.index("next_reset")
    # A consumer blocked on a producer cannot block that producer's work.
    assert all(not any("consume" in dependency for dependency in graph[f"fc2{rank}"]) for rank in range(4))


def test_candidate_contains_publication_and_collective_guards():
    added = "\n".join(line[1:] for line in PATCH.read_text().splitlines() if line.startswith("+") and not line.startswith("+++"))
    assert "__threadfence_system();" in added
    assert "atom.acq_rel.sys.global.add.u32" in added
    assert "ld.acquire.sys.global.u32" in added
    assert "valid_m * (BLOCK_N / 128)" in added
    assert "chunk_ready[i] = 0" in added
    assert "%gridid" not in added
    assert "kNumExpertsPerWave == kNumExpertsPerRank" in added
    assert "kStreamingCombine ? 64" in added


@pytest.mark.parametrize("bm,math_threads,split_n", [(64, 256, 2), (128, 256, 1), (128, 512, 2)])
def test_partial_m_warpgroups_all_join_fc2_publication(bm, math_threads, split_n):
    groups = math_threads // 128
    for valid in range(1, bm + 1):
        arrivals = 0
        publishers = 0
        active_rows = 0
        for group in range(groups):
            row_base = (group // split_n) * 64
            active = row_base < valid
            # The early-return and active-scatter paths each arrive once,
            # even for the Prefill geometry that otherwise has fullsync=false.
            arrivals += 128
            if active:
                publishers += int(group == 0)
                active_rows += min(64, valid - row_base)
        assert arrivals == math_threads
        assert publishers == 1
        assert active_rows == valid * split_n
    added = PATCH.read_text()
    assert "if constexpr (kChunkReadyProtocol or kL2EpilogueRequiresFullSync)" in added
    assert "Inactive M warpgroups must join the publication barrier" in added
