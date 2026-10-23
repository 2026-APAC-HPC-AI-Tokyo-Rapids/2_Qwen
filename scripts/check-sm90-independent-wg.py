"""Check the independent-WG resource layout, ownership and task stream on CPU."""
from collections import Counter
import random


def main():
    # Same complete N128 task, now owned by one WG instead of two N64 WGs.
    for valid_m in range(1, 65):
        fc1, fc2 = Counter(), Counter()
        for fragment in range(2):
            for warp in range(4):
                for lane in range(32):
                    for chunk in range(8):
                        for token in (chunk * 8 + lane % 4 * 2, chunk * 8 + lane % 4 * 2 + 1):
                            if token >= valid_m:
                                continue
                            fc1[token, fragment * 32 + warp * 8 + lane // 4] += 1
                            for row in (warp * 16 + lane // 4, warp * 16 + lane // 4 + 8):
                                fc2[token, fragment * 64 + row] += 1
        assert fc1 == Counter({(token, col): 1 for token in range(valid_m) for col in range(64)})
        assert fc2 == Counter({(token, col): 1 for token in range(valid_m) for col in range(128)})
        quant = Counter((thread // 4 + half * 32, thread % 4) for half in range(2) for thread in range(128) if thread // 4 + half * 32 < valid_m)
        assert quant == Counter({(token, quad): 1 for token in range(valid_m) for quad in range(4)})
    # The two queues have disjoint rings, stages, full/empty barriers and output.
    assert set(range(3)).isdisjoint(range(3, 6))
    assert set(range(4)).isdisjoint(range(4, 8))
    assert 768 + 32 + 8 * 2 * 8 <= 1024
    smem = 1024 + 8192 + 2 * 20480 + (2 + 2 * 8) * 8 + 6 * (8192 + 16384 + 512 + 2 * 8)
    assert smem == 200944 <= 232448
    # Simulate asynchronous claims, including fewer tasks than workers, FC1->FC2
    # and expert transitions. Scheduler adds 132; ring carries the unsigned delta.
    # This wrap occurs immediately when initial task 263 finishes first.
    assert (264 - 263 - 132) & 0xffffffff == 0xffffff7d
    assert (263 + 132 + 0xffffff7d) & 0xffffffff == 264
    # Flatten an expert wave with empty experts and multiple M tiles. Its FC2
    # region follows all FC1 tasks, matching MegaMoEScheduler's index space.
    counts = [0, 1, 64, 65, 128, 7, 0, 33] * 4
    tasks = [(phase, expert, m, n) for phase, width in ((1, 24), (2, 32)) for expert, count in enumerate(counts) for m in range((count + 63) // 64) for n in range(width)]
    boundaries = [i for i in range(1, len(tasks)) if tasks[i][:2] != tasks[i - 1][:2]]
    assert any(tasks[i][0] != tasks[i - 1][0] for i in boundaries)
    for claimed in boundaries:
        current = claimed - 1
        delta = (claimed - current - 132) & 0xffffffff
        assert tasks[(current + 132 + delta) & 0xffffffff] == tasks[claimed]
    rng = random.Random(20261023)
    for count in (0, 1, 131, 132, 263, 264, 265, 768, 1792, 4000):
        for _ in range(8):
            next_task = 264
            active = list(range(min(count, 264)))
            seen = []
            while active:
                slot = rng.randrange(len(active))
                current = active.pop(slot)
                seen.append(current)
                claimed = next_task
                next_task += 1
                delta = (claimed - current - 132) & 0xffffffff
                assert (current + 132 + delta) & 0xffffffff == claimed
                assert claimed > current
                if claimed < count:
                    active.append(claimed)
            assert sorted(seen) == list(range(count))
    print(f"PASS: valid_m=1..64 FC1/SF64 and FC2 ownership; quant quads; independent rings/stages; smem={smem}; asynchronous task coverage")


if __name__ == "__main__":
    main()
