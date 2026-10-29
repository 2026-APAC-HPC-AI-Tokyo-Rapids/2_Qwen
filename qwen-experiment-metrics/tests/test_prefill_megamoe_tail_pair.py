"""CPU model of the paired-tail MegaMoE schedule (DG_SM90_PREFILL_TAIL_PAIR).

Every role of a CTA (A loader, B loader, math warps) walks the same item stream through
sm90_fp8_mega_moe_for_each_block_dynamic_l2 and decides skip / pair with the same
predicate. The model replays that stream per role for every 128-expert row distribution
of the archived routing capture plus synthetic and edge distributions, and checks that
the stage-ring uses (stage, phase) agree between roles, that every stage use gets the
producer and consumer arrivals its barriers expect, that every output tile is computed
exactly once, and that the FC1 arrivals per pool block equal what the FC2 A loader waits for.
"""
import array
import ast
import itertools
import random
import re
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPERTS, TOPK, SMS, STAGES = 128, 8, 132, 4
BLOCK_M, PAIR_ROWS = 128, 64
L1_NS, L2_NS, L1_KS, L2_KS = 12, 16, 32, 12
MATH_WARPS, ARRIVAL_PARTS = 8, 4


def pool_blocks(rows):
    """(expert, m block, valid rows) in pool order."""
    return [(e, m, min(BLOCK_M, r - m * BLOCK_M)) for e, r in enumerate(rows) for m in range(-(-int(r) // BLOCK_M))]


def items(blocks, num_ns):
    return [(b, n) for b in range(len(blocks)) for n in range(num_ns)]


def streams(blocks, seed):
    """Per-CTA item streams: FC1 static round robin, FC2 claimed in increasing order by random CTAs."""
    fc1 = items(blocks, L1_NS)
    per_cta = [[("fc1", fc1[i]) for i in range(c, len(fc1), SMS)] for c in range(SMS)]
    rng = random.Random(seed)
    for item in items(blocks, L2_NS):
        per_cta[rng.randrange(SMS)].append(("fc2", item))
    return per_cta


def driver(stream, blocks, pair):
    """sm90_fp8_mega_moe_for_each_block_dynamic_l2: the items each role processes."""
    for phase, (b, n) in stream:
        if pair and n % 2 and blocks[b][2] <= PAIR_ROWS:
            continue
        yield phase, b, n


def role_uses(stream, blocks, pair, role):
    """Per processed item: the ring position (stage, phase) it starts at and, per K block, what the
    role does to each stage it uses (barrier counts are per use, so a K block's pattern suffices)."""
    position, uses = 0, []
    for phase, b, n in driver(stream, blocks, pair):
        ks = L1_KS if phase == "fc1" else L2_KS
        paired = pair and blocks[b][2] <= PAIR_ROWS
        if role == "a":
            pattern = ("a+sfa", "plain") if paired else ("a+sfa",)
        elif role == "b":
            pattern = (f"b{n}", f"b{n + 1}") if paired else (f"b{n}",)
        else:
            pattern = (MATH_WARPS,) * (2 if paired else 1)
        uses.append(((position % STAGES, (position // STAGES) % 2), ks, pattern))
        position += ks * len(pattern)
    uses.append(((position % STAGES, (position // STAGES) % 2), 0, ()))
    return uses


def math_work(stream, blocks, pair):
    """(phase, pool block, N block, row half) per warpgroup, and FC1 arrivals per pool block."""
    work, arrivals = [], {}
    for phase, b, n in driver(stream, blocks, pair):
        valid = blocks[b][2]
        paired = pair and valid <= PAIR_ROWS
        for wg in range(2):
            n_wg, half = (n + wg, 0) if paired else (n, wg)
            if half * 64 >= valid:
                continue  # padding-only warpgroup: no output, no arrival
            work.append((phase, b, n_wg, half))
            if phase == "fc1":
                arrivals[b] = arrivals.get(b, 0) + ARRIVAL_PARTS
    return work, arrivals


def check(rows, seed=0):
    rows = [int(r) for r in rows]
    assert len(rows) == EXPERTS and min(rows) >= 0
    blocks = pool_blocks(rows)
    per_cta = streams(blocks, seed)
    for pair in (False, True):
        work, arrivals = [], {}
        for stream in per_cta:
            a, b, m = (role_uses(stream, blocks, pair, role) for role in "abm")
            # Same items, ring positions and uses per K block in every role, so every stage use
            # (stage, phase) is produced and consumed by the same item and K block.
            assert [(u[0], u[1], len(u[2])) for u in a] == [(u[0], u[1], len(u[2])) for u in b] == [(u[0], u[1], len(u[2])) for u in m]
            for ua, ub, um in zip(a, b, m, strict=True):
                # Per use: two producer arrivals (barrier count 2), B always with bytes, A with
                # bytes on the first use of a K block; eight math-warp releases (count 8).
                assert all(x.startswith("b") for x in ub[2]) and ua[2][:1] in ((), ("a+sfa",)) and all(x == "plain" for x in ua[2][1:])
                assert all(x == MATH_WARPS for x in um[2])
            w, arr = math_work(stream, blocks, pair)
            work += w
            for k, v in arr.items():
                arrivals[k] = arrivals.get(k, 0) + v
        # Every (phase, pool block, N block, row half with valid rows) exactly once.
        expected = [(phase, bi, n, half) for phase, ns in (("fc1", L1_NS), ("fc2", L2_NS)) for bi, (_, _, valid) in enumerate(blocks) for n in range(ns) for half in range(2) if half * 64 < valid]
        assert sorted(work) == sorted(expected)
        # The FC2 A loader waits for N1 x ceil(valid / 64) x 4 arrivals per pool block.
        for bi, (_, _, valid) in enumerate(blocks):
            assert arrivals.get(bi, 0) == L1_NS * -(-valid // 64) * ARRIVAL_PARTS
    return blocks


def archived_distributions():
    """Per-expert rows of the routing capture (stdlib .npy reader: the suite may run without numpy)."""
    raw = zipfile.ZipFile(ROOT / "kernels/probes/megamoe-local-ep1-routing.npz").read("logical_ids.npy")
    header_len = int.from_bytes(raw[8:10], "little")
    header = ast.literal_eval(raw[10:10 + header_len].decode())
    assert raw[:8] == b"\x93NUMPY\x01\x00" and header["descr"] == "<i2" and not header["fortran_order"] and sys.byteorder == "little"
    samples, layers, ranks, tokens, topk = header["shape"]
    ids = array.array("h", raw[10 + header_len:])
    assert len(ids) == samples * layers * ranks * tokens * topk

    def rank_ids(s, layer, r):
        base = (((s * layers) + layer) * ranks + r) * tokens * topk
        return ids[base:base + tokens * topk]

    out = []
    for s, layer in itertools.product(range(samples), range(layers)):
        # One-request (one rank's tokens) and two-request (PP4 microbatch) batches.
        for group in ((0,), (1,), (2,), (3,), (0, 1), (2, 3)):
            rows = [0] * EXPERTS
            for r in group:
                for expert in rank_ids(s, layer, r):
                    if expert >= 0:
                        rows[expert] += 1
            out.append(rows)
    return out


def synthetic(kind, tokens=8064, seed=0):
    """The selection's routings (select-firmus-prefill-megamoe.py), with Python's generator."""
    rng = random.Random(seed)
    rows = [0] * EXPERTS
    edge_rows = (1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 66, 127, 128, 129, 191, 192, 193, 255, 256, 1, 63, 64, 65, 1, 2, 3, 4)
    replaced = list(itertools.chain.from_iterable([e] * n for e, n in enumerate(edge_rows, start=16)))
    for t in range(tokens):
        scores = [rng.random() for _ in range(EXPERTS)]
        if kind == "hot":
            scores[0] += 1.0
        if kind == "edges":
            scores[:48] = [-1.0] * 48
        top = sorted(range(EXPERTS), key=lambda e: -scores[e])[:TOPK]
        if kind == "edges" and t < len(replaced):
            top[-1] = replaced[t]
        for e in top:
            rows[e] += 1
    return rows


def test_archived_routings():
    distributions = archived_distributions()
    assert len(distributions) == 36
    tails = 0
    for i, rows in enumerate(distributions):
        blocks = check(rows, seed=i)
        tails += sum(valid <= PAIR_ROWS for _, _, valid in blocks)
    assert tails > 0


@pytest.mark.parametrize("kind", ["balanced", "hot", "edges"])
def test_selection_routings(kind):
    rows = synthetic(kind)
    if kind == "edges":
        assert rows[:16] == [0] * 16 and rows[16] == 1 and rows[16 + 13] == 64 and rows[16 + 14] == 65
    check(rows)


@pytest.mark.parametrize("rows", [
    [0] * 127 + [64512],                  # one expert, no tail
    [1] * 128,                            # every expert one row: all items paired
    [64] * 128,                           # every expert exactly one half tile
    [65] * 128,                           # one row past a half tile: none paired
    [0, 1, 63, 64, 65, 127, 128, 129] * 16,
    [0] * 64 + [129] * 64,                # odd block counts per expert, empty experts
])
def test_edge_distributions(rows):
    check(rows, seed=len(rows))


def test_the_header_uses_one_predicate_in_every_role():
    header = (ROOT / "patches/deep-gemm-bench/prefill-tail-pair/sm90-prefill-tail-pair.patch").read_text()
    added = "\n".join(line[1:] for line in header.splitlines() if line.startswith("+") and not line.startswith("+++"))
    # Driver (FC1 and FC2 skip), A loader, B loader, math warps.
    assert added.count("sm90_fp8_mega_moe_tail_pair<kPairRows>(scheduler.template get_valid_m<false>())") == 2
    assert added.count("sm90_fp8_mega_moe_tail_pair<WG_BLOCK_M>(scheduler.template get_valid_m<false>())") == 2
    assert added.count("sm90_fp8_mega_moe_tail_pair<WG_BLOCK_M>(valid_m)") == 1
    assert added.count("kNumL2BlockKs, kPrefillTailPair, WG_BLOCK_M>(") == 3
    assert "(n_block_idx & 1u) and" in added and "((info.x >> 16) & 1u) and" in added
    # Two ring advances per K block in each role's paired loop.
    loops = re.findall(r"advance_pipeline\(slot\);", added)
    assert len(loops) == 2 + 1 + 2
    assert added.count("full_barriers[stage_idx]->arrive();") == 1 and added.count("arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE)") == 1
    assert added.count("empty_barriers[stage_idx]->arrive();") == 2
    # Warpgroup g of a paired item: rows 0..63 and N block n + g.
    assert "const uint32_t n_block_idx = item_n_block_idx + (pair_tile ? epilogue_wg_idx : 0u);" in added
    assert "(kIndependentWG or pair_tile) ? 0 : epilogue_wg_idx / kWarpgroupSplitN" in added
