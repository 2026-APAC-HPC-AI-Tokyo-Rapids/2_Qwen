"""CPU proofs of the stream-K ownership and WGMMA store mapping."""
import pytest


@pytest.mark.parametrize("n,k", [(9216, 4096), (4096, 8192)])
def test_stream_k_every_unit_once_and_fixup_reads_only_written_slots(n, k):
    blocks = k // 128
    total = n // 128 * blocks
    owners = {}
    written = {}
    lengths = []
    for cta in range(132):
        begin, end = total * cta // 132, total * (cta + 1) // 132
        lengths.append(end - begin)
        tiles = set()
        for unit in range(begin, end):
            assert unit not in owners
            owners[unit] = cta
            tile = unit // blocks
            slot = 2 * cta + tile - begin // blocks
            assert 2 * cta <= slot < 2 * cta + 2
            written.setdefault((tile, slot), []).append(unit % blocks)
            tiles.add(tile)
        assert 1 <= len(tiles) <= 2
    assert set(owners) == set(range(total))
    assert max(lengths) - min(lengths) == 1
    for tile in range(n // 128):
        first = ((tile * blocks + 1) * 132 + total - 1) // total - 1
        last = (((tile + 1) * blocks) * 132 + total - 1) // total - 1
        ks = []
        for cta in range(first, last + 1):
            slot = 2 * cta + tile - (total * cta // 132) // blocks
            ks.extend(written[(tile, slot)])
        assert ks == list(range(blocks))


@pytest.mark.parametrize("bn", [128, 160, 176])
def test_two_warpgroup_accumulators_cover_transposed_partial_exactly(bn):
    stores = set()
    for thread in range(256):
        lane, warp, wg = thread % 32, thread % 128 // 32, thread // 128
        r0 = wg * 64 + warp * 16 + lane // 4
        for i in range(bn // 2):
            token = i // 4 * 8 + lane % 4 * 2 + i % 2
            weight = r0 + i % 4 // 2 * 8
            assert (token, weight) not in stores
            stores.add((token, weight))
    assert stores == {(t, w) for t in range(bn) for w in range(128)}
    # Two complete fragments per CTA; fits H200's 50 MiB L2 at the cap.
    assert 264 * bn * 128 * 4 < 50 * 1024**2
    # Swizzle atom and stage bases meet the descriptor's 1024-byte alignment.
    assert (128 + bn) * 128 % 1024 == 0
