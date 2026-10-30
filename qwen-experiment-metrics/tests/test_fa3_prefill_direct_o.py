"""CPU checks for the register-direct FP8 output of the fixed N192 Prefill FA3."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'kernels/fa3-verify'


def vcolmajor_source_columns():
    """Column pair held by (lane, block) before permute_output_fp8_Vcolmajor.

    Mirrors hopper/utils.h at the pinned sgl-attn commit: lanes 0 and 3 keep the
    upper/lower order, upper comes from upper_map[q] and lower from upper_map[q]^2.
    After the shuffle, lane q block s holds the standard columns 8s+2q and 8s+2q+1.
    """
    upper_map = (0, 2, 3, 1)
    source = {}
    for i in range(8):
        sent_upper, sent_lower = {}, {}
        for q in range(4):
            keep = q in (0, 3)
            sent_upper[q] = (q, 2 * i) if keep else (q, 2 * i + 1)
            sent_lower[q] = (q, 2 * i + 1) if keep else (q, 2 * i)
        for q in range(4):
            keep = q in (0, 3)
            upper, lower = sent_upper[upper_map[q]], sent_lower[upper_map[q] ^ 2]
            for block, origin in ((2 * i, upper if keep else lower), (2 * i + 1, lower if keep else upper)):
                source[origin] = 8 * block + 2 * q
    return source


def test_direct_output_indexing_matches_the_unpermuted_columns():
    patch = (ROOT / 'prefill-direct-o.patch').read_text()
    added = '\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    reads = re.findall(r'x\[(\d)\] = values\(make_coord\((\d), r, 2 \* i( \+ 1)?\), _0\{\}\) \* quant_scale;', added)
    assert [int(k) for k, _, _ in reads] == [0, 1, 2, 3]
    assert 'int const col = 2 * get<1>(coords(_0{}, _0{}));' in added
    assert '*reinterpret_cast<uint32_t*>(dst + 16 * i) = ' in added
    source = vcolmajor_source_columns()
    assert len(source) == 64 and sorted(source.values()) == sorted(8 * s + 2 * q for s in range(16) for q in range(4))
    for q in range(4):
        col = 2 * (2 * q)  # col of fragment (0, r, 0) in the standard C layout is 2q
        for i in range(8):
            for k, c, odd in reads:
                assert source[(q, 2 * i + bool(odd))] + int(c) == col + 16 * i + int(k)


def test_direct_output_candidate_differs_from_n192_only_by_the_output_bit():
    rows = {r['name']: r for r in json.loads((ROOT / 'prefill-candidates.json').read_text())}
    row = rows['n192-direct-o']
    base = rows[row['bitwise_reference']]
    assert row['variant'] == base['variant'] | 32 and not base['variant'] & 32
    assert (row['block_n'], row['byte_image'], row['bitwise']) == (base['block_n'], base['byte_image'], base['bitwise'])
    assert {name for name, r in rows.items() if 'bitwise_reference' in r} == {'n192-direct-o', 'n192-varlen', 'n192-direct-o-varlen', 'n192-direct-o-varlen-pipe', 'n192-direct-o-varlen-pipe-diag'}
    for name, bit in (('n192-varlen', 64), ('n192-direct-o-varlen', 96)):
        assert rows[name]['varlen'] and rows[name]['variant'] == bit and rows[name]['bitwise_reference'] == 'n192'
        assert (rows[name]['block_n'], rows[name]['byte_image']) == (192, False)


def test_direct_output_patch_is_last_in_the_build_chain():
    source = (ROOT / 'build.py').read_text()
    assert source.index('"decode-merge.patch"') < source.index('"prefill-direct-o.patch"') < source.index('"prefill-varlen.patch"') < source.index('cuda = Path("/usr/local/cuda")')
    patch = (ROOT / 'prefill-direct-o.patch').read_text()
    assert 'PrefillDirectO = (PrefillVariant & 32) != 0;' in patch
    assert 'if constexpr (CollectiveMainloop::PrefillDirectO) {' in patch
