"""CPU contract and actual fixed-source patch chain checks before a GPU build."""
import ast
import importlib.util
import random
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('routing_oracle', ROOT / 'scripts/prefill-routing-oracle.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


@pytest.mark.parametrize('tokens', [0, 1, 31, 32, 63, 64, 65, 4096, 8192])
@pytest.mark.parametrize('pattern', ['balanced', 'skew', 'hot'])
def test_maps_and_metadata(tokens, pattern):
    rng = random.Random(20261005)
    ids = [(i % 128 if pattern == 'balanced' else rng.randrange(16) if pattern == 'skew' else 7) for i in range(tokens * 8)]
    x = oracle.prepare_reference(ids)
    expected = [ids.count(e) for e in range(128)]
    assert [row[0] for row in x['p1']] == expected
    assert [row[0] for row in x['p2']] == expected
    assert x['offsets'][0] == 0 and x['offsets'][-1] == len(ids)
    assert sorted(x['c_map']) == list(range(len(ids)))
    for i, (e, dst) in enumerate(zip(ids, x['c_map'])):
        assert x['offsets'][e] <= dst < x['offsets'][e + 1]
        assert x['a_map'][dst] == i // 8
    for tile, offsets in enumerate(x['tile_offsets']):
        for e in range(128):
            assert offsets[e] == x['offsets'][e] + sum(c[e] for c in x['counts'][:tile])


def test_helper_only_invalid_not_whole_moe_support():
    ids = [-1, 128, 7, 7, 0, -4, 127, 1]
    x = oracle.prepare_reference(ids)
    assert x['offsets'][-1] == 5
    for i in [0, 1, 5]:
        assert x['c_map'][i] == -1
    assert x['a_map'][5:] == [-1] * 3


def test_new_invocation_resets_not_accumulates():
    a = oracle.prepare_reference([7] * 512)
    b = oracle.prepare_reference([2] * 520)
    assert a['p1'][7][0] == 512
    assert b['p1'][7][0] == 0 and b['p1'][2][0] == 520
    assert b['offsets'][-1] == 520


def test_production_empty_boundary_and_guards():
    patch = (ROOT / 'patches/sglang/prefill-tiled-routing.patch').read_text()
    assert patch.index('if m == 0:') < patch.index('prepare_tiled_routing(topk_ids')
    for guard in ['num_experts == 128', 'topk == 8', 'm <= 8192', 'not use_mxfp8', 'moe_ep_size == 1', '"prefill"']:
        assert guard in patch
    assert '.item()' not in patch.replace('counts.item()', '')


def test_cuda_integer_only_and_graph_reset():
    text = (ROOT / 'kernels/prefill/moe_routing.cu').read_text()
    assert 'hist[threadIdx.x] = 0' in text and 'cursor[threadIdx.x] = 0' in text
    assert '__match_any_sync(mask, expert)' in text
    assert 'tile_offsets[blockIdx.x * E + expert] + start + rank' in text
    assert 'MAX_ROWS = 8192 * 8' in text
    assert 'float' not in text and 'cutlass' not in text.lower()
    build = (ROOT / 'containers/build-prefill-moe-routing.py').read_text()
    assert 'moe_tiles.cu' not in build and 'cutlass' not in build.lower()
    ast.parse(build)


def test_actual_four_patch_chain(tmp_path):
    mirror = Path('/home/USER/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src')
    if not mirror.exists():
        pytest.skip('fixed source mirror unavailable')
    revision = 'fc275674a97d78553b22597ee72c8fd48f5420f4'
    patches = ['prefill-cutlass-silu-quant.patch', 'draft-extend-selected-head-rows.patch', 'prefill-fp8-scale-row-shuffle.patch', 'prefill-tiled-routing.patch']
    for name in patches:
        text = (ROOT / 'patches/sglang' / name).read_text()
        for line in text.splitlines():
            if line.startswith('--- a/'):
                relative = line[6:]
                target = tmp_path / relative
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(subprocess.check_output(['git', '-C', str(mirror), 'show', f'{revision}:{relative}']))
        subprocess.run(['git', 'apply', '--check', str(ROOT / 'patches/sglang' / name)], cwd=tmp_path, check=True)
        subprocess.run(['git', 'apply', str(ROOT / 'patches/sglang' / name)], cwd=tmp_path, check=True)
    for path in tmp_path.rglob('*.py'):
        ast.parse(path.read_text())
    patched = (tmp_path / 'python/sglang/srt/layers/moe/cutlass_moe.py').read_text()
    assert 'prepare_tiled_routing(topk_ids' in patched
    assert 'shuffle_fp8_rows_with_scales' in patched
    assert '_prefill_tile_modes' not in patched
