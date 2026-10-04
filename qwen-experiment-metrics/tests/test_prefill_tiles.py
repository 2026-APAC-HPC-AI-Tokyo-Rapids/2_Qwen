"""CPU checks for finite condition dispatch and unchanged routing contracts."""
import ast
import sys
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
PATCH = ROOT / 'patches/sglang/prefill-cutlass-tiles.patch'


def function(name):
    source = '\n'.join(line[1:] for line in PATCH.read_text().splitlines() if line.startswith('+') and not line.startswith('+++'))
    source = source[source.index("def " + name):].split("\n\n\n", 1)[0]
    node = ast.parse(source).body[0]
    namespace = {'torch': SimpleNamespace(bfloat16='bf16')}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PATCH), 'exec'), namespace)  # noqa: S102
    return namespace, namespace[name]


@pytest.mark.parametrize('up,down,result', [('off', 'off', (-1, -1)), ('stock', 'stock', (0, 0)), ('coop128_c1x1_s2', 'coop128_c2x2_s4', (5, 15)), ('off', 'coop128_c1x2_s3', (-1, 2))])
def test_cached_condition_scope(up, down, result):
    _, choose = function('_prefill_tile_modes')
    assert choose(up, down, True, True, 1, False) == result
    for prefill, hopper, ep, mxfp8 in [(False, True, 1, False), (True, False, 1, False), (True, True, 4, False), (True, True, 1, True)]:
        assert choose(up, down, prefill, hopper, ep, mxfp8) == (-1, -1)
    with pytest.raises(ValueError):
        choose('unknown', down, True, True, 1, False)


@pytest.mark.parametrize('mode,rows,shape,dtype,native', [(-1, 65536, (384, 4096, 128), 'bf16', False), (1, 0, (384, 4096, 128), 'bf16', False), (1, 2048, (384, 4096, 128), 'bf16', False), (1, 65536, (384, 4096, 128), 'half', False), (1, 65536, (768, 4096, 128), 'bf16', False), (3, 65536, (384, 4096, 128), 'bf16', True), (4, 65536, (4096, 768, 128), 'bf16', True)])
def test_actual_dispatch_never_imports_or_compiles_when_outside_scope(monkeypatch, mode, rows, shape, dtype, native):
    namespace, dispatch = function('_prefill_tile_mm')
    calls = []
    namespace['fp8_blockwise_scaled_grouped_mm'] = lambda *args: calls.append(('wheel', args))
    monkeypatch.setitem(sys.modules, 'apac_prefill_moe_tiles', SimpleNamespace(grouped_mm=lambda *args: calls.append(('native', args))))
    k, n, e = shape
    args = [SimpleNamespace(dtype=dtype, shape=(rows, n)), *[None]*5, SimpleNamespace(shape=(rows, k)), SimpleNamespace(shape=(e, k, n)), *[None]*10]
    dispatch(mode, *args)
    assert calls == [('native', (*args, mode))] if native else calls == [('wheel', tuple(args))]


def test_route_fixture_offsets_cover_top8_without_duplicates_or_negative_expert_rows():
    for tokens, skew in [(32, False), (4096, True), (8192, False), (8192, True)]:
        counts = [0]*128
        for row in range(tokens):
            ids = [(row*17+lane*13)%128 if skew and row%4 == 0 else (row+lane)%16 if skew else (row*8+lane)%128 for lane in range(8)]
            assert len(set(ids)) == 8
            for expert in ids:
                counts[expert] += 1
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1]+count)
        assert offsets[-1] == tokens*8 and all(x<=y for x,y in pairwise(offsets))


def test_actual_build_patch_chain_checks_the_updated_private_tree(tmp_path):
    import os
    import subprocess

    source = tmp_path / 'source'
    source.mkdir()
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    (source / 'dependent.txt').write_text('stage0\n')
    names = ['sglang_patch', 'draft_head_patch', 'es_patch', 'fused_shuffle_patch', 'tile_patch']
    env = dict(os.environ, source_dir=str(source), GITHUB_WORKSPACE=str(tmp_path))
    for index, name in enumerate(names):
        patch = tmp_path / f'{name}.patch'
        patch.write_text(f'--- a/dependent.txt\n+++ b/dependent.txt\n@@ -1 +1 @@\n-stage{index}\n+stage{index + 1}\n')
        env[name] = patch.name
    script = (ROOT / 'scripts/build-sglang-container.sh').read_text()
    start = script.index('for patch in "${sglang_patch}"')
    chain = script[start:script.index('\ndone', start) + len('\ndone')]
    subprocess.run(['bash', '-euc', chain], env=env, check=True)
    assert (source / 'dependent.txt').read_text() == 'stage5\n'
    dockerfile = (ROOT / 'containers/Dockerfile').read_text()
    for line in dockerfile.splitlines():
        if line.startswith('COPY containers/') and line.split()[1].endswith('.sh'):
            assert os.access(ROOT / line.split()[1], os.X_OK), line


def test_rejected_wide_n_geometry_is_not_instantiated_or_selected():
    _, choose = function('_prefill_tile_modes')
    with pytest.raises(ValueError):
        choose('off', 'n256', True, True, 1, False)
    assert 'WideN' not in (ROOT / 'kernels/prefill/moe_tiles.cu').read_text()
    assert 'down-n256' not in (ROOT / 'scripts/benchmark-prefill-moe-tiles.py').read_text()


def test_finite_grid_preserves_precision_and_bounds_compile_memory():
    import json
    candidates = json.loads((ROOT / 'kernels/prefill/grid-configurations.json').read_text())
    assert len(candidates) == 16
    assert {tuple(c['cluster_mnk']) for c in candidates} == {(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1)}
    assert {c['requested_stages'] for c in candidates} == {'auto', 2, 3, 4}
    assert all(c['tile_mnk'] == [128, 128, 128] for c in candidates)
    header = (ROOT / 'kernels/prefill/moe_grid.cuh').read_text()
    assert 'Sm90BlockwiseScaleConfig<1, 128, 128' in header
    assert 'ResolvedStages == RequestedStages' in header
    assert 'shared storage exceeds SM90 capacity' in header
    assert 'StageCountAutoCarveout<Reserve>' in header
    recipe = (ROOT / 'containers/build-prefill-moe-tiles.py').read_text()
    assert "os.environ['MAX_JOBS'] = '1'" in recipe
    for index in range(4):
        cluster = (ROOT / f'kernels/prefill/moe_grid_cluster_{index}.cu').read_text()
        assert cluster.count('run_tile<Config') == 4
