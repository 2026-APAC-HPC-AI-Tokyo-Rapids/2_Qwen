"""Retire the settled gate without changing the measured kernel or its preparation."""
import gzip
import hashlib
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PINS = {'DG_SM90_PREFILL_FC1_FIRST_K': [41767886175], 'DG_SM90_PREFILL_DENSE_FIRST_K': [41811427804]}


@pytest.mark.parametrize('key', PINS)
def test_pin_is_recorded_in_both_manifests_and_only_prefill_environment(tmp_path, key):
    config = json.loads((ROOT / 'configs/firmus-prefill-pp4.json').read_text())
    for role in ('source', 'results'):
        (tmp_path / role).mkdir()
    runpy.run_path(str(ROOT / 'scripts/prepare-firmus-pp4-settings.py'))['prepare'](config, ROOT, tmp_path)
    manifest = json.loads((tmp_path / 'source/pp4-kernel-settings.json').read_text())
    assert manifest == json.loads((tmp_path / 'results/pp4-kernel-settings.json').read_text())
    assert manifest['environment'][key] == manifest['pp4_selected'][key] == '1'
    assert manifest['pp4_selected_runs'][key] == PINS[key]
    assert (tmp_path / 'source/prefill-kernels.env').read_text().splitlines().count(key + '=1') == 1
    assert key not in (tmp_path / 'source/decode-kernels.env').read_text()
    assert 'megamoe_fc1_first_k_gate' not in manifest
    assert 'megamoe_fc1_first_k_gate' not in config['prefill']


def test_retired_gate_diagnostics_and_staging_are_removed_but_dispatch_stays():
    shell = (ROOT / 'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    provision = (ROOT / 'scripts/provision-firmus-framework.sh').read_text()
    selector = (ROOT / 'scripts/select-firmus-prefill-megamoe.py').read_text()
    assert shell.index('# Decode greedy gate end.') < shell.index('# FC1 first-K is pinned') < shell.index('bash "${ROOT}/source/worker.sh" prefill-0')
    for text in (shell, provision, selector):
        assert '--fc1-first-k' not in text and 'firmus_prefill_fc1_first_k' not in text
        assert 'main returning' not in text and 'faulthandler' not in text
        assert 'prefill-kernels.r32.env' not in text
    assert not (ROOT / 'scripts/firmus_prefill_fc1_first_k.py').exists()
    assert 'prefill-r32-dispatch-${worker}.txt' in shell
    assert '(prewarm done|using candidate|using control)' in shell
    assert 'no repeated kernel sweep' in shell


def test_dense_flag_requires_the_compiled_source_generator_contract():
    fixture = json.loads(gzip.decompress((ROOT / 'qwen-experiment-metrics/tests/fixtures/prefill-r37-host-contract.json.gz').read_bytes()))
    for name, text in fixture['files'].items():
        assert hashlib.sha256(text.encode()).hexdigest() == fixture['sha256'][name]
    host = fixture['files']['csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp']
    start = host.index('    static std::string generate_impl(')
    generate = host[start:host.index('    static void launch_impl(', start)]
    assert '#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>' in generate
    assert 'get_env' not in generate and '#define' not in generate
    assert 'get_compiled_dim' in generate and 'get_default_epilogue_type' in generate
    api = fixture['files']['csrc/apis/gemm.hpp']
    start = api.index('    m.def("fp8_fp4_gemm_nt"')
    binding = api[start:api.index('    m.def(', start + 10)]
    assert 'compiled_dims' in binding and 'epilogue_type' not in binding
    compiler = fixture['files']['csrc/jit/compiler.hpp']
    assert 'name, signature, flags, code' in compiler
    parser = fixture['files']['csrc/jit/include_parser.hpp']
    assert 'if (cache.count(path) > 0)' in parser
    stage = (ROOT / 'scripts/stage-qwen-overlays.sh').read_text()
    assert 'stage_overlay deep-gemm deep_gemm_overlay_patches opt/sglang/venv/lib/python3.12/site-packages/deep_gemm deep_gemm' in stage
    assert 'csrc/jit_kernels' not in stage
