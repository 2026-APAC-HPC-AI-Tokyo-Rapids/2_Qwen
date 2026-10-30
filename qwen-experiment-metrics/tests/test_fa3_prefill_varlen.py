"""CPU checks for packed-request (varlen) fixed Prefill FA3 candidates."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]


def added_lines(patch):
    return '\n'.join(line[1:] for line in patch.read_text().splitlines() if line.startswith('+') and not line.startswith('+++'))


@pytest.mark.parametrize('lengths', [(3994,), (3986, 4027), (4096, 3584, 4000), (3584, 3584, 4096, 4096)])
def test_varlen_tile_order_covers_every_tile_once(tmp_path, lengths):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('g++ required')
    block = added_lines(ROOT / 'kernels/fa3-verify/prefill-varlen.patch')
    body = block.split('            if constexpr (FixedVarlen) {', 1)[1].rsplit('\n            }', 1)[0]
    source = r'''
#include <cassert>
#include <set>
#include <tuple>
constexpr int kBlockM = 128;
namespace cute { int ceil_div(int a, int b) { return (a + b - 1) / b; } }
struct Params { int num_head, num_batch, qhead_per_khead, seqlen; const int* cu_seqlens; };
struct Work { int tile_idx, block, bidh, bidb; };
Work varlen(Params const& params, int next_tile_idx) {
''' + body + r'''
}
Work single(Params const& params, int next_tile_idx) {
    int blocks = cute::ceil_div(params.seqlen * params.qhead_per_khead, kBlockM);
    if (next_tile_idx >= blocks * params.num_head) { return {next_tile_idx, 0, 0, params.num_batch}; }
    return {0, blocks - 1 - next_tile_idx / params.num_head, next_tile_idx % params.num_head, 0};
}
int main() {
    int lengths[] = {LENGTHS};
    int batch = sizeof(lengths) / sizeof(int), cu[5] = {0};
    for (int b = 0; b < batch; ++b) cu[b + 1] = cu[b] + lengths[b];
    Params p{4, batch, 16, 0, cu};
    std::set<std::tuple<int, int, int>> seen;
    int expected = 0, previous_batch = 0, t = 0;
    for (int b = 0; b < batch; ++b) expected += (lengths[b] * 16 + kBlockM - 1) / kBlockM * 4;
    for (; t < expected; ++t) {
        Work w = varlen(p, t);
        assert(w.bidb >= previous_batch && w.bidb < batch && w.bidh >= 0 && w.bidh < 4);
        assert(w.block >= 0 && w.block * kBlockM < lengths[w.bidb] * 16);
        assert(seen.insert({w.bidb, w.block, w.bidh}).second);
        previous_batch = w.bidb;
        if (batch == 1) { p.seqlen = lengths[0]; Work s = single(p, t); assert(s.block == w.block && s.bidh == w.bidh && s.bidb == 0); }
    }
    for (int extra = 0; extra < 300; ++extra) assert(varlen(p, t + extra).bidb == batch);
    return 0;
}
'''
    path = tmp_path / 'order.cpp'
    path.write_text(source.replace('LENGTHS', ','.join(map(str, lengths))))
    binary = tmp_path / 'order'
    subprocess.run([compiler, '-O1', '-std=c++17', str(path), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_varlen_bit_reaches_only_the_fixed_scheduler():
    patch = (ROOT / 'kernels/fa3-verify/prefill-varlen.patch').read_text()
    assert 'FixedPrefill, FixedPrefill && (PrefillVariant & 64) != 0 /*FixedVarlen*/>' in patch
    assert 'bool FixedPrefill = false, bool FixedVarlen = false>' in patch
    wrapper = (ROOT / 'kernels/fa3-verify/wrapper.cpp').read_text()
    assert 'struct PrefillCandidate { const char* name; bool byte_image; bool varlen; Launch launch; };' in wrapper
    assert re.search(r'varlen \? maxlen>=3584 && maxlen<=4096 && rows>=batch\*3584 && rows<=batch\*maxlen : batch==1 && rows==maxlen', wrapper)


@pytest.fixture
def guard(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'apply', '--include=python/sglang/srt/layers/attention/prefill_contiguous.py', str(ROOT / 'patches/sglang/prefill-attention-contiguous-kv.patch')], cwd=tmp_path, check=True)
    subprocess.run(['git', 'apply', str(ROOT / 'patches/sglang/prefill-fa3-varlen.patch')], cwd=tmp_path, check=True)
    props = NS(major=9, minor=0, multi_processor_count=132, shared_memory_per_block_optin=227*1024)
    selected = {'varlen': True}
    monkeypatch.setitem(sys.modules, 'torch', NS(float8_e4m3fn='fp8', cuda=NS(get_device_properties=lambda _: props)))
    monkeypatch.setitem(sys.modules, 'sglang.srt.runtime_context', NS(get_server_args=lambda: NS(disaggregation_mode='prefill', enable_lora=False, cuda_graph_config=NS(prefill=NS(backend='disabled')))))
    monkeypatch.setitem(sys.modules, 'sglang.srt.model_executor.forward_batch_info', NS(ForwardMode=NS(EXTEND='extend')))
    monkeypatch.setitem(sys.modules, 'sglang.srt.layers.attention.prefill_fp8', NS(prefill_candidate=lambda: selected))
    import runpy
    ns = runpy.run_path(str(tmp_path / 'python/sglang/srt/layers/attention/prefill_contiguous.py'))
    batch = NS(extend_seq_lens_cpu=[3986], extend_prefix_lens_cpu=[0], forward_mode='extend', encoder_lens=None, mha_return_lse=False)
    layer = NS(tp_q_head_num=64, tp_k_head_num=4, tp_v_head_num=4, head_dim=128, v_head_dim=128, is_cross_attention=False, logit_cap=0, k_scale=None, v_scale=None)
    backend = NS(fa_impl_ver=3, use_mla=False, num_splits=1, attn_cp_size=1, forward_metadata=NS(cu_seqlens_q=NS(numel=lambda: len(batch.extend_seq_lens_cpu) + 1)))
    q = NS(dtype='fp8', shape=(8200, 64, 128), device='cuda')
    return ns, batch, layer, backend, q, selected


@pytest.mark.parametrize('lengths,varlen,expected', [([3986], False, True), ([3986], True, True), ([3986, 4027], False, False), ([3986, 4027], True, True), ([3986, 2048], True, False), ([4096, 4097], True, False), ([3600]*5, True, False)])
def test_packed_requests_need_a_selected_varlen_row(guard, lengths, varlen, expected):
    ns, batch, layer, backend, q, selected = guard
    selected['varlen'] = varlen
    batch.extend_seq_lens_cpu = lengths
    batch.extend_prefix_lens_cpu = [0] * len(lengths)
    q.shape = (max(sum(lengths), 8200), 64, 128)
    assert ns['use_fixed_prefill'](batch, layer, backend, q, q, q, output_supported=True) is expected


def test_packed_requests_keep_the_row_count_and_cu_seqlens_contracts(guard):
    ns, batch, layer, backend, q, _ = guard
    batch.extend_seq_lens_cpu = [3986, 4027]
    batch.extend_prefix_lens_cpu = [0, 0]
    q.shape = (8012, 64, 128)
    assert not ns['use_fixed_prefill'](batch, layer, backend, q, q, q, output_supported=True)
    q.shape = (8013, 64, 128)
    backend.forward_metadata.cu_seqlens_q.numel = lambda: 2
    assert not ns['use_fixed_prefill'](batch, layer, backend, q, q, q, output_supported=True)


def test_varlen_rows_compare_against_n192_per_request():
    rows = [r for r in json.loads((ROOT / 'kernels/fa3-verify/prefill-candidates.json').read_text()) if r.get('varlen')]
    assert [r['name'] for r in rows] == ['n192-varlen', 'n192-direct-o-varlen', 'n192-direct-o-varlen-pipe']
    gate = (ROOT / 'scripts/check-firmus-decode-kernels.py').read_text()
    assert 'packed_control = name == "control" and len(common["lengths"]) > 1' in gate
    assert 'torch.ops.apac_fa3_verify.prefill_fwd_out(common["q"], common["k"], common["v"], common["cu"], out, scales, lse, metadata, maxlen, 128**-.5)' in gate
    assert 'for start, length, base_step in base_steps:' in gate
