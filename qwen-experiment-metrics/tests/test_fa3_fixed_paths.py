"""CPU integration coverage for canonical candidate dispatch and fixed eligibility."""
import ast
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
TABLE = json.loads((ROOT / 'kernels/fa3-verify/candidates.json').read_text())


def test_dispatch_compiles_unique_main_kernels_and_keeps_split_counts(tmp_path):
    ns = runpy.run_path(str(ROOT / 'kernels/fa3-verify/build.py'))
    tasks = ns['generate_dispatch'](ROOT / 'kernels/fa3-verify', tmp_path)
    assert len(tasks) == 9  # output and split-count variants share their main kernel; two CTAs/SM is its own
    header = (tmp_path / 'candidates.h').read_text()
    assert len(TABLE) == 19 and len({c['name'] for c in TABLE}) == 19
    for row in TABLE:
        assert row['name'] in header
        assert 11264 + 384 * row['block_n'] * row['stages'] <= ns['SM90_OPTIN_BYTES']
        assert row.get('ctas', 1) * (11264 + 384 * row['block_n'] * row['stages'] + 1024) <= ns['SM90_SM_BYTES']
    for name, splits in [('p128split3-s3-fp8', 3), ('p128split3-s4-fp8', 3), ('p128split2-s3-fp8', 2)]:
        assert next(c for c in TABLE if c['name'] == name)['splits'] == splits
        assert next(c for c in TABLE if c['name'] == name)['output'] == 'fp8'


@pytest.fixture
def fixed(tmp_path, monkeypatch):
    subprocess.run(['git', 'apply', '--include=python/sglang/srt/layers/attention/prefill_contiguous.py', str(ROOT / 'patches/sglang/prefill-attention-contiguous-kv.patch')], cwd=tmp_path, check=True)
    props = NS(major=9, minor=0, multi_processor_count=132, shared_memory_per_block_optin=227*1024)
    torch = NS(float8_e4m3fn='fp8', cuda=NS(get_device_properties=lambda _: props))
    args = NS(disaggregation_mode='prefill', enable_lora=False, cuda_graph_config=NS(prefill=NS(backend='disabled')))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'sglang.srt.runtime_context', NS(get_server_args=lambda: args))
    monkeypatch.setitem(sys.modules, 'sglang.srt.model_executor.forward_batch_info', NS(ForwardMode=NS(EXTEND='extend')))
    ns = runpy.run_path(str(tmp_path / 'python/sglang/srt/layers/attention/prefill_contiguous.py'))
    batch = NS(extend_seq_lens_cpu=[3986], extend_prefix_lens_cpu=[0], forward_mode='extend', encoder_lens=None, mha_return_lse=False)
    layer = NS(tp_q_head_num=64, tp_k_head_num=4, tp_v_head_num=4, head_dim=128, v_head_dim=128, is_cross_attention=False, logit_cap=0, k_scale=None, v_scale=None)
    backend = NS(fa_impl_ver=3, use_mla=False, num_splits=1, attn_cp_size=1, forward_metadata=NS(cu_seqlens_q=NS(numel=lambda:len(batch.extend_seq_lens_cpu)+1)))
    q = NS(dtype='fp8', shape=(4100,64,128), device='cuda')
    return ns, batch, layer, backend, q, args, props


@pytest.mark.parametrize('lengths,expected', [([17,13],False), ([2048,2048],False), ([3583],False), ([3584],True), ([3986],True), ([4027],True), ([4096],True), ([4097],False)])
def test_fixed_range_and_multisequence_fallback(fixed, lengths, expected):
    ns, batch, layer, backend, q, _, _ = fixed
    batch.extend_seq_lens_cpu = lengths
    batch.extend_prefix_lens_cpu = [0]*len(lengths)
    assert ns['use_fixed_prefill'](batch,layer,backend,q,q,q,output_supported=True) is expected
    assert ns['use_contiguous_prefill'](batch), 'fallback must remain eligible for the existing contiguous path'


@pytest.mark.parametrize('failure', ['prefix','heads','sm','smem','lse','graph','quant','split','descale','extra','cache','output'])
def test_fixed_guard_rejects_unsupported_contracts(fixed, failure):
    ns, batch, layer, backend, q, args, props = fixed
    kw = {'output_supported':True}
    if failure == 'prefix': batch.extend_prefix_lens_cpu = [1]
    elif failure == 'heads': layer.tp_v_head_num = 8
    elif failure == 'sm': props.multi_processor_count = 114
    elif failure == 'smem': props.shared_memory_per_block_optin = 191487
    elif failure == 'lse': batch.mha_return_lse = True
    elif failure == 'graph': args.cuda_graph_config.prefill.backend = 'full'
    elif failure == 'quant': kw['output_supported'] = False
    elif failure == 'split': backend.num_splits = 2
    elif failure == 'descale': kw['descales'] = (None,object(),None)
    elif failure == 'extra': kw['extras'] = (object(),None,None,None,None)
    elif failure == 'cache': kw['save_kv_cache'] = True
    else: kw['out'] = object()
    assert not ns['use_fixed_prefill'](batch,layer,backend,q,q,q,**kw)


@pytest.mark.parametrize('name,expected_quant', [('stock',94),('control96',94),('p128split3-s3-fp8',0)])
def test_timed_step_quantizes_bf16_paths_once_per_layer(monkeypatch, name, expected_quant):
    tree = ast.parse((ROOT / 'scripts/check-firmus-decode-kernels.py').read_text())
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'verify_step')
    calls = []

    class Tensor:
        shape = (160,64,128)
        device = 'cuda'

        @property
        def T(self):
            return self

        def view(self,*shape):
            return self

    native = NS(output_fp8=lambda n:n.endswith('-fp8'), allocate=lambda *a:object(), prepare=lambda *a:calls.append('prepare'), forward=lambda *a:calls.append('native'))
    monkeypatch.setitem(sys.modules,'sglang.srt.layers.attention',NS(decode_fa3_verify=native))
    monkeypatch.setitem(sys.modules,'sgl_kernel.flash_attn',NS(flash_attn_with_kvcache=lambda **kw:calls.append('stock')))
    torch = NS(empty_like=lambda *a,**kw:Tensor(), empty=lambda *a,**kw:Tensor(), bfloat16='bf16',float32='fp32')
    ns = {'torch':torch,'verify_stock_metadata':lambda _:calls.append('metadata'), 'sglang_per_token_group_quant_fp8':lambda *a,**kw:calls.append('quant')}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'timed-verify-step','exec'),ns)  # noqa: S102 - actual tracked function, GPU calls mocked
    commons = [{'q':Tensor(),'k_cache':Tensor(),'v_cache':Tensor(),'page_table':Tensor(),'cache_seqlens':Tensor(),'cu_seqlens_q':Tensor(),'softmax_scale':.1} for _ in range(4)]
    step = ns['verify_step'](commons,name)
    step()
    assert calls.count('quant') == expected_quant
    assert calls.count('stock') + calls.count('native') == 94
    assert calls.count('metadata') + calls.count('prepare') == 1
