"""Round 38 failure regressions against the pinned image, using CPU fixtures only."""
import ast
import json
import runpy
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_prefill_r37_bundle import (
    overlaid as overlaid,  # noqa: PLC0414 - explicit pytest fixture re-export
)

ROOT = Path(__file__).resolve().parents[2]
GATE = runpy.run_path(str(ROOT/'scripts/select-firmus-prefill-r37.py'))


class Tensor:
    """Just the in-place tensor API used by the image's state generators."""
    def __init__(self, values): self.values = np.asarray(values)
    def __getitem__(self, key): return Tensor(self.values[key])
    def __setitem__(self, key, value): self.values[key] = value
    @property
    def dtype(self): return self.values.dtype
    def __mod__(self, value): return Tensor(self.values % value)
    def __add__(self, value): return Tensor(self.values + value)
    def to(self, dtype): return Tensor(self.values.astype(dtype))
    def copy_(self, other): np.copyto(self.values, other.values); return self
    def zero_(self): self.values.fill(0); return self
    def fill_(self, value): self.values.fill(value); return self
    def neg_(self): self.values *= -1; return self
    def mul_(self, value): self.values *= value; return self
    def view(self, dtype): return Tensor(self.values.view(dtype))


def image_ops(overlaid):
    tree = ast.parse((overlaid/'python/sglang/srt/layers/prefill_r37.py').read_text())
    scope = {'torch':NS(int64='i64',uint8=np.uint8,arange=lambda *args,**kw:Tensor(np.arange(*args,dtype=np.int64)))}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef)],type_ignores=[]),'image/prefill_r37.py','exec'),scope)  # noqa: S102 - pinned image classes, no imports or launches
    return NS(**{name:scope[name] for name in ('Writer','Onepass')})


def onepass_case(cls, rows):
    case = cls.__new__(cls)
    case.rows = rows
    case.saved = Tensor(np.full((rows,4096),.25,dtype=np.float32))
    case.saved_r = Tensor(np.full((rows,4096),.5,dtype=np.float32))
    case.x = Tensor(np.empty_like(case.saved.values))
    case.residual = Tensor(np.empty_like(case.saved_r.values))
    case.out = {mode:[Tensor(np.ones((8192,8),dtype=np.float32)) for _ in range(5)] for mode in ('control','candidate')}
    return case


def test_c_fault_is_image_boundary_row_one_not_router_or_endurance(overlaid):
    ops = image_ops(overlaid)
    with pytest.raises(IndexError,match='index 1 .* size 1'):
        onepass_case(ops.Onepass,1).state('boundary')
    # The actual operation and every public output stay inherited unchanged.
    fixed = GATE['gate_cases'](ops)['C']
    assert fixed.step is ops.Onepass.step
    assert fixed.outputs is ops.Onepass.outputs and fixed.final is ops.Onepass.final
    for rows in (1,2,3,4,5,6,7,63,64,65,7988,7989,8192):
        case = onepass_case(fixed,rows)
        for state in ('initial','boundary','changed','initial'):
            case.state(state)
            assert not case.out['candidate'][1].values.any()
            assert not case.out['candidate'][2].values.any()
        np.testing.assert_array_equal(case.x.values,case.saved.values)
        np.testing.assert_array_equal(case.residual.values,case.saved_r.values)
        case.state('boundary')
        if rows >= 7:
            original = onepass_case(ops.Onepass,rows); original.state('boundary')
            assert case.x.values.tobytes() == original.x.values.tobytes()
            assert case.residual.values.tobytes() == original.residual.values.tobytes()
        assert case.x.values[0].tobytes() == np.zeros(4096,dtype=np.float32).tobytes()
        if rows > 1: assert np.signbit(case.x.values[1]).all()



def test_b_small_boundary_rows_keep_the_production_writer_operands(overlaid):
    ops = image_ops(overlaid); fixed = GATE['gate_cases'](ops)['B']
    assert fixed.step is ops.Writer.step and fixed.args is ops.Writer.args
    assert fixed.outputs is ops.Writer.outputs
    def case(cls, rows):
        value = cls.__new__(cls); value.rows=rows
        value.saved = Tensor(np.full((rows,72,128),.25,dtype=np.float32))
        value.packed = Tensor(np.empty_like(value.saved.values))
        value.pos = Tensor(np.arange(rows,dtype=np.int64)); value.loc = Tensor(np.arange(rows,dtype=np.int64))
        value.out = {mode:[Tensor(np.ones((rows,4),dtype=np.float32))] for mode in ('control','candidate')}
        return value
    with pytest.raises(IndexError): case(ops.Writer,1).state('boundary')
    for rows in (1,2,3,4,5,6,31,32,33):
        value=case(fixed,rows); value.state('boundary')
        assert value.pos.values[0] == 4095 and value.loc.values[0] == 0
        assert not value.out['candidate'][0].values.any()
        if rows >= 6:
            original=case(ops.Writer,rows); original.state('boundary')
            assert value.packed.values.tobytes() == original.packed.values.tobytes()
        value.state('initial')
        np.testing.assert_array_equal(value.packed.values,value.saved.values)

def production_fixed(overlaid, monkeypatch):
    # Run the real host predicates with all non-shape guards set to served PP4.
    mode = object()
    monkeypatch.setitem(sys.modules,'sglang.srt.model_executor.forward_batch_info',NS(ForwardMode=NS(EXTEND=mode)))
    args = NS(disaggregation_mode='prefill',enable_lora=False,cuda_graph_config=NS(prefill=NS(backend='disabled')))
    monkeypatch.setitem(sys.modules,'sglang.srt.runtime_context',NS(get_server_args=lambda:args))
    monkeypatch.setitem(sys.modules,'torch',NS(float8_e4m3fn='fp8',cuda=NS(get_device_properties=lambda d:NS(major=9,minor=0,multi_processor_count=132,shared_memory_per_block_optin=191488))))
    scope = runpy.run_path(str(overlaid/'python/sglang/srt/layers/attention/prefill_contiguous.py'))
    def call(lengths, varlen):
        monkeypatch.setitem(sys.modules,'sglang.srt.layers.attention.prefill_fp8',NS(prefill_candidate=lambda:{'varlen':varlen}))
        batch = NS(extend_seq_lens_cpu=lengths,extend_prefix_lens_cpu=[0]*len(lengths),forward_mode=mode,encoder_lens=None,mha_return_lse=False)
        layer = NS(tp_q_head_num=64,tp_k_head_num=4,tp_v_head_num=4,head_dim=128,v_head_dim=128,is_cross_attention=False,logit_cap=0,k_scale=None,v_scale=None)
        backend = NS(fa_impl_ver=3,use_mla=False,num_splits=1,attn_cp_size=1,forward_metadata=NS(cu_seqlens_q=NS(numel=lambda:len(lengths)+1)))
        q = NS(dtype='fp8',shape=(sum(lengths),64,128),device='cpu-double')
        return scope['use_fixed_prefill'](batch,layer,backend,q,q,q,output_supported=True)
    return call


def test_b_case_domain_and_attention_boundaries_match_production(overlaid,monkeypatch):
    rope = (overlaid/'python/sglang/kernels/jit/csrc/elementwise/rope.cuh').read_text()
    start = rope.index('    if (fp32) {',rope.index('const auto num_tokens ='))
    dispatch = rope[start:rope.index('    if (contiguous_k) {',start)]
    assert 'num_qo_heads == 64 && num_kv_heads == 4' in dispatch
    assert all(f'params.{p}_stride_bytes == 18432' in dispatch for p in ('q','k','v'))
    assert 'num_tokens' not in dispatch[:dispatch.index('const auto prefill')]
    assert 'if (num_tokens == 0) return;' in rope
    assert 'with_dtype<int32_t, int64_t>(id_type)' in rope
    config = json.loads((ROOT/'configs/firmus-prefill-pp4.json').read_text())['prefill']
    assert config['chunked_prefill_size'] == 8192 and config['pp_max_micro_batch_size'] == 2
    lengths = GATE['B_LENGTHS']
    assert all(1 <= len(x) <= 2 and all(n > 0 for n in x) and sum(x) <= 8192 for x in lengths)
    assert {(31,),(32,),(33,),(1,),(8192,),(3994,3994),(3994,3995),(4096,4096)} <= set(lengths)
    assert {(3583,),(3584,),(4096,),(4097,),(3584,3584),(3583,3584),(3899,3900),(3900,3900)} <= set(lengths)
    expected = {f'{"+".join(map(str,x))}/{bits}' for x in lengths for bits in (32,64)}
    assert set(GATE['CASES']['B']) == expected
    real = production_fixed(overlaid,monkeypatch)
    # Exhaust every row count in the configured chunk domain, not just samples.
    for varlen in (False,True):
        for rows in range(1,8193):
            assert GATE['writer_fixed_attention']((rows,),varlen) == real((rows,),varlen)
        for x in (*lengths,(1,8191),(3584,4096),(4096,3584),(4096,4097)):
            assert GATE['writer_fixed_attention'](x,varlen) == real(x,varlen)


def test_b_final_uses_actual_production_attention_and_prepack_routes(overlaid,monkeypatch):
    ops = image_ops(overlaid); cls = GATE['gate_cases'](ops)['B']
    calls = []
    torch = NS(uint8='u8',zeros=lambda shape,**kw:('image',shape),ops=NS(apac_fa3_verify=NS(prefill_prepack=lambda *args:calls.append(('prepack',args)))))
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'sglang.srt.layers.attention.decode_fa3_verify',NS(load=lambda:None))
    base = {'name':'base','byte_image':False,'varlen':True}
    native = {'name':'native','byte_image':True,'native_v_n':96,'varlen':True}
    def attention(*args,**kw): calls.append(('attention',kw)); return ('out','scales')
    def pair(lengths): return native if len(lengths)==2 and all(3900 <= n <= 4096 for n in lengths) else base
    monkeypatch.setitem(sys.modules,'sglang.srt.layers.attention.prefill_fp8',NS(attention=attention,prefill_candidate=lambda:base,pair_candidate=pair))
    for lengths in GATE['B_LENGTHS']:
        calls.clear()
        case = cls.__new__(cls); case.rows=sum(lengths); case.lengths=lengths; case.cu=object(); case.native={}
        case.out={'candidate':[NS(device='cpu-double'),object(),object(),object(),object()]}
        result = case.final('candidate')
        assert len([c for c in calls if c[0]=='attention']) == 1
        assert calls[-1][1]['fixed'] == GATE['writer_fixed_attention'](lengths,True)
        prepacked = pair(lengths) is native
        assert any(c[0]=='prepack' for c in calls) == prepacked
        assert len(result) == (3 if prepacked else 2)
        assert calls[-1][1]['lengths'] == lengths


def test_a_absent_and_candidate_failures_publish_full_tracebacks(overlaid,tmp_path,monkeypatch):
    ops = image_ops(overlaid)
    assert set(GATE['KEYS']) == set(GATE['CASES']) == set(GATE['COMPONENTS']) == set(GATE['gate_cases'](ops)) == {'B','C'}
    shell = (ROOT/'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    block = shell[shell.index('# Round 37 reuses'):shell.index('bash "${ROOT}/source/worker.sh" prefill-0')]
    assert '"A"' not in block and '"B"' in block and '"C"' in block
    run = GATE['run']; scope = run.__globals__
    monkeypatch.setenv('LOCAL_RANK','0'); monkeypatch.setenv('WORLD_SIZE','4')
    monkeypatch.setenv('DG_SM90_PREFILL_DENSE_FIRST_K','1')
    monkeypatch.setitem(sys.modules,'torch',NS(int32='i32',int64='i64',set_num_threads=lambda n:None,manual_seed=lambda n:None,cuda=NS(set_device=lambda n:None,get_device_properties=lambda n:NS(uuid='gpu-0'),get_device_name=lambda n:'NVIDIA H200')))
    package = overlaid/'python/sglang'
    monkeypatch.setitem(sys.modules,'sglang',NS(__file__=str(package/'__init__.py')))
    ops.__file__ = str(package/'srt/layers/prefill_r37.py')
    monkeypatch.setitem(sys.modules,'sglang.srt.layers',NS(prefill_r37=ops))
    dense = tmp_path/'deep_gemm'; header = dense/'include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh'
    header.parent.mkdir(parents=True); header.write_text('synthetic source provenance')
    monkeypatch.setitem(sys.modules,'deep_gemm',NS(__file__=str(dense/'__init__.py')))
    def case_failure(*args): raise ValueError('synthetic case fault')
    def compiler_failure(): raise RuntimeError('synthetic compiler fault')
    futures = [Future(),Future()]; futures[0].set_result(None)
    try: compiler_failure()
    except RuntimeError as error: futures[1].set_exception(error)
    monkeypatch.setitem(scope,'snapshot_weights',lambda p:({'B':None,'C':None},'f'*64))
    monkeypatch.setitem(scope,'prepare_compilers',lambda *args:(NS(shutdown=lambda **kw:None),futures))
    monkeypatch.setitem(scope,'gate_cases',lambda ops:{'B':case_failure,'C':case_failure})
    output=tmp_path/'prefill-0'
    run(NS(output=output,model_path=tmp_path,run_id='42',image_sha256='a'*64,deadline=float('inf')))
    assert not (output/'A-rank-0.json').exists()
    for name,fault in (('B','case_failure'),('C','compiler_failure')):
        record=json.loads((output/f'{name}-rank-0.json').read_text())
        assert record['healthy'] is False and fault in record['traceback']
        assert 'Traceback (most recent call last)' in record['traceback'] and record['error'] in record['traceback']
    import os
    assert os.environ['DG_SM90_PREFILL_DENSE_FIRST_K'] == '1'
