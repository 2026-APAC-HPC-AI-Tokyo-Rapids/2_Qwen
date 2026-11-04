"""History-independent checks for the pinned production A/B/C kernels and prewarm."""
import ast
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KEYS = {'A': 'DG_SM90_PREFILL_DENSE_FIRST_K', 'B': 'SGLANG_PREFILL_QKNORM_FIXED', 'C': 'SGLANG_PREFILL_ONEPASS_DIV448'}


@pytest.fixture
def overlaid(tmp_path):
    fixture = json.loads(gzip.decompress((ROOT/'qwen-experiment-metrics/tests/fixtures/prefill-r32-image-source.json.gz').read_bytes()))
    for name,text in fixture['files'].items():
        assert hashlib.sha256(text.encode()).hexdigest() == fixture['sha256'][name]
        p = tmp_path/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text)
    config = json.loads((ROOT/'configs/firmus-prefill-pp4.json').read_text())
    for patch in config['sglang_overlay_patches']:
        subprocess.run(['git','apply','--check',str(ROOT/patch)],cwd=tmp_path,check=True,capture_output=True)
        subprocess.run(['git','apply',str(ROOT/patch)],cwd=tmp_path,check=True,capture_output=True)
    return tmp_path



def test_original_writer_and_onepass_jit_bodies_stay_byte_identical(overlaid):
    fixture = json.loads(gzip.decompress((ROOT/'qwen-experiment-metrics/tests/fixtures/prefill-r37-production-control.json.gz').read_bytes()))
    rope = (overlaid/'python/sglang/kernels/jit/csrc/elementwise/rope.cuh').read_text()
    marker = 'template<bool kUsePDL, typename IdType>\n__global__ __launch_bounds__(256, 5) void qknorm_rope_prefill_fp32_kernel('
    start = rope.index(marker); end = rope.index('\ntemplate<bool kUsePDL, typename IdType>',start+len(marker))
    assert rope[start:end].rstrip() == fixture['writer'].rstrip()
    source = (overlaid/'python/sglang/kernels/ops/moe/prefill_moe_input.py').read_text()
    fn = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == '_onepass_moe_input')
    assert ast.get_source_segment(source,fn) == fixture['onepass']
    assert 'tl.div_rn(tl.full((BLOCK_M,), 448.0, tl.float32), absmax)' in source
    for file in overlaid.rglob('*.py'): ast.parse(file.read_text())



def test_real_shape_prewarm_order_and_actual_dispatches(overlaid):
    runner = (overlaid/'python/sglang/srt/model_executor/model_runner.py').read_text()
    assert runner.index('self.configure_kv_cache_dtype()') < runner.index('prewarm_fc1_first_k(self.model)') < runner.index('prewarm(self.model)')
    tree = ast.parse((overlaid/'python/sglang/srt/layers/prefill_r37.py').read_text())
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'prewarm')
    text = ast.unparse(fn)
    assert 'factory(7988, weights)' in text and 'torch.random.fork_rng' in text
    assert text.index("case.step('candidate')") < text.index('torch.cuda.synchronize()') < text.index('prewarm done') < text.index('_prewarmed = True')
    assert 'a.qkv_proj.weight' in text and 'a.o_proj.weight' in text and 'layer.mlp.gate.weight' in text
    model = (overlaid/'python/sglang/srt/models/qwen3_moe.py').read_text()
    assert model.count('dispatch("B", q.shape[0])') == 2
    assert model.index('onepass_moe_input(hidden_states') < model.index('dispatch("C", num_tokens)')
    linear = (overlaid/'python/sglang/srt/layers/attention/prefill_fp8.py').read_text()
    assert linear.index('deep_gemm.fp8_gemm_nt') < linear.index('dispatch("A", a.shape[0])')



def generate_function(source):
    start = source.index('    static std::string generate_impl(')
    return source[start:source.index('    static void launch_impl(',start)]



def test_actual_host_generator_off_text_and_on_macro(tmp_path):
    if not shutil.which('g++'): pytest.skip('g++ unavailable')
    fixture = json.loads(gzip.decompress((ROOT/'qwen-experiment-metrics/tests/fixtures/prefill-r37-host-contract.json.gz').read_bytes()))
    name = 'csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp'
    control = fixture['files'][name]
    p = tmp_path/name; p.parent.mkdir(parents=True); p.write_text(control)
    subprocess.run(['git','apply',str(ROOT/'patches/deep-gemm-host/sm90-prefill-dense-first-k-r37.patch')],cwd=tmp_path,check=True,capture_output=True)
    prefix = r'''
#include <cassert>
#include <sstream>
#include <string>
bool enabled = false;
template<class T> T get_env(const char*, T) { return enabled; }
namespace fmt { template<class... T> std::string format(const char* s, T... args) { std::ostringstream o; o<<s; ((o<<args<<";"),...); return o.str(); } }
std::string to_string(int x) { return std::to_string(x); }
int get_compiled_dim(int n, char, const std::string&) { return n; }
int get_default_epilogue_type(int n) { return n; }
struct Desc { int m,n,k,num_groups=1,gemm_type=0,cd_dtype=1; std::string compiled_dims="nk"; };
struct Layout { int block_m=256,block_n=128,block_k=128,cluster_n=2; int get_cluster_size() const { return cluster_n; } };
struct Storage { int swizzle_a_mode=128,swizzle_b_mode=128,swizzle_cd_mode=128; };
struct Pipeline { int num_stages=3; };
struct Launch { int num_tma_threads=128,num_math_threads=256,num_sms=132; };
struct Config { Layout layout; Storage storage_config; Pipeline pipeline_config; Launch launch_config; };
struct Args { Desc gemm_desc; Config gemm_config; int major_sfb=0,epilogue_type=0; };
'''
    code = prefix+'struct Control {\n'+generate_function(control)+'};\nstruct Candidate {\n'+generate_function(p.read_text())+'};\n'+r'''
int main() { for (int m: {1,192,2048,7988,7989,8192}) for (int n: {4096,9216,12288}) for (int k:{4096,8192}) {
 Args a; a.gemm_desc.m=m; a.gemm_desc.n=n; a.gemm_desc.k=k;
 enabled=false; auto control=Control::generate_impl(a); assert(Candidate::generate_impl(a)==control);
 enabled=true; assert(Candidate::generate_impl(a)=="#define DG_SM90_PREFILL_DENSE_FIRST_K 1\n"+control);
} }
'''
    cpp = tmp_path/'host.cpp'; cpp.write_text(code)
    subprocess.run(['g++','-std=c++17','-O2',str(cpp),'-o',str(tmp_path/'host')],check=True,capture_output=True)
    subprocess.run([str(tmp_path/'host')],check=True)



def test_a_constexpr_guard_and_off_control_body(tmp_path):
    fixture = json.loads(gzip.decompress((ROOT/'qwen-experiment-metrics/tests/fixtures/prefill-r37-production-control.json.gz').read_bytes()))
    name = 'deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh'
    p = tmp_path/name; p.parent.mkdir(parents=True); p.write_text(fixture['dense'])
    subprocess.run(['git','apply',str(ROOT/'patches/deep-gemm-overlay/sm90-prefill-dense-first-k-r37.patch')],cwd=tmp_path,check=True,capture_output=True)
    source = p.read_text()
    begin = fixture['dense'].index('                    #pragma unroll 8\n                    for (uint32_t k_block_idx = 0;')
    end = fixture['dense'].index('\n                });',begin)
    assert fixture['dense'][begin:end] in source
    assert 'SHAPE_M == 0' in source and 'SHAPE_N == 9216 && SHAPE_K == 4096' in source and 'SHAPE_N == 4096 && SHAPE_K == 8192' in source
    assert 'kNumStages == 3' in source and 'kNumMathThreads == 256' in source and 'if constexpr (first_k)' in source
    assert '(first && u == 0) ? 0.0f' in source and source.count('__fmaf_rn(') == 4



@pytest.mark.parametrize('selected',((),('A',),('B',),('C',),('A','B','C')))
def test_preparation_executes_each_accepted_shape_once_before_marking_ready(overlaid,monkeypatch,selected,caplog):
    import logging
    import os
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace as NS

    calls = []
    class Layer:
        def __init__(self):
            weight=NS(weight=object(),weight_scale_inv=object(),variance_epsilon=1e-6)
            self.self_attn=NS(qkv_proj=weight,o_proj=weight,q_norm=weight,k_norm=weight,rotary_emb=NS(cos_sin_cache=object()))
            self.post_attention_layernorm=weight; self.mlp=NS(gate=weight)
    class Case:
        def __init__(self,rows,weights): calls.append(('allocate',rows))
        def step(self,mode): calls.append(('execute',mode))
    monkeypatch.setitem(sys.modules,'sglang.srt.models.qwen3_moe',NS(Qwen3MoeDecoderLayer=Layer))
    production_keys = {'A':'DG_SM90_PREFILL_DENSE_FIRST_K', **KEYS}
    for c,key in production_keys.items(): monkeypatch.setenv(key,str(int(c in selected)))
    source = (overlaid/'python/sglang/srt/layers/prefill_r37.py').read_text()
    fn = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='prewarm')
    namespace={'os':os,'torch':NS(inference_mode=nullcontext,random=NS(fork_rng=lambda **kw:nullcontext()),cuda=NS(current_device=lambda:0,synchronize=lambda:calls.append(('sync',)))),'Dense':Case,'Writer':Case,'Onepass':Case,'KEYS':production_keys,'_prewarmed':False,'logging':logging,'__name__':'r37-prewarm-test'}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'r37-prewarm','exec'),namespace)  # noqa: S102 - actual overlaid preparation with CPU doubles
    model=NS(modules=lambda:[Layer()])
    namespace['prewarm'](model); namespace['prewarm'](model)
    assert calls == [('allocate',7988),('execute','candidate'),('sync',)] * len(selected)
    assert caplog.text.count('prewarm done') == len(selected)



def test_archived_production_off_cubins_and_guard_encodings_are_identical():
    import re
    import tarfile

    record = json.loads((ROOT/'research/runtime-performance/prefill-small-bundle-r37-production-cpu-validation-20261104.json').read_text())
    assert record['cuda_builds'] == 42 and record['triton_builds'] == 3
    with tarfile.open(ROOT/'research/runtime-performance/prefill-small-bundle-r37-production-cpu-proof-20261104.tar.gz') as archive:
        for name, row in {**record['cuda_records'], **record['triton_records']}.items():
            binary = archive.extractfile('proof/device/'+name+'.cubin').read()
            assert hashlib.sha256(binary).hexdigest() == row['cubin_sha256']
            sass = archive.extractfile('proof/device/'+name+'.sass').read().decode()
            encoded = b''.join(int(x,16).to_bytes(8,'little') for x in re.findall(r'/\* 0x([0-9a-f]{16}) \*/',sass))
            assert len(encoded) == row['sass_bytes'] and hashlib.sha256(encoded).hexdigest() == row['sass_sha256']
            if name.endswith('-off') or (name.startswith('guard-') and name.endswith('-on')):
                control = archive.extractfile('proof/device/'+name.rsplit('-',1)[0]+'-base.cubin').read()
                assert binary == control
