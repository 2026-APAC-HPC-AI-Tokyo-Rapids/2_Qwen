"""History-independent overlays, published-result decisions and startup ordering."""
import ast
import gzip
import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GATE = runpy.run_path(str(ROOT/'scripts/select-firmus-prefill-r37.py'))
KEYS = GATE['KEYS']
IMAGE = 'e'*64


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


def samples(saving=1):
    return {o: {'control':[100.0]*12,'candidate':[100.0-saving]*12} for o in ('forward','reverse')}


@pytest.fixture
def published(tmp_path,monkeypatch):
    # These are synthetic decision fixtures, not GPU performance evidence.
    gate = GATE['aggregate']; monkeypatch.setattr(gate.__globals__['time'],'time',lambda:1000)
    prior = tmp_path/'pair.json'
    records = []
    for worker in range(3):
        directory = tmp_path/f'prefill-{worker}'; directory.mkdir()
        for rank in range(4):
            uuid = f'gpu-{worker}-{rank}'
            records.append({'worker':f'prefill-{worker}','rank':rank,'uuid':uuid,'run_id':'42-old'})
            for candidate in KEYS:
                r = {'schema':GATE['SCHEMA'],'run_id':'42-new','candidate':candidate,'worker':directory.name,'rank':rank,'uuid':uuid,'device':'NVIDIA H200','image_sha256':IMAGE,'healthy':True,'elapsed_seconds':55,'finished_at':995,'exact':{c:dict.fromkeys(GATE['STATES'],True) for c in GATE['CASES'][candidate]},'endurance':{'calls':1200,'buffers':2,'states':['initial','changed'],'calls_per_state':[600,600],'every_call_equal':True,'restored_equal':True},'timing':{p:samples() for p in GATE['COMPONENTS'][candidate]},'source_sha256':'a'*64,'snapshot_sha256':'b'*64}
                (directory/f'{candidate}-rank-{rank}.json').write_text(json.dumps(r))
    prior.write_text(json.dumps({'gpus':records,'adopted':True}))
    env = tmp_path/'prefill.env'; env.write_text('DG_SM90_PREFILL_FC1_FIRST_K=1\nDG_SM90_PREFILL_DENSE_FIRST_K=1\n'+''.join(k+'=0\n' for k in KEYS.values()))
    def run():
        return gate(tmp_path,env,prior,'42-new',IMAGE,1010,[124,137,0])
    def alter(candidate, mutate, worker=0, rank=0):
        p = tmp_path/f'prefill-{worker}'/f'{candidate}-rank-{rank}.json'
        r = json.loads(p.read_text()); mutate(r); p.write_text(json.dumps(r))
    return tmp_path,env,run,alter


def test_independent_adoption_uses_publication_even_after_wrapper_timeout(published):
    _,env,run,_ = published
    result = run()
    assert all(d['adopted'] for d in result['candidates'].values())
    assert result['worker_exit_codes'] == [124,137,0]
    assert all(k+'=1' in env.read_text() for k in KEYS.values())
    assert 'DG_SM90_PREFILL_FC1_FIRST_K=1' in env.read_text()
    assert 'DG_SM90_PREFILL_DENSE_FIRST_K=1' in env.read_text()
    assert set(result['candidates']) == {'B','C'}


@pytest.mark.parametrize('candidate',KEYS)
@pytest.mark.parametrize('failure',('missing','unhealthy','late','wrong-image','wrong-run','uuid','case','bytes','endurance','changed','buffers','restored','rounds','nonfinite','source','snapshot'))
def test_each_invalid_candidate_keeps_others_independent(published,candidate,failure):
    directory,env,run,alter = published
    if failure == 'missing':
        (directory/'prefill-0'/f'{candidate}-rank-0.json').unlink()
    else:
        def mutate(r):
            if failure == 'unhealthy': r['healthy'] = False
            elif failure == 'late': r['finished_at'] = 1000.001
            elif failure == 'wrong-image': r['image_sha256'] = 'c'*64
            elif failure == 'wrong-run': r['run_id'] = '41-new'
            elif failure == 'uuid': r['uuid'] = 'gpu-0-1'
            elif failure == 'case': r['exact'].pop(next(iter(r['exact'])))
            elif failure == 'bytes': next(iter(r['exact'].values()))['boundary'] = False
            elif failure == 'endurance': r['endurance']['calls'] = 1199
            elif failure == 'changed': r['endurance']['states'] = ['initial','initial']
            elif failure == 'buffers': r['endurance']['buffers'] = 1
            elif failure == 'restored': r['endurance']['restored_equal'] = False
            elif failure == 'rounds': next(iter(r['timing'].values()))['forward']['control'].pop()
            elif failure == 'nonfinite': next(iter(r['timing'].values()))['forward']['control'][0] = float('nan')
            elif failure == 'source': r['source_sha256'] = 'f'*64
            elif failure == 'snapshot': r['snapshot_sha256'] = 'f'*64
        alter(candidate,mutate)
    result = run()['candidates']
    assert result[candidate]['adopted'] is False
    assert all(d['adopted'] for c,d in result.items() if c != candidate)
    assert KEYS[candidate]+'=0' in env.read_text()


@pytest.mark.parametrize('rank',range(12))
def test_all_twelve_gpus_are_required(published,rank):
    _,_,run,alter = published
    alter('B',lambda r:r.update(healthy=False),rank//4,rank%4)
    assert run()['candidates']['B']['adopted'] is False


@pytest.mark.parametrize('order',('forward','reverse'))
def test_both_order_medians_must_be_positive(order):
    values = [samples() for _ in range(12)]
    for value in values: value[order]['candidate'] = [100]*12
    with pytest.raises(AssertionError,match='nonpositive'): GATE['speed_rule'](values)


def test_zero_pooled_eight_positive_and_worst_gpu_branches():
    with pytest.raises(AssertionError): GATE['speed_rule']([samples(0) for _ in range(12)])
    with pytest.raises(AssertionError,match='nine'): GATE['speed_rule']([samples(1)]*8+[samples(-.1)]*4)
    with pytest.raises(AssertionError,match='one percent'): GATE['speed_rule']([samples(1)]*11+[samples(-1.001)])
    assert GATE['speed_rule']([samples(1)]*9+[samples(-1)]*3)['positive_gpus'] == 9


def test_pool_within_gpu_before_median():
    values = [samples(2) for _ in range(12)]
    for v in values:
        v['forward']['control'] = [80]*12
        v['forward']['candidate'] = [79]*12
        v['reverse']['control'] = [120]*12
        v['reverse']['candidate'] = [117]*12
    assert GATE['speed_rule'](values)['saving_median_us']['pooled'] == 2


def test_expired_aggregate_and_slurm_gpu_reassignment(published,monkeypatch):
    directory,_,run,_ = published
    prior = directory/'pair.json'; value = json.loads(prior.read_text())
    for i,gpu in enumerate(value['gpus']): gpu['uuid'] = f'GPU-gpu-{(i+1)%12//4}-{(i+1)%4}'
    prior.write_text(json.dumps(value))
    assert all(d['adopted'] for d in run()['candidates'].values())
    # Fresh control environment for the second, expired aggregation.
    (directory/'prefill.env').write_text(''.join(k+'=0\n' for k in KEYS.values()))
    monkeypatch.setattr(GATE['aggregate'].__globals__['time'],'time',lambda:1010.001)
    assert all(not d['adopted'] for d in run()['candidates'].values())


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


def test_budget_staging_prefill_flags_and_post_aiperf_collection(tmp_path):
    shell = (ROOT/'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    assert shell.index('# Decode greedy gate end.') < shell.index('# Round 37 reuses') < shell.index('bash "${ROOT}/source/worker.sh" prefill-0')
    assert 'SECONDS + 105 + 630 <= 1020' in shell and 'timeout --kill-after=2 95 srun' in shell
    assert 'prefill-r37-dispatch-${worker}.txt' in shell
    assert shell.index('scripts/official-aiperf.sh') < shell.index('prefill-r37-dispatch-${worker}.txt')
    assert GATE['CALLS'] == 1200 and GATE['ROUNDS'] == 12 and 33+55 < GATE['WORKER_SECONDS']
    assert 285+105+630 == 1020 and 286+105+630 > 1020
    wrapper = (ROOT/'scripts/run-firmus-prefill-r37.sh').read_text()
    assert wrapper.count('/opt/sglang/venv/bin/torchrun ') == 1
    assert 'jq ' not in wrapper  # the image has no jq; pass the host-parsed pin as argv
    assert 'source "${root}/source/prefill-kernels.env"' in wrapper
    for file in ('scripts/run-firmus-prefill-r37.sh','platforms/firmus/framework-comparison/sglang.sbatch'):
        assert '\\\n' not in (ROOT/file).read_text()
        subprocess.run(['bash','-n',str(ROOT/file)],check=True)
    for d in ('source','results'): (tmp_path/d).mkdir()
    config = json.loads((ROOT/'configs/firmus-prefill-pp4.json').read_text())
    runpy.run_path(str(ROOT/'scripts/prepare-firmus-pp4-settings.py'))['prepare'](config,ROOT,tmp_path)
    for key in KEYS.values():
        assert (tmp_path/'source/prefill-kernels.env').read_text().splitlines().count(key+'=0') == 1
        assert key not in (tmp_path/'source/decode-kernels.env').read_text()
    assert (tmp_path/'source/pp4-kernel-settings.json').read_bytes() == (tmp_path/'results/pp4-kernel-settings.json').read_bytes()
    assert json.loads((tmp_path/'source/pp4-kernel-settings.json').read_text())['small_bundle_r37_gate'] is True


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


def test_cpu_jit_preparation_warms_production_keys_without_launch(monkeypatch):
    import sys
    from types import SimpleNamespace as NS

    calls = []
    class Kernel:
        def warmup(self,*args,**kwargs): calls.append(('warmup',args[-2],kwargs))
    class Case:
        def __init__(self,*args):
            self.x=self.w=self.g=object(); self.eps=1e-6
            self.out={'candidate':[object() for _ in range(5)]}
    monkeypatch.setitem(sys.modules,'torch',NS(cuda=NS(set_device=lambda r:calls.append(('device',r)),synchronize=lambda:calls.append(('sync',)))))
    for name in ('deep_gemm', 'triton', 'triton.language', 'sglang.kernels.ops.quantization'):
        monkeypatch.setitem(sys.modules,name,NS(fp8_kernel=object()))
    monkeypatch.setitem(sys.modules,'sglang.kernels.ops.attention.rope',NS(_jit_rope_fp8_q_module=lambda:calls.append(('writer',))))
    monkeypatch.setitem(sys.modules,'sglang.kernels.ops.moe.prefill_moe_input',NS(_onepass_moe_input=Kernel(),_onepass_moe_input_div448=Kernel()))
    pool,pending = GATE['prepare_compilers'](NS(Onepass=Case),{'C':object()},2)
    for future in pending: future.result()
    pool.shutdown(wait=True)
    warmups = [c for c in calls if c[0]=='warmup']
    assert [c[1] for c in warmups] == [1,64,7988]*2
    assert all(c[2] == {'BLOCK_M':64,'num_warps':8,'num_stages':3,'grid':(128,)} for c in warmups)
    source = (ROOT/'scripts/select-firmus-prefill-r37.py').read_text()
    assert source.index('future.result()') < source.index("r['endurance'] = endurance") < source.index("r['timing'][component] = timing")


def test_compiler_thread_functions_have_no_imports_and_main_imports_before_pool():
    tree = ast.parse((ROOT/'scripts/select-firmus-prefill-r37.py').read_text())
    prepare = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'prepare_compilers')
    threads = [n for n in prepare.body if isinstance(n,ast.FunctionDef)]
    assert {n.name for n in threads} == {'writer','onepass'}
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for fn in threads for n in ast.walk(fn))
    pool_line = next(n.lineno for n in ast.walk(prepare) if isinstance(n,ast.Call) and ast.unparse(n.func) == 'ThreadPoolExecutor')
    imports = [n for n in prepare.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert all(n.lineno < pool_line for n in imports)
    modules = {alias.name if isinstance(n,ast.Import) else n.module for n in imports for alias in n.names}
    assert {'torch','deep_gemm','triton','triton.language','sglang.kernels.ops.quantization','sglang.kernels.ops.attention.rope','sglang.kernels.ops.moe.prefill_moe_input'} <= modules


def test_dense_snapshot_matches_sm90_production_dtype_shape_stride_and_qkv_order():
    torch = pytest.importorskip('torch')
    tensors = {}
    for p,n,k,value in (('q_proj',8192,4096,1),('k_proj',512,4096,2),('v_proj',512,4096,3),('o_proj',4096,8192,4)):
        tensors[f'self_attn.{p}.weight'] = torch.full((n,k),value,dtype=torch.float8_e4m3fn)
        tensors[f'self_attn.{p}.weight_scale_inv'] = ((torch.arange(n//128*k//128).reshape(n//128,k//128) % 251 + 1) / 256).to(torch.bfloat16)
    dense = GATE['dense_snapshot'](tensors)
    for (weight,scale),shape,scale_shape in zip(dense,((9216,4096),(4096,8192)),((72,32),(32,64)),strict=True):
        assert weight.dtype == torch.float8_e4m3fn and weight.shape == shape and weight.stride() == (shape[1],1)
        assert scale.dtype == torch.float32 and scale.shape == scale_shape and scale.stride() == (scale_shape[1],1)
    for start,end,p in ((0,8192,'q_proj'),(8192,8704,'k_proj'),(8704,9216,'v_proj')):
        assert torch.equal(dense[0][0][start:end].view(torch.uint8),tensors[f'self_attn.{p}.weight'].view(torch.uint8))
        assert torch.equal(dense[0][1][start//128:end//128],tensors[f'self_attn.{p}.weight_scale_inv'].float())
    assert torch.equal(dense[1][0].view(torch.uint8),tensors['self_attn.o_proj.weight'].view(torch.uint8))
    assert torch.equal(dense[1][1],tensors['self_attn.o_proj.weight_scale_inv'].float())
    assert all(tensors[f'self_attn.{p}.weight_scale_inv'].dtype == torch.bfloat16 for p in ('q_proj','k_proj','v_proj','o_proj'))


def test_snapshot_hashes_raw_bf16_checkpoint_before_dense_conversion(tmp_path,monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace as NS

    torch = pytest.importorskip('torch')
    prefix = 'model.layers.0.'
    tensors = {prefix+'self_attn.'+p+'.'+s: torch.full((2,2),1.0078125,dtype=torch.bfloat16 if s == 'weight_scale_inv' else torch.float8_e4m3fn) for p in ('q_proj','k_proj','v_proj','o_proj') for s in ('weight','weight_scale_inv')}
    tensors.update({prefix+s:torch.ones(2,dtype=torch.bfloat16) for s in ('self_attn.q_norm.weight','self_attn.k_norm.weight','post_attention_layernorm.weight','mlp.gate.weight')})
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':dict.fromkeys(tensors,'synthetic.safetensors')}))
    (tmp_path/'config.json').write_text(json.dumps({'rms_norm_eps':1e-6,'rope_theta':1000000}))
    monkeypatch.setitem(sys.modules,'safetensors',NS(safe_open=lambda *a,**kw:nullcontext(NS(get_tensor=tensors.__getitem__))))
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    arange = torch.arange
    monkeypatch.setattr(torch,'arange',lambda *a,**kw:arange(*a,**{k:v for k,v in kw.items() if k != 'device'}))
    expected = hashlib.sha256()
    for name,tensor in sorted(tensors.items()):
        expected.update(name.encode()); expected.update(tensor.view(torch.uint8).numpy().tobytes())
    weights,digest = GATE['snapshot_weights'](tmp_path)
    assert digest == expected.hexdigest()
    assert all(scale.dtype == torch.float32 and scale.is_contiguous() for _,scale in weights['A'])


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
