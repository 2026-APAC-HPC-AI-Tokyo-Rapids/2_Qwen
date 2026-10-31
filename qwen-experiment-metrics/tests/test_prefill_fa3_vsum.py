"""Native V, composed dispatch, and all-GPU refusal contracts without CUDA."""
import ast
import json
import os
import runpy
import shlex
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'scripts/select-firmus-prefill-fa3-vsum.py'
GATE=runpy.run_path(str(SCRIPT))
PIN=GATE['KEY']+'='+GATE['CONTROL']+'\nOTHER=kept\n'


@pytest.fixture
def dispatch(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    for patch in ('prefill-fp8-integration.patch','prefill-fa3-pair.patch','prefill-fa3-vsum.patch'):
        subprocess.run(['git','apply','--include=python/sglang/srt/layers/attention/prefill_fp8.py',str(ROOT/'patches/sglang'/patch)],cwd=tmp_path,check=True)
    program=ast.parse((tmp_path/'python/sglang/srt/layers/attention/prefill_fp8.py').read_text())
    program.body=[n for n in program.body if isinstance(n,ast.FunctionDef) and n.name in ('prefill_candidate','pair_candidate')]
    class Table:
        def __init__(self,_): pass
        def read_text(self): return (ROOT/'kernels/fa3-verify/prefill-candidates.json').read_text()
    scope={'json':json,'os':os,'lru_cache':lru_cache,'Path':Table}
    exec(compile(program,'<real routing>','exec'),scope)  # noqa: S102 - execute actual host-only dispatch
    return scope


@pytest.mark.parametrize('lengths,n', [((3994,3994),96),((3900,4096),96),((3994,),0),((3584,),0),((4096,),0),((3583,),0),((3899,4096),0),((4097,),0),((3994,)*3,0),(None,0)])
def test_composed_host_dispatch(dispatch,monkeypatch,lengths,n):
    monkeypatch.setenv(GATE['KEY'],GATE['DISPATCH'])
    assert dispatch['pair_candidate'](lengths).get('native_v_n',0)==n
    monkeypatch.setenv(GATE['KEY'],GATE['CONTROL'])
    assert not dispatch['pair_candidate'](lengths).get('native_v_n')


@pytest.fixture
def screen(tmp_path):
    env=tmp_path/'prefill.env'; env.write_text(PIN)
    now=time.time(); pair=[]
    for worker in range(3):
        directory=tmp_path/f'prefill-{worker}'; directory.mkdir()
        for rank in range(4):
            identity={'worker':f'prefill-{worker}','rank':rank,'uuid':f'gpu-{worker}-{rank}','run_id':'123-pair'}
            pair.append(identity)
            metric={'relative_l2':.01,'max_scaled_error':.02}
            shape={'byte_image_first':True,'producer_bytes_equal':True,'replays':2,'candidate_fp32':[metric]*3,'control_fp32':[metric]*3,'samples_us':{order:{'control':[900.]*10,'candidate':[895.]*10} for order in ('forward','reverse')}}
            record={**identity,'run_id':'123-vsum','schema':GATE['SCHEMA'],'device':'NVIDIA H200','finished_at':now,'elapsed_seconds':60,'passed':True,'shapes':{'+'.join(map(str,s)):shape for s in GATE['SHAPES']},'endurance':{'calls':1200,'bytes_equal':True,'dispatch_equal':True,'initial_fp32':metric,'final_fp32':metric}}
            single={'row':'n192-direct-o-varlen-pipe','writer':'fp32','producer_bytes_equal':True,'attention_bytes_equal':True,'replays':2,'candidate_fp32':[metric]*3,'control_fp32':[metric]*3}
            record['singles']={str(lengths[0]):single for lengths in GATE['SINGLE_SHAPES']}
            (directory/f'rank-{rank}.json').write_text(json.dumps(record))
    pair_path=tmp_path/'pair.json'; pair_path.write_text(json.dumps({'schema':'firmus-prefill-fa3-pair/v1','adopted':True,'gpus':pair}))
    return tmp_path,env,pair_path,now+100


def aggregate(screen,**kwargs):
    directory,env,pair,deadline=screen
    return GATE['aggregate'](directory,env,pair,kwargs.get('run_id','123-vsum'),kwargs.get('deadline',deadline),kwargs.get('workers_ok',True))


def test_all_twelve_adopt_atomic_composite(screen):
    assert aggregate(screen)['adopted']
    assert screen[1].read_text()==PIN.replace(GATE['CONTROL'],GATE['DISPATCH'])
    assert not list(screen[0].glob('*.tmp'))


def test_independent_slurm_steps_may_permute_worker_gpu_assignment(screen):
    pair=json.loads(screen[2].read_text())
    # Actual run 41240194309 swapped the two four-GPU workers on the head node.
    for entry in pair['gpus']:
        if entry['worker'] in ('prefill-0','prefill-1'):
            entry['worker']='prefill-1' if entry['worker']=='prefill-0' else 'prefill-0'
    screen[2].write_text(json.dumps(pair))
    assert aggregate(screen)['adopted']
    assert screen[1].read_text()==PIN.replace(GATE['CONTROL'],GATE['DISPATCH'])


def test_component_diagnostics_keep_producer_out_of_attention_timing(monkeypatch):
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    events=[]
    rows=2
    common={'lengths':(3994,3994),'rows':rows,'rope_args':('rope',),'q':'q','k':'k','v':'v','cu':'cu'}
    outputs={mode:(torch.empty(rows,64,128),torch.empty(rows,64)) for mode in ('control','candidate')}
    state={'image':object(),'n':96,'output':outputs,'weight':object(),'o_outputs':{mode:object() for mode in outputs}}

    def native(common_arg,image,n):
        assert common_arg is common and image is state['image'] and n==96
        events.append('native')

    def old(*args):
        assert args==('rope',)
        events.append('old')

    def attention(*args):
        index=args[-1]
        mode=('control','candidate')[index]
        assert args[:3]==('q','k','v') and args[4]=='cu'
        assert args[5] is outputs[mode][0] and args[6] is outputs[mode][1]
        assert args[9:11]==(3994,128**-.5)
        assert (args[3] is state['image']) if index else args[3].numel()==0
        events.append('attention-'+mode)

    def gemm(data,weight,out,**kwargs):
        assert data[0].shape==(rows,8192) and weight is state['weight'] and kwargs=={'compiled_dims':'nk'}
        assert any(out is x for x in state['o_outputs'].values())
        events.append('o')

    snapshots=[]
    def measure(_,checks,timed):
        start=len(events)
        for call in timed['timed'].values(): call()
        snapshots.append(events[start:])
        return {order:{'control':[900.]*10,'candidate':[895.]*10} for order in ('forward','reverse')}

    scope=GATE['component_timings'].__globals__
    monkeypatch.setitem(scope,'native_producer',native)
    monkeypatch.setitem(scope,'measure',measure)
    monkeypatch.setitem(sys.modules,'deep_gemm',SimpleNamespace(fp8_gemm_nt=gemm))
    monkeypatch.setitem(sys.modules,'sglang.kernels.ops.attention.rope',SimpleNamespace(apply_qknorm_rope_fp8_prefill_fp32=old))
    fake=SimpleNamespace(empty=lambda *args,device,**kwargs:torch.empty(*args,**kwargs),int32=torch.int32,float32=torch.float32,uint8=torch.uint8,ops=SimpleNamespace(apac_fa3_verify=SimpleNamespace(prefill_variant_fwd_out=attention)))
    checks={'prefill_specs':lambda:[{'name':GATE['row_name'](common['lengths'],candidate)} for candidate in (False,True)]}
    result=GATE['component_timings'](fake,checks,common,state)
    assert snapshots[0]==['old']*12+['native']*12
    assert snapshots[1]==['attention-control','o']*12+['attention-candidate','o']*12
    assert result['producer_summary']['savings_us']['pooled']==5
    assert result['attention_o_summary']['savings_us']['pooled']==5


@pytest.mark.parametrize('case',['missing','malformed','stale','uuid','allocation','late','elapsed','failed','shape','single_missing','single_row','single_writer','single_producer','single_output','single_replay','single_numeric','single_timing','old_schema','dump','producer','replay','numeric','reverse','incomplete','nan','overflow','endurance','bytes','dispatch','pair_failed','pair_uuid','timeout'])
def test_one_failed_cell_keeps_env(screen,case):
    path=screen[0]/'prefill-2/rank-3.json'; record=json.loads(path.read_text()); shape=record['shapes']['3994+3994']
    if case=='missing': path.unlink()
    elif case=='malformed': path.write_text('{')
    elif case=='timeout': pass
    elif case=='overflow': path.write_text(path.read_text().replace('"passed": true','"passed": true, "extra": 1e309',1))
    elif case.startswith('pair_'):
        pair=json.loads(screen[2].read_text())
        if case=='pair_failed': pair['adopted']=False
        else: pair['gpus'][-1]['uuid']='foreign'
        screen[2].write_text(json.dumps(pair))
    else:
        if case=='stale': record['run_id']='old'
        elif case=='uuid': record['uuid']='gpu-0-0'
        elif case=='allocation':
            pair=json.loads(screen[2].read_text()); pair['gpus'][-1]['run_id']='999-pair'; screen[2].write_text(json.dumps(pair))
        elif case=='late': record['finished_at']=screen[3]+1
        elif case=='elapsed': record['elapsed_seconds']=GATE['SECONDS']
        elif case=='failed': record['passed']=False
        elif case=='shape': record['shapes'].pop('3994+3994')
        elif case=='old_schema': record['schema']='firmus-prefill-fa3-vsum/v1'
        elif case=='single_missing': record['singles'].pop('3584')
        elif case=='single_row': record['singles']['3994']['row']='m192n96-direct-o-varlen-pipe-vsum'
        elif case=='single_writer': record['singles']['3994']['writer']='native'
        elif case=='single_producer': record['singles']['3994']['producer_bytes_equal']=False
        elif case=='single_output': record['singles']['3994']['attention_bytes_equal']=False
        elif case=='single_replay': record['singles']['3994']['replays']=1
        elif case=='single_numeric': record['singles']['3994']['candidate_fp32'][0]['relative_l2']=.051
        elif case=='single_timing': record['singles']['3994']['samples_us']={}
        elif case=='dump': shape['byte_image_first']=False
        elif case=='producer': shape['producer_bytes_equal']=False
        elif case=='replay': shape['replays']=1
        elif case=='numeric': shape['candidate_fp32'][0]['relative_l2']=.051
        elif case=='reverse': shape['samples_us']['reverse']['candidate']=[895.01]*10
        elif case=='incomplete': shape['samples_us']['forward']['control'].pop()
        elif case=='nan': shape['samples_us']['forward']['candidate'][0]=float('nan')
        elif case=='endurance': record['endurance']['calls']=1199
        elif case=='bytes': record['endurance']['bytes_equal']=False
        elif case=='dispatch': record['endurance']['dispatch_equal']=False
        path.write_text(json.dumps(record))
    result=aggregate(screen,workers_ok=case!='timeout')
    assert not result['adopted'] and screen[1].read_text()==PIN
    json.dumps(result,allow_nan=False)


def test_atomic_rename_failure_keeps_env(screen,monkeypatch):
    def fail(*_): raise OSError('rename failure')
    monkeypatch.setattr(Path,'replace',fail)
    assert not aggregate(screen)['adopted'] and screen[1].read_text()==PIN
    assert not list(screen[0].glob('*.tmp'))


def block():
    source=(ROOT/'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    return '# Default off. Native V'+source.split('# Default off. Native V',1)[1].split('\nsrun --exact',1)[0]


def test_timeout_continues_with_round12_control(tmp_path):
    for directory in ('source','scripts','logs','results/prefill-fa3-pair'): (tmp_path/directory).mkdir(parents=True,exist_ok=True)
    (tmp_path/'source/prefill-kernels.env').write_text(PIN)
    (tmp_path/'results/prefill-fa3-pair/selection.json').write_text(json.dumps({'schema':'firmus-prefill-fa3-pair/v1','adopted':True,'gpus':[]}))
    (tmp_path/'scripts'/SCRIPT.name).write_text(SCRIPT.read_text())
    preamble=f'''set -euo pipefail
ROOT={shlex.quote(str(tmp_path))}
ART="$ROOT/logs"
prefill_pp=4
HEAD=head
TAIL=tail
SLURM_JOB_ID=123
SGLANG=unused
jq() {{ printf 'true\\n'; }}
mounts() {{ printf 'unused\\n'; }}
timeout() {{ return 124; }}
uv() {{ shift 5; {shlex.quote(sys.executable)} "$@"; }}
'''
    subprocess.run(['bash','-c',preamble+block()+'\nprintf continued >"$ROOT/continued"\n'],check=True,capture_output=True,text=True)
    assert (tmp_path/'continued').read_text()=='continued'
    assert (tmp_path/'source/prefill-kernels.env').read_text()==PIN
    assert not json.loads((tmp_path/'results/prefill-fa3-vsum/selection.json').read_text())['adopted']


def test_native_writer_and_gate_wiring():
    config=json.loads((ROOT/'configs/firmus-prefill-pp4.json').read_text())
    assert config['prefill']['fa3_vsum_gate'] is True
    assert config['sglang_overlay_patches'][-1]=='patches/sglang/prefill-fa3-vsum.patch'
    assert config['images']['sglang']['sha256']==config['container_sha256'] and config['container_sha256'].startswith('a8232b16')
    source=(ROOT/'patches/sglang/prefill-fa3-vsum.patch').read_text()
    for contract in ('qknorm_rope_prefill_native_kernel','native_offset<NativeN>','local<length','max_blocks*NativeN','self.attn.fixed_prefill_output','forward_metadata.cu_seqlens_q','apply_qknorm_rope_fp8_prefill_native'):
        assert contract in source
    build=(ROOT/'kernels/fa3-verify/build.py').read_text()
    assert 'str(root / "prefill-vsum.patch")' in build and '"prefill_native_image"' in build
    gate=SCRIPT.read_text()
    assert 'compiled_dims="nk"' in gate and 'prefill_dump_native' in gate
    assert gate.index('torch.ops.apac_fa3_verify.prefill_dump_native')<gate.index('for mode,call in state["attention"].items()')
    assert 'timeout --kill-after=2 100 srun' in block()
    assert 'select-firmus-prefill-fa3-vsum.py' in (ROOT/'scripts/provision-firmus-framework.sh').read_text()
    table=json.loads((ROOT/'kernels/fa3-verify/prefill-candidates.json').read_text())
    assert [(row['native_v_n'],row['variant']) for row in table if row.get('native_v_n') and not row.get('retired')]==[(96,57568)]


def test_deployed_offset_matches_exhaustive_cute_maps(tmp_path):
    import hashlib
    import struct
    patch=(ROOT/'patches/sglang/prefill-fa3-vsum.patch').read_text()
    added='\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    start='template<int N> __device__ __forceinline__ int native_offset'
    function=start+added.split(start,1)[1].split('return out; }',1)[0]+'return out; }'
    function=function.replace('__device__ __forceinline__ ','')
    source=tmp_path/'map.cpp'
    source.write_text('#include <cstdio>\n'+function+'\nint main(){for(int x=0;x<96*128;++x)std::printf("%d\\n",native_offset<96>(x/128,x%128));}\n')
    binary=tmp_path/'map'
    subprocess.run(['g++','-O2','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True)
    values=[int(x) for x in subprocess.check_output([str(binary)],text=True).splitlines()]
    proof=json.loads((ROOT/'research/runtime-performance/fa3-prefill-vsum-cpu-validation-20261101.json').read_text())
    for n,offset in ((96,0),):
        mapping=values[offset:offset+n*128]
        assert set(mapping)==set(range(n*128))
        assert hashlib.sha256(struct.pack('<'+'i'*len(mapping),*mapping)).hexdigest()==proof['layouts'][str(n)]['map_i32le_sha256']
        assert proof['layouts'][str(n)]['validated_stages']==[0,1]
    # The dedicated producer/dump gate must remain the sole adoption route.
    legacy=runpy.run_path(str(ROOT/'scripts/select-firmus-prefill-fa3.py'))
    assert not any(name.endswith('-vsum') for name in legacy['NAMES'])


def test_native_bulk_stride_does_not_enter_unused_tensor_map(tmp_path):
    patch=(ROOT/'kernels/fa3-verify/prefill-vsum.patch').read_text()
    added='\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    assert 'select<1, 0, 2, 3>(descriptor_stride_V)' in added
    assert 'get<2>(params.stride_V)' in added
    body='auto descriptor_stride_V = [&] {'+added.split('auto descriptor_stride_V = [&] {',1)[1].split('}();',1)[0]+'}();'
    source=tmp_path/'descriptor.cpp'
    source.write_text('''#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <tuple>
struct _1 {};
using Stride=std::tuple<int64_t,_1,int64_t,int64_t>;
template<int I,class T> auto get(T const& value){return std::get<I>(value);}
template<class... T> auto make_stride(T... value){return std::make_tuple(value...);}
struct Args {std::tuple<int,int,int,int> shape_K; int headdim_v; Stride stride_V;};
template<bool NativeV> auto descriptor(Args const& args){
'''+body+'''
return descriptor_stride_V;
}
int main(){for(int n:{96,192}){Args a{{7988,128,4,1},128,{512,_1{},int64_t(42*n*128),0}};
auto native=descriptor<true>(a), control=descriptor<false>(a);
assert(get<0>(native)==512 && get<2>(native)==128 && get<3>(native)==0);
assert(get<2>(control)==get<2>(a.stride_V) && get<2>(a.stride_V)==42*n*128);
}}
''')
    binary=tmp_path/'descriptor'
    subprocess.run(['g++','-O2','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary)],check=True)


@pytest.mark.parametrize('lengths,fixed,native', [((3994,3994),True,True),((3900,4096),True,True),((4096,3900),True,True),((4096,4096),True,True),((3994,),True,False),((4096,),True,False),((3584,),True,False),((3899,4096),True,False),((3900,4097),True,False),((3994,)*3,True,False),(None,True,False),((3994,3994),False,False)])
def test_actual_model_writer_branch_uses_same_host_pair_condition(dispatch,monkeypatch,lengths,fixed,native):
    import textwrap
    from types import SimpleNamespace
    monkeypatch.setenv(GATE['KEY'],GATE['DISPATCH'])
    patch=(ROOT/'patches/sglang/prefill-fa3-vsum.patch').read_text().split('+++ b/python/sglang/srt/models/qwen3_moe.py',1)[1]
    after='\n'.join(line[1:] for line in patch.splitlines() if line.startswith((' ','+')))
    fragment='                selected = pair_candidate'+after.split('                selected = pair_candidate',1)[1]
    calls=[]
    scope={**dispatch,'self':SimpleNamespace(attn=SimpleNamespace(fixed_prefill_output=fixed)), 'forward_batch':SimpleNamespace(extend_seq_lens_cpu=lengths), 'torch':SimpleNamespace(empty=lambda *a,**k:'image',uint8='uint8'), 'k':SimpleNamespace(device='cuda'), '_PREFILL_QKNORM':'fp32', 'rope_args':(), 'kv_args':(), 'contiguous_k':None,'contiguous_v':None,'get_attn_backend':lambda:SimpleNamespace(forward_metadata=SimpleNamespace(cu_seqlens_q='cu'))}
    for name in ('native','fp32',''):
        scope['apply_qknorm_rope_fp8_prefill'+('_'+name if name else '')]=lambda *args,name=name:calls.append(name)
    exec(compile(textwrap.dedent(fragment),'<actual model writer>','exec'),scope)  # noqa: S102 - actual patched branch with stubs
    assert calls==['native' if native else 'fp32']
    assert (scope['self'].attn.prefill_v_image is not None)==native


def test_published_ids_retained_without_retired_native_build(tmp_path,dispatch,monkeypatch):
    table=json.loads((ROOT/'kernels/fa3-verify/prefill-candidates.json').read_text())
    assert len(table)==15
    assert table[11]['name']=='n192-direct-o-varlen-pipe'
    assert table[12]['name']=='m192n96-direct-o-varlen-pipe'
    assert table[13]['name']=='n192-direct-o-varlen-pipe-vsum' and table[13]['retired'] is True
    assert table[14]['name']=='m192n96-direct-o-varlen-pipe-vsum'
    with pytest.raises(StopIteration):
        dispatch['prefill_candidate'](table[13]['name'])
    monkeypatch.setenv(GATE['KEY'],GATE['DISPATCH'])
    assert dispatch['pair_candidate']((3994,3994))['id']==14
    assert dispatch['pair_candidate']((3994,))['id']==11
    build=runpy.run_path(str(ROOT/'kernels/fa3-verify/build.py'))
    tasks=build['generate_prefill_dispatch'](ROOT/'kernels/fa3-verify',tmp_path)
    assert not any(name=='prefill_n192_direct_o_varlen_pipe_vsum' for _,name,_ in tasks)
    assert any(name=='prefill_m192n96_direct_o_varlen_pipe_vsum' for _,name,_ in tasks)
    header=(tmp_path/'prefill-candidates.h').read_text()
    assert '{"n192-direct-o-varlen-pipe-vsum", false, false, nullptr, 0}' in header
    wrapper=(ROOT/'kernels/fa3-verify/wrapper.cpp').read_text()
    assert 'TORCH_CHECK(prefill_candidates[id].launch, "FA3 Prefill candidate retired")' in wrapper
    assert 'if (c.launch) c.launch' in wrapper
    writer=(ROOT/'patches/sglang/prefill-fa3-vsum.patch').read_text()
    dump=(ROOT/'kernels/fa3-verify/prefill_native_image.cu').read_text()
    assert 'integral_constant<int,192>' not in writer and 'if constexpr (N==192)' not in writer
    assert 'prefill_dump_native_kernel<192>' not in dump


def test_single_checks_have_no_timing_and_require_served_bytes():
    source=SCRIPT.read_text()
    tree=ast.parse(source)
    single=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='single_check')
    names={n.id for n in ast.walk(single) if isinstance(n,ast.Name)}
    assert 'measure' not in names and 'timing_summary' not in names and 'native_producer' not in names
    assert {'attention','pair_candidate','apply_qknorm_rope_fp8_prefill_fp32'}<=names
    assert len(GATE['SHAPES'])==7 and all(len(s)==2 for s in GATE['SHAPES'])
    assert GATE['SINGLE_SHAPES']==((3994,),(4096,),(3584,))
    assert GATE['row_name']((3994,),True)=='n192-direct-o-varlen-pipe'
    assert 'single Q/K/V or paged KV changed' in source
    assert 'single attention differs from round-12 served row' in source


@pytest.mark.parametrize('failure',[None,'q','k','v','kc','vc','attention','scale'])
def test_single_gate_replays_and_rejects_each_producer_or_attention_byte_change(dispatch,monkeypatch,failure):
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    rows=2
    common={'lengths':(3994,),'rows':rows,'packed':torch.zeros(4),'cu':torch.tensor([0,rows]),'rope_args':()}
    common.update({name:torch.zeros((rows if name not in ('kc','vc') else rows+1,64 if name=='q' else 4,128),dtype=torch.uint8) for name in ('q','k','v','kc','vc')})
    calls=[]

    def writer(*_):
        calls.append(common['packed'].clone())
        value=int(common['packed'].abs().sum())%100
        for name in ('q','k','v','kc','vc'):
            common[name].fill_(value)
        if failure in ('q','k','v','kc','vc') and len(calls)%2==0:
            common[failure].reshape(-1)[0]+=1

    def output():
        return common['q'].clone(),torch.ones((64,rows+2)).T[:rows]

    def attention(*_,**kwargs):
        assert kwargs['v_image'] is None and kwargs['fixed'] is True
        assert dispatch['pair_candidate'](kwargs['lengths'])['name']=='n192-direct-o-varlen-pipe'
        out,scale=output()
        if failure=='attention': out[0,0,0]+=1
        if failure=='scale': scale[0,0]=2
        return out.view(rows,8192),scale

    def control(*_):
        writer()
        return output()

    monkeypatch.setitem(sys.modules,'sglang.kernels.ops.attention.rope',SimpleNamespace(apply_qknorm_rope_fp8_prefill_fp32=writer))
    monkeypatch.setitem(sys.modules,'sglang.srt.layers.attention.prefill_fp8',SimpleNamespace(attention=attention,pair_candidate=dispatch['pair_candidate'],prefill_candidate=dispatch['prefill_candidate']))
    checks={'prefill_selection_step':lambda common,name,layers:control,'prefill_fp32_reference':lambda common:(torch.tensor([0]),torch.zeros((1,64,128))),'prefill_assert_errors':lambda *args:{'relative_l2':.01,'max_scaled_error':.02}}
    if failure:
        with pytest.raises(AssertionError,match='single Q/K/V or paged KV changed|single attention differs'):
            GATE['single_check'](torch,checks,common)
    else:
        record=GATE['single_check'](torch,checks,common)
        assert record['producer_bytes_equal'] and record['attention_bytes_equal'] and record['replays']==2
        assert len(calls)==6 and not torch.equal(calls[0],calls[2]) and not torch.equal(calls[2],calls[4])


def test_native_v_lanes_cover_exact_bytes_with_full_sector_stores(tmp_path):
    """Run deployed integer expressions against the independently checked CuTe map."""
    patch=(ROOT/'patches/sglang/prefill-fa3-vsum.patch').read_text()
    added='\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    start='template<int N> __device__ __forceinline__ int native_offset'
    offset=start+added.split(start,1)[1].split('return out; }',1)[0]+'return out; }'
    offset=offset.replace('__device__ __forceinline__ ','')
    kernel=added.split('void qknorm_rope_prefill_native_kernel(',1)[1]
    geometry='const int head='+kernel.split('const int head=',1)[1].split('const int h=',1)[0]
    second=kernel.split('vy=*reinterpret_cast<const uint4*>(tile+',1)[1].split(');',1)[0]
    write=kernel.split('const int native_base=',1)[1]
    base='const int native_base='+write.split(';',1)[0]+';'
    stores=[line.split('dst[',1)[1].split(']=',1)[0] for line in write.splitlines() if 'dst[' in line]
    assert len(stores)==2
    source=tmp_path/'lanes.cpp'
    source.write_text('''#include <array>
#include <cassert>
#include <set>
'''+offset+'''
struct Index {int x,y;};
struct Params {int num_qo_heads=64,num_kv_heads=4;};
int main(){
    constexpr int NativeN=96;
    Params p;
    std::array<int,12288> owners{};
    for(int segment=0;segment<3;++segment){
        for(int head_id=0;head_id<72;++head_id){
            std::array<std::array<std::set<int>,16>,8> sectors;
            for(int tid=0;tid<256;++tid){
                Index threadIdx{tid,0},blockIdx{segment,head_id};
'''+geometry+'''
                const int second_col='''+second+''';
                if(head_id<68){
                    assert(token==segment*32+tid/8);
                    assert(col==(tid%8)*8 && second_col==col+64);
                    continue;
                }
                assert(token==segment*32+tid%32);
                assert(col%16==0 && col>=0 && second_col+7<128);
                assert(second_col==col+8);
                const int local=token;
'''+base+'''
                for(int i=0;i<8;++i){
                    const int a='''+stores[0]+''';
                    const int b='''+stores[1]+''';
                    assert(a==native_offset<96>(local,col+i));
                    assert(b==native_offset<96>(local,second_col+i));
                    assert(a>=0 && a<12288 && b>=0 && b<12288);
                    if(head_id==68){++owners[a];++owners[b];}
                    sectors[tid/32][i].insert(a/32);
                    sectors[tid/32][8+i].insert(b/32);
                }
            }
            if(head_id>=68)for(auto const& warp:sectors)for(auto const& instruction:warp)assert(instruction.size()==1);
        }
    }
    for(int count:owners)assert(count==1);
}
''')
    binary=tmp_path/'lanes'
    subprocess.run(['g++','-O2','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary)],check=True,capture_output=True)
