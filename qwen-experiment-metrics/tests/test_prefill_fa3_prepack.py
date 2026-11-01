"""Native V, composed dispatch, and all-GPU refusal contracts without CUDA."""
import ast
import copy
import gzip
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
SCRIPT=ROOT/'scripts/select-firmus-prefill-fa3-prepack.py'
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
    monkeypatch.setenv(GATE['KEY'],GATE['DISPATCH']['P'])
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
            shape={'row':GATE['ROWS']['P'],'variant':GATE['VARIANTS']['P'],'baseline':'C','byte_image_first':True,'producer_bytes_equal':True,'replays':2,'candidate_fp32':[metric]*3,'control_fp32':[metric]*3,'samples_us':{order:{'control':[900.]*10,'candidate':[895.]*10} for order in ('forward','reverse')}}
            candidate={'passed':True,'shapes':{'+'.join(map(str,s)):copy.deepcopy(shape) for s in GATE['SHAPES']},'endurance':{'calls':1200,'bytes_equal':True,'dispatch_equal':True,'initial_fp32':metric,'final_fp32':metric}}
            record={**identity,'run_id':'123-prepack','schema':GATE['SCHEMA'],'device':'NVIDIA H200','finished_at':now,'elapsed_seconds':60,'healthy':True,'binary_sha256':'b'*64,'image_sha256':'a'*64,'p3_enabled':True,'candidates':{'P':candidate,'P3':{'passed':False}}}
            single={'row':'n192-direct-o-varlen-pipe','writer':'fp32','producer_bytes_equal':True,'attention_bytes_equal':True,'replays':2,'candidate_fp32':[metric]*3,'control_fp32':[metric]*3}
            candidate['singles']={str(lengths[0]):copy.deepcopy(single) for lengths in GATE['SINGLE_SHAPES']}
            (directory/f'rank-{rank}.json').write_text(json.dumps(record))
    pair_path=tmp_path/'pair.json'; pair_path.write_text(json.dumps({'schema':'firmus-prefill-fa3-pair/v1','adopted':True,'gpus':pair}))
    return tmp_path,env,pair_path,now+100


def aggregate(screen,**kwargs):
    directory,env,pair,deadline=screen
    return GATE['aggregate'](directory,env,pair,kwargs.get('run_id','123-prepack'),kwargs.get('deadline',deadline),kwargs.get('workers_ok',True),'a'*64,kwargs.get('p3_enabled',True))


def test_all_twelve_adopt_atomic_composite(screen):
    assert aggregate(screen)['adopted']
    assert screen[1].read_text()==PIN.replace(GATE['CONTROL'],GATE['DISPATCH']['P'])
    assert not list(screen[0].glob('*.tmp'))


@pytest.mark.parametrize('case',['missing','malformed','stale','uuid','allocation','late','elapsed','failed','shape','single_missing','single_row','single_writer','single_producer','single_output','single_replay','single_numeric','single_timing','old_schema','dump','producer','replay','numeric','reverse','incomplete','nan','overflow','endurance','bytes','dispatch','pair_failed','pair_uuid','timeout'])
def test_one_failed_cell_keeps_env(screen,case):
    path=screen[0]/'prefill-2/rank-3.json'; record=json.loads(path.read_text()); candidate=record['candidates']['P']; shape=candidate['shapes']['3994+3994']
    if case=='missing': path.unlink()
    elif case=='malformed': path.write_text('{')
    elif case=='timeout': pass
    elif case=='overflow': path.write_text(path.read_text().replace('"healthy": true','"healthy": true, "extra": 1e309',1))
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
        elif case=='failed': candidate['passed']=False
        elif case=='shape': candidate['shapes'].pop('3994+3994')
        elif case=='old_schema': record['schema']='firmus-prefill-fa3-vsum/v1'
        elif case=='single_missing': candidate['singles'].pop('3584')
        elif case=='single_row': candidate['singles']['3994']['row']='m192n96-direct-o-varlen-pipe-vsum'
        elif case=='single_writer': candidate['singles']['3994']['writer']='native'
        elif case=='single_producer': candidate['singles']['3994']['producer_bytes_equal']=False
        elif case=='single_output': candidate['singles']['3994']['attention_bytes_equal']=False
        elif case=='single_replay': candidate['singles']['3994']['replays']=1
        elif case=='single_numeric': candidate['singles']['3994']['candidate_fp32'][0]['relative_l2']=.051
        elif case=='single_timing': candidate['singles']['3994']['samples_us']={}
        elif case=='dump': shape['byte_image_first']=False
        elif case=='producer': shape['producer_bytes_equal']=False
        elif case=='replay': shape['replays']=1
        elif case=='numeric': shape['candidate_fp32'][0]['relative_l2']=.051
        elif case=='reverse': shape['samples_us']['reverse']['candidate']=[895.01]*10
        elif case=='incomplete': shape['samples_us']['forward']['control'].pop()
        elif case=='nan': shape['samples_us']['forward']['candidate'][0]=float('nan')
        elif case=='endurance': candidate['endurance']['calls']=1199
        elif case=='bytes': candidate['endurance']['bytes_equal']=False
        elif case=='dispatch': candidate['endurance']['dispatch_equal']=False
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
    return '# P is revalidated; optional P3'+source.split('# P is revalidated; optional P3',1)[1].split('\nsrun --exact',1)[0]


@pytest.mark.parametrize("stage",["workers","aggregate"])
def test_timeout_continues_with_round12_control(tmp_path,stage):
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
jq() {{ if [[ "$*" == *sha256* ]]; then printf '%064d\\n' 0; else printf 'true\\n'; fi; }}
mounts() {{ printf 'unused\\n'; }}
timeout() {{ if [[ "$1" == --kill-after=2 ]]; then return 124; fi; shift 2; "$@"; }}
uv() {{ shift 5; {shlex.quote(sys.executable)} "$@"; }}
'''
    if stage=="aggregate":
        preamble+='timeout() { if [[ "$1" == --kill-after=2 ]]; then return 0; fi; return 124; }\n'
    subprocess.run(['bash','-c',preamble+block()+'\nprintf continued >"$ROOT/continued"\n'],check=True,capture_output=True,text=True)
    assert (tmp_path/'continued').read_text()=='continued'
    assert (tmp_path/'source/prefill-kernels.env').read_text()==PIN
    selection=tmp_path/'results/prefill-fa3-prepack/selection.json'
    if stage=="workers": assert not json.loads(selection.read_text())['adopted']
    else: assert not selection.exists()


def test_native_writer_and_gate_wiring():
    config=json.loads((ROOT/'configs/firmus-prefill-pp4.json').read_text())
    assert config['prefill']['fa3_prepack_gate'] is True
    assert config['prefill']['fa3_v3_gate'] is False
    assert config['sglang_overlay_patches'][-1]=='patches/sglang/prefill-fa3-vsum.patch'
    assert config['images']['sglang']['sha256']==config['container_sha256'] and config['container_sha256'].startswith('5a3df8c4')
    source=(ROOT/'patches/sglang/prefill-fa3-vsum.patch').read_text()
    for contract in ('qknorm_rope_prefill_native_kernel','native_offset<NativeN>','local<length','max_blocks*NativeN','self.attn.fixed_prefill_output','forward_metadata.cu_seqlens_q','prefill_prepack'):
        assert contract in source
    build=(ROOT/'kernels/fa3-verify/build.py').read_text()
    assert 'str(root / "prefill-vsum.patch")' in build and '"prefill_native_image"' in build
    gate=SCRIPT.read_text()
    assert 'compiled_dims="nk"' in gate and 'prefill_dump_native' in gate
    assert gate.index('torch.ops.apac_fa3_verify.prefill_dump_native')<gate.index('for mode,values in metrics.items():')
    assert 'timeout --kill-after=2 "${prefill_fa3_prepack_remaining}" srun' in block()
    assert 'select-firmus-prefill-fa3-prepack.py' in (ROOT/'scripts/provision-firmus-framework.sh').read_text()
    table=json.loads((ROOT/'kernels/fa3-verify/prefill-candidates.json').read_text())
    assert [(row['native_v_n'],row['variant']) for row in table if row.get('native_v_n') and not row.get('retired')]==[(96,41184),(96,172256)]


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
@pytest.mark.parametrize('candidate',['P','P3'])
def test_actual_model_writer_branch_uses_same_host_pair_condition(dispatch,monkeypatch,lengths,fixed,native,candidate):
    import textwrap
    from types import SimpleNamespace
    monkeypatch.setenv(GATE['KEY'],GATE['DISPATCH'][candidate])
    native=native and candidate in ('P','P3')
    patch=(ROOT/'patches/sglang/prefill-fa3-vsum.patch').read_text().split('+++ b/python/sglang/srt/models/qwen3_moe.py',1)[1]
    after='\n'.join(line[1:] for line in patch.splitlines() if line.startswith((' ','+')))
    fragment='                selected = pair_candidate'+after.split('                selected = pair_candidate',1)[1]
    calls=[]
    monkeypatch.setitem(sys.modules,'sglang.srt.layers.attention.decode_fa3_verify',SimpleNamespace(load=lambda:calls.append("load")))
    scope={**dispatch,'self':SimpleNamespace(attn=SimpleNamespace(fixed_prefill_output=fixed)), 'forward_batch':SimpleNamespace(extend_seq_lens_cpu=lengths), 'torch':SimpleNamespace(empty=lambda *a,**k:'image',uint8='uint8',ops=SimpleNamespace(apac_fa3_verify=SimpleNamespace(prefill_prepack=lambda *a:calls.append('pack')))), 'k':SimpleNamespace(device='cuda'), '_PREFILL_QKNORM':'fp32', 'rope_args':(), 'kv_args':(), 'contiguous_k':None,'contiguous_v':None,'get_attn_backend':lambda:SimpleNamespace(forward_metadata=SimpleNamespace(cu_seqlens_q='cu'))}
    for name in ('native','fp32',''):
        scope['apply_qknorm_rope_fp8_prefill'+('_'+name if name else '')]=lambda *args,name=name:calls.append(name)
    exec(compile(textwrap.dedent(fragment),'<actual model writer>','exec'),scope)  # noqa: S102 - actual patched branch with stubs
    assert calls==(['load','fp32','pack'] if native else ['fp32'])
    assert (scope['self'].attn.prefill_v_image is not None)==native


def test_published_ids_retained_without_retired_native_build(tmp_path,dispatch,monkeypatch):
    table=json.loads((ROOT/'kernels/fa3-verify/prefill-candidates.json').read_text())
    assert len(table)==18
    assert table[11]['name']=='n192-direct-o-varlen-pipe'
    assert table[12]['name']=='m192n96-direct-o-varlen-pipe'
    assert table[13]['name']=='n192-direct-o-varlen-pipe-vsum' and table[13]['retired'] is True
    assert table[14]['name']=='m192n96-direct-o-varlen-pipe-vsum' and table[14]['retired'] is True
    assert table[16]['retired'] is True and table[17]['variant']==172256
    with pytest.raises(StopIteration):
        dispatch['prefill_candidate'](table[13]['name'])
    monkeypatch.setenv(GATE['KEY'],GATE['DISPATCH']['P'])
    assert dispatch['pair_candidate']((3994,3994))['id']==15
    assert dispatch['pair_candidate']((3994,))['id']==11
    build=runpy.run_path(str(ROOT/'kernels/fa3-verify/build.py'))
    tasks=build['generate_prefill_dispatch'](ROOT/'kernels/fa3-verify',tmp_path)
    assert not any(name=='prefill_n192_direct_o_varlen_pipe_vsum' for _,name,_ in tasks)
    assert not any(name in ('prefill_m192n96_direct_o_varlen_pipe_vsum','prefill_m192n96_direct_o_varlen_pipe_sum') for _,name,_ in tasks)
    assert any(name=='prefill_m192n96_direct_o_varlen_pipe_native_v3' for _,name,_ in tasks)
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
    assert GATE['row_name']((3994,),'P')=='n192-direct-o-varlen-pipe'
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


def enable_p3(screen):
    for path in screen[0].glob('prefill-*/rank-*.json'):
        record=json.loads(path.read_text())
        record['candidates']['P3']=copy.deepcopy(record['candidates']['P'])
        for cell in record['candidates']['P3']['shapes'].values():
            cell.update(row=GATE['ROWS']['P3'],variant=GATE['VARIANTS']['P3'],baseline='P',p_bytes_equal=True,baseline_fp32=cell['control_fp32'])
        path.write_text(json.dumps(record))


def test_priority_and_candidate_local_rejection(screen):
    enable_p3(screen)
    assert aggregate(screen)['selected']=='P3'
    screen[1].write_text(PIN)
    path=screen[0]/'prefill-2/rank-3.json'; record=json.loads(path.read_text())
    record['candidates']['P3']['shapes']['3994+3994']['samples_us']['reverse']['candidate']=[896.]*10
    path.write_text(json.dumps(record))
    result=aggregate(screen)
    assert result['adopted'] and result['selected']=='P'
    assert screen[1].read_text()==PIN.replace(GATE['CONTROL'],GATE['DISPATCH']['P'])
    screen[1].write_text(PIN)
    record['candidates']['P']['passed']=False
    record['candidates']['P3']['passed']=True
    path.write_text(json.dumps(record))
    assert not aggregate(screen)['adopted'] and screen[1].read_text()==PIN


@pytest.mark.parametrize('case',['fatal','binary','image','duplicate_uuid','late_aggregate'])
def test_whole_group_failure_discards_even_complete_p_and_p3(screen,case):
    path=screen[0]/'prefill-2/rank-3.json'; record=json.loads(path.read_text())
    record['candidates']['P3']=copy.deepcopy(record['candidates']['P'])
    for cell in record['candidates']['P3']['shapes'].values(): cell.update(row=GATE['ROWS']['P3'],variant=GATE['VARIANTS']['P3'],baseline='P',p_bytes_equal=True,baseline_fp32=cell['control_fp32'])
    if case=='fatal': record['healthy']=False
    elif case=='binary': record['binary_sha256']='c'*64
    elif case=='image': record['image_sha256']='c'*64
    elif case=='duplicate_uuid': record['uuid']='gpu-0-0'
    path.write_text(json.dumps(record))
    result=aggregate(screen,deadline=time.time()-1) if case=='late_aggregate' else aggregate(screen)
    assert not result['adopted'] and screen[1].read_text()==PIN


@pytest.mark.parametrize('candidate,variant,n',[('P',41184,96),('P3',172256,96)])
@pytest.mark.parametrize('lengths,pair',[((3994,3994),True),((3900,4096),True),((4096,3901),True),((3994,),False),((4096,),False),((3584,),False),((3899,4096),False),((4096,4097),False),((3994,)*3,False),(None,False)])
def test_both_candidates_dispatch_boundaries(dispatch,monkeypatch,candidate,variant,n,lengths,pair):
    monkeypatch.setenv(GATE['KEY'],GATE['DISPATCH'][candidate])
    selected=dispatch['pair_candidate'](lengths)
    assert selected['variant']==(variant if pair else 224)
    assert selected.get('native_v_n',0)==(n if pair else 0)


def test_time_reservation_never_shortens_candidate(monkeypatch):
    monkeypatch.setattr(GATE['time'],'time',lambda:100.)
    with pytest.raises(AssertionError,match='insufficient time'):
        GATE['require_time'](145.,GATE['CANDIDATE_SECONDS'])
    GATE['require_time'](145.001,GATE['CANDIDATE_SECONDS'])
    assert GATE['ENDURANCE']==1200 and GATE['ROUNDS']==10 and GATE['REPS']==12


def test_production_pack_inverse_exhaustive_against_old_writer_and_host_map(tmp_path):
    import hashlib
    import struct
    source=(ROOT/'kernels/fa3-verify/prefill_prepack.cu').read_text()
    function=source.split('__host__ __device__ inline ',1)[1].split('__global__',1)[0]
    unit=tmp_path/'inverse.cpp'
    unit.write_text('#include <cstdio>\n'+function+'\nint main(){for(int x=0;x<96*128;++x)std::printf("%d\\n",prefill_v_logical_offset(x));}\n')
    binary=tmp_path/'inverse'
    subprocess.run(['g++','-O2',str(unit),'-o',str(binary)],check=True)
    inverse=list(map(int,subprocess.check_output([str(binary)],text=True).splitlines()))
    assert set(inverse)==set(range(96*128))
    for physical in range(0,96*128,16):
        base=inverse[physical]
        assert [base+(b&1)*128+(b&2)*512+(b&4)*64+(b&8)*64 for b in range(16)]==inverse[physical:physical+16]
    forward=[None]*len(inverse)
    for physical,logical in enumerate(inverse): forward[logical]=physical
    proof=json.loads((ROOT/'research/runtime-performance/fa3-prefill-vsum-cpu-validation-20261101.json').read_text())
    assert hashlib.sha256(struct.pack('<'+'i'*len(forward),*forward)).hexdigest()==proof['layouts']['96']['map_i32le_sha256']
    assert 'reinterpret_cast<uint4*>(dst+x)' in source
    assert 'prefill_prepack' in (ROOT/'kernels/fa3-verify/wrapper.cpp').read_text()
    # Raw bytes including signed zero and every FP8 encoding are moved, never converted.
    for lengths in GATE['SHAPES']:
        blocks=(max(lengths)+95)//96
        for request,length in enumerate(lengths):
            for head in range(4):
                for tile in (0,blocks-1):
                    logical=[((request*37+head*11+(tile*96+i//128)*128+i%128)&255) if tile*96+i//128<length else 0 for i in range(96*128)]
                    old_image=[0]*len(inverse)
                    for i,value in enumerate(logical): old_image[forward[i]]=value
                    packed=[logical[i] for i in inverse]
                    assert packed==old_image


@pytest.mark.parametrize('case',['row','variant'])
def test_candidate_identity_is_checked_at_aggregate(screen,case):
    path=screen[0]/'prefill-2/rank-3.json'; record=json.loads(path.read_text())
    record['candidates']['P']['shapes']['3994+3994'][case]=GATE['ROWS']['P3'] if case=='row' else 24800
    path.write_text(json.dumps(record))
    assert not aggregate(screen)['adopted'] and screen[1].read_text()==PIN


def test_pair_refusal_never_launches_new_workers(tmp_path):
    for directory in ('source','scripts','logs','results/prefill-fa3-pair'): (tmp_path/directory).mkdir(parents=True,exist_ok=True)
    (tmp_path/'source/prefill-kernels.env').write_text(PIN)
    (tmp_path/'results/prefill-fa3-pair/selection.json').write_text(json.dumps({'schema':'firmus-prefill-fa3-pair/v1','adopted':False,'gpus':[]}))
    (tmp_path/'scripts'/SCRIPT.name).write_text(SCRIPT.read_text())
    preamble=f'''set -euo pipefail
ROOT={shlex.quote(str(tmp_path))}
ART="$ROOT/logs"
prefill_pp=4
HEAD=head
TAIL=tail
SLURM_JOB_ID=123
SGLANG=unused
jq() {{ if [[ "$*" == *adopted* ]]; then return 1; elif [[ "$*" == *sha256* ]]; then printf '%064d\\n' 0; else printf 'true\\n'; fi; }}
mounts() {{ printf 'unused\\n'; }}
timeout() {{ if [[ "$1" == --kill-after=2 ]]; then touch "$ROOT/unexpected-launch"; return 1; fi; shift 2; "$@"; }}
uv() {{ shift 5; {shlex.quote(sys.executable)} "$@"; }}
'''
    subprocess.run(['bash','-c',preamble+block()+'\nprintf continued >"$ROOT/continued"\n'],check=True,capture_output=True,text=True)
    assert not (tmp_path/'unexpected-launch').exists()
    assert (tmp_path/'source/prefill-kernels.env').read_text()==PIN
    assert (tmp_path/'continued').read_text()=='continued'


@pytest.mark.parametrize('case',['baseline','forward','reverse','pooled','bytes','numeric','shape','single','endurance'])
def test_p3_requires_every_cell_against_p(screen,case):
    enable_p3(screen)
    path=screen[0]/'prefill-2/rank-3.json'; record=json.loads(path.read_text())
    candidate=record['candidates']['P3']; shape=candidate['shapes']['4096+3901']
    if case=='baseline': shape['baseline']='C'
    elif case in ('forward','reverse'):
        shape['samples_us'][case]['candidate']=[895.01]*10
    elif case=='pooled':
        # Each order saves 5, but pooling changes which middle samples are selected.
        shape['samples_us']={'forward':{'control':[100.]*6+[200.]*4,'candidate':[95.]*6+[199.]*4},'reverse':{'control':[200.]*6+[100.]*4,'candidate':[195.]*6+[100.]*4}}
        summary=GATE['timing_summary'](shape['samples_us'])['savings_us']
        assert summary['forward']==summary['reverse']==5 and summary['pooled']<5
    elif case=='bytes': shape['p_bytes_equal']=False
    elif case=='numeric': shape['baseline_fp32'][0]['max_scaled_error']=.251
    elif case=='shape': candidate['shapes'].pop('4096+3901')
    elif case=='single': candidate['singles']['3584']['attention_bytes_equal']=False
    elif case=='endurance': candidate['endurance']['bytes_equal']=False
    path.write_text(json.dumps(record))
    result=aggregate(screen)
    assert result['adopted'] and result['selected']=='P' and 'P3' in result['candidate_rejections']


def test_p3_samples_really_use_p_and_disabled_gate_ignores_p3(screen):
    samples={o:{'C':[1000.]*10,'P':[900.]*10,'P3':[896.]*10} for o in ('forward','reverse')}
    paired=GATE['paired'](samples,'P3')
    assert all(v==4 for v in GATE['timing_summary'](paired)['savings_us'].values())
    enable_p3(screen)
    for path in screen[0].glob('prefill-*/rank-*.json'):
        record=json.loads(path.read_text()); record['p3_enabled']=False
        path.write_text(json.dumps(record))
    assert aggregate(screen,p3_enabled=False)['selected']=='P'


@pytest.mark.parametrize('replay',[0,1,2])
@pytest.mark.parametrize('failure',[None,'attention','scale','o'])
def test_real_validation_compares_p3_bytes_on_original_and_changed_replays(monkeypatch,replay,failure):
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    validate=GATE['validate']
    common={'packed':torch.zeros(4),'cu':None}
    common.update({k:torch.zeros((2,64,128),dtype=torch.uint8) for k in ('q','k','v','kc','vc')})
    modes=('C','P','P3'); counts={k:0 for k in modes}
    state={'writers':{'C':lambda:None},'pack':lambda:None,'image':torch.zeros(4,dtype=torch.uint8),'output':{k:(torch.zeros((2,64,128),dtype=torch.uint8),torch.ones((2,64))) for k in modes},'o_outputs':{k:torch.zeros(4,dtype=torch.uint8) for k in modes}}
    def call(mode):
        n=counts[mode]; counts[mode]+=1
        out,scale=state['output'][mode]; out.fill_(n+1); scale.fill_(1); state['o_outputs'][mode].fill_(n+2)
        if mode=='P3' and n==replay and failure:
            target={'attention':out,'scale':scale,'o':state['o_outputs'][mode]}[failure]
            target.reshape(-1)[0]+=1
    state['composite']={mode:lambda mode=mode:call(mode) for mode in modes}
    monkeypatch.setitem(validate.__globals__,'host_image',lambda *args:torch.zeros(4,dtype=torch.uint8))
    monkeypatch.setitem(validate.__globals__,'native_producer',lambda *args:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.ops.apac_fa3_verify,'prefill_dump_native',lambda v,dump,cu,n:dump.zero_(),raising=False)
    checks={'prefill_selection_capture':lambda fn:SimpleNamespace(replay=fn),'prefill_fp32_reference':lambda common:(torch.tensor([0]),torch.zeros((1,64,128))),'prefill_assert_errors':lambda *args:{'relative_l2':.01,'max_scaled_error':.02}}
    result=validate(torch,checks,common,state,('P3',))['P3']
    assert result['p_bytes_equal'] is (failure is None)
    assert ('error' in result) is (failure is not None)
    if failure: assert result['error']=='P3 output differs from P'
    else: assert counts=={'C':3,'P':3,'P3':3}


def test_removed_diagnostics_and_new_build_input():
    source=SCRIPT.read_text()
    for name in ('def diagnostics(', 'writer_only','fa3_only','interaction_us','only_o','diagnostic_numerics'):
        assert name not in source
    assert set(GATE['VARIANTS'])=={'C','P','P3'}
    build=(ROOT/'kernels/fa3-verify/build.py').read_text()
    assert build.index('str(root / "prefill-vsum.patch")')<build.index('str(root / "prefill-v3.patch")')
    assert ' + 150' in block() and ' + 155' in block() and '--p3 ' in block()
    assert GATE['SECONDS']==150


@pytest.fixture
def round15_screen(tmp_path,monkeypatch):
    """Replay run 41273087893 attempt 2 without changing its P measurements.

    The gzip contains the parsed pair selection and all twelve raw prepack
    fragments from pair-41273087893/ and prepack-41273087893/. Round 15 wrote
    v2 for P/P2; schema and disabled P3 flag are adapted to this branch.
    UUIDs, timings, hashes, run IDs and numerical evidence stay raw.
    """
    with gzip.open(ROOT/'qwen-experiment-metrics/tests/fixtures/prefill-fa3-prepack-41273087893.json.gz','rt') as stream:
        data=json.load(stream)
    pair=tmp_path/'pair.json'; pair.write_text(json.dumps(data['pair']))
    env=tmp_path/'prefill.env'; env.write_text(PIN)
    for record in data['prepack']:
        assert record['schema']=='firmus-prefill-fa3-prepack/v2'
        record['schema']=GATE['SCHEMA']
        record['p3_enabled']=False
        directory=tmp_path/record['worker']; directory.mkdir(exist_ok=True)
        (directory/f"rank-{record['rank']}.json").write_text(json.dumps(record))
    # Historical group started at 1793511659, with a 155-second r15 budget.
    deadline=1793511814
    monkeypatch.setattr(time,'time',lambda:1793511731.)
    return tmp_path,env,pair,deadline


def aggregate_round15(screen):
    directory,env,pair,deadline=screen
    return GATE['aggregate'](directory,env,pair,'28027-1793511659',deadline,True,'1166ee307946ce6b6f35b90f2ec42e3f61f4201e9f83b54a3c1ac8853f095871')


def test_round15_swapped_workers_adopt_p(round15_screen):
    pair=json.loads(round15_screen[2].read_text())
    by_uuid={r['uuid']:r for r in pair['gpus']}
    for worker in range(3):
        for rank in range(4):
            record=json.loads((round15_screen[0]/f'prefill-{worker}/rank-{rank}.json').read_text())
            assert by_uuid[record['uuid']]['worker']==f'prefill-{1-worker if worker<2 else 2}'
    result=aggregate_round15(round15_screen)
    assert result['adopted'] and result['selected']=='P', result['reason']
    assert len(result['gpus'])==12 and {r['uuid'] for r in result['gpus']}==set(by_uuid)
    assert round15_screen[1].read_text()==PIN.replace(GATE['CONTROL'],GATE['DISPATCH']['P'])
    assert not list(round15_screen[0].glob('*.tmp'))


@pytest.mark.parametrize('case,reason',[
    ('foreign','wrong GPU'),('duplicate','wrong GPU'),('allocation','wrong allocation'),
    ('missing','FileNotFoundError'),('pair_duplicate','incomplete pair GPUs'),
    ('pair_missing','incomplete pair GPUs'),('pair_extra','incomplete pair GPUs'),
    ('worker','wrong worker'),('rank','wrong worker'),('device','wrong GPU'),
    ('raw_schema','stale result'),
])
def test_round15_invalid_gpu_evidence_keeps_env(round15_screen,case,reason):
    directory,env,pair_path,_=round15_screen
    path=directory/'prefill-0/rank-0.json'
    record=json.loads(path.read_text())
    pair=json.loads(pair_path.read_text())
    if case=='missing': path.unlink()
    elif case.startswith('pair_'):
        if case=='pair_duplicate': pair['gpus'][-1]['uuid']=pair['gpus'][0]['uuid']
        elif case=='pair_missing': pair['gpus'].pop()
        else: pair['gpus'].append(copy.deepcopy(pair['gpus'][0]))
        pair_path.write_text(json.dumps(pair))
    elif case=='allocation':
        # Mutate the matching UUID, which belonged to prefill-1 in the pair gate.
        next(r for r in pair['gpus'] if r['uuid']==record['uuid'])['run_id']='999-pair'
        pair_path.write_text(json.dumps(pair))
    else:
        if case=='foreign': record['uuid']='foreign'
        elif case=='duplicate': record['uuid']=json.loads((directory/'prefill-0/rank-1.json').read_text())['uuid']
        elif case=='worker': record['worker']='prefill-1'
        elif case=='rank': record['rank']=1
        elif case=='device': record['device']='NVIDIA H100'
        elif case=='raw_schema': record['schema']='firmus-prefill-fa3-prepack/v2'
        path.write_text(json.dumps(record))
    result=aggregate_round15(round15_screen)
    assert not result['adopted'] and result['selected'] is None
    assert reason in result['reason'], result['reason']
    assert env.read_text()==PIN and not list(directory.glob('*.tmp'))


def test_gpu_rank_permutation_and_pair_order_do_not_affect_adoption(screen):
    for worker in range(3):
        records=[json.loads((screen[0]/f'prefill-{worker}/rank-{rank}.json').read_text()) for rank in range(4)]
        for rank,record in enumerate(reversed(records)):
            record['rank']=rank
            (screen[0]/f'prefill-{worker}/rank-{rank}.json').write_text(json.dumps(record))
    pair=json.loads(screen[2].read_text()); pair['gpus'].reverse(); screen[2].write_text(json.dumps(pair))
    assert aggregate(screen)['adopted']



@pytest.mark.parametrize('case', ['mismatch', 'missing', 'old_schema'])
def test_p3_configuration_and_schema_must_match(screen, case):
    enable_p3(screen)
    path=screen[0]/'prefill-0/rank-0.json'
    record=json.loads(path.read_text())
    if case=='mismatch': record['p3_enabled']=False
    elif case=='missing': record.pop('p3_enabled')
    else: record['schema']='firmus-prefill-fa3-prepack/v2'
    path.write_text(json.dumps(record))
    assert not aggregate(screen)['adopted'] and screen[1].read_text()==PIN


def test_p3_requires_prepack_gate(tmp_path):
    config=json.loads((ROOT/'configs/firmus-prefill-pp4.json').read_text())
    config['prefill'].update(fa3_v3_gate=True,fa3_prepack_gate=False)
    for name in ('source','results'): (tmp_path/name).mkdir()
    with pytest.raises(AssertionError,match='V3 gate requires prepack gate'):
        runpy.run_path(str(ROOT/'scripts/prepare-firmus-pp4-settings.py'))['prepare'](config,ROOT,tmp_path)
