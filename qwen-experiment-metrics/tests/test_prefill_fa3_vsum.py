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


@pytest.mark.parametrize('lengths,n', [((3994,3994),96),((3900,4096),96),((3994,),192),((3584,),192),((4096,),192),((3583,),0),((3899,4096),0),((4097,),0),((3994,)*3,0),(None,0)])
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


@pytest.mark.parametrize('case',['missing','malformed','stale','uuid','allocation','late','elapsed','failed','shape','dump','producer','replay','numeric','reverse','incomplete','nan','overflow','endurance','bytes','dispatch','pair_failed','pair_uuid','timeout'])
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
        elif case=='shape': record['shapes'].pop('3584')
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
    assert config['prefill']['fa3_vsum_gate'] is False
    assert config['sglang_overlay_patches'][-1]=='patches/sglang/prefill-fa3-vsum.patch'
    assert config['images']['sglang']['sha256']==config['container_sha256'] and config['container_sha256'].startswith('e7b09add')
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
    assert [(row['native_v_n'],row['variant']) for row in table if row.get('native_v_n')]==[(192,49376),(96,57568)]


def test_deployed_offset_matches_exhaustive_cute_maps(tmp_path):
    import hashlib
    import struct
    patch=(ROOT/'patches/sglang/prefill-fa3-vsum.patch').read_text()
    added='\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    start='template<int N> __device__ __forceinline__ int native_offset'
    function=start+added.split(start,1)[1].split('return out; }',1)[0]+'return out; }'
    function=function.replace('__device__ __forceinline__ ','')
    source=tmp_path/'map.cpp'
    source.write_text('#include <cstdio>\n'+function+'\nint main(){for(int x=0;x<96*128;++x)std::printf("%d\\n",native_offset<96>(x/128,x%128)); for(int x=0;x<192*128;++x)std::printf("%d\\n",native_offset<192>(x/128,x%128));}\n')
    binary=tmp_path/'map'
    subprocess.run(['g++','-O2','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True)
    values=[int(x) for x in subprocess.check_output([str(binary)],text=True).splitlines()]
    proof=json.loads((ROOT/'research/runtime-performance/fa3-prefill-vsum-cpu-validation-20261101.json').read_text())
    for n,offset in ((96,0),(192,96*128)):
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
