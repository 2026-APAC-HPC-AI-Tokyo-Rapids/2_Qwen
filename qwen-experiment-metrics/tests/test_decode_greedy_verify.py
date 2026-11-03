"""Actual shared CUDA math versus independent NumPy control, plus bounded gate."""
import ast
import copy
import ctypes
import json
import logging
import os
import runpy
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'scripts/select-firmus-decode-greedy.py'
GATE=runpy.run_path(str(SCRIPT))
PATCH=ROOT/'patches/sglang/decode-greedy-verify.patch'


def module_source(tmp_path):
    subprocess.run(['git','apply','--include=python/sglang/srt/speculative/greedy_verify.py',str(PATCH)],cwd=tmp_path,check=True)
    return (tmp_path/'python/sglang/srt/speculative/greedy_verify.py').read_text()


@pytest.fixture
def dispatch(tmp_path,monkeypatch):
    tree=ast.parse(module_source(tmp_path))
    tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.Assign))]
    scope={'__name__':__name__,'logging':logging,'os':os,'torch':NS(bfloat16='bf16'),'envs':NS(SGLANG_ENABLE_ASYNC_ASSERT=NS(get=lambda:False))}
    exec(compile(tree,'<actual dispatch>','exec'),scope)  # noqa: S102 - execute the actual patched dispatch helpers
    monkeypatch.setenv(GATE['KEY'],'1')
    return scope


@pytest.mark.parametrize('case',['valid','off','dtype','cpu','vocab','stride','rows','batch','eagle','mode','draft','random','additive','scaling','bias','grammar','logprob','assert'])
def test_actual_dispatch(dispatch,monkeypatch,case):
    logits=NS(is_cuda=True,dtype='bf16',ndim=2,shape=(176,151936),stride=lambda i:1 if i==1 else 151936)
    info=NS(vocab_size=151936,is_all_greedy=True,acc_additive_penalties=None,acc_scaling_penalties=None,logit_bias=None)
    verify=NS(draft_token_num=4)
    batch=NS(seq_lens=[1]*44,spec_algorithm=NS(is_eagle3=lambda:True),forward_mode=NS(is_target_verify=lambda:True),return_logprob=False)
    grammar=None
    if case=='off': monkeypatch.delenv(GATE['KEY'])
    elif case=='dtype': logits.dtype='float32'
    elif case=='cpu': logits.is_cuda=False
    elif case=='vocab': logits.shape=(176,151935)
    elif case=='stride': logits.stride=lambda _:2
    elif case=='rows': logits.shape=(172,151936)
    elif case=='batch': batch.seq_lens=[1]*49
    elif case=='eagle': batch.spec_algorithm.is_eagle3=lambda:False
    elif case=='mode': batch.forward_mode.is_target_verify=lambda:False
    elif case=='draft': verify.draft_token_num=3
    elif case=='random': info.is_all_greedy=False
    elif case=='additive': info.acc_additive_penalties=1
    elif case=='scaling': info.acc_scaling_penalties=1
    elif case=='bias': info.logit_bias=1
    elif case=='grammar': grammar=1
    elif case=='logprob': batch.return_logprob=True
    elif case=='assert': dispatch['envs'].SGLANG_ENABLE_ASYNC_ASSERT.get=lambda:True
    assert dispatch['eligible'](logits,info,verify,batch,grammar) is (case=='valid')


@pytest.mark.parametrize('vocab',[1,4096,4097,151936,152064])
@pytest.mark.parametrize('padding',[0,64])
def test_dispatch_model_vocab_and_padding_stride(dispatch,vocab,padding):
    logits=NS(is_cuda=True,dtype='bf16',ndim=2,shape=(4,vocab),stride=lambda i:1 if i==1 else vocab+padding)
    assert dispatch['logits_eligible'](logits,vocab)
    assert not dispatch['logits_eligible'](logits,vocab+1)


def test_fallback_and_used_logs_once_per_process(dispatch,caplog,monkeypatch):
    logits=NS(is_cuda=True,dtype='bf16',ndim=2,shape=(4,151936),stride=lambda i:1 if i==1 else 152064)
    with caplog.at_level(logging.INFO):
        assert dispatch['logits_eligible'](logits,151936)
        assert not dispatch['logits_eligible'](logits,152064)
        assert not dispatch['logits_eligible'](logits,151936,1.0)
        calls=[]
        dispatch['torch'].ops=NS(apac_fa3_verify=NS(greedy_argmax=lambda *a:calls.append(a)))
        workspace=dispatch['GreedyWorkspace'].__new__(dispatch['GreedyWorkspace'])
        workspace.state=(None,None,None); workspace.partial=None; workspace.output='tokens'
        workspace.penalty=False; workspace.sanitize=False
        assert workspace.run(logits)=='tokens'
        assert workspace.run(logits)=='tokens'
    assert len(calls)==2
    assert len([r for r in caplog.records if 'fallback:' in r.message])==1
    assert '151936' in caplog.records[0].message and '152064' in caplog.records[0].message
    assert len([r for r in caplog.records if 'using fused kernels' in r.message])==1
    assert 'vocab=151936 stride=152064' in caplog.records[-1].message
    caplog.clear(); dispatch['_fallback_logged']=False
    monkeypatch.setenv(GATE['KEY'],'0')
    assert not dispatch['logits_eligible'](logits,152064)
    assert not caplog.records


@pytest.mark.parametrize('reason',['dtype','stride','softcap'])
def test_logits_fallback_reason(dispatch,caplog,reason):
    logits=NS(is_cuda=True,dtype='bf16',ndim=2,shape=(4,151936),stride=lambda i:1 if i==1 else 151936)
    if reason=='dtype': logits.dtype='fp32'
    if reason=='stride': logits.stride=lambda _:2
    assert not dispatch['logits_eligible'](logits,151936,1 if reason=='softcap' else None)
    assert {'dtype':'dtype','stride':'stride','softcap':'softcapping'}[reason] in caplog.text


@pytest.fixture(scope='module')
def cpp_math(tmp_path_factory):
    directory=tmp_path_factory.mktemp('greedy-math')
    source=directory/'math.cpp'; library=directory/'math.so'
    source.write_text('''#include <climits>
#include "greedy_verify_math.cuh"
extern "C" void run(const unsigned short* x, int rows, int vocab, int stride, const long* stops, int ns, const int* minimum, const int* lengths, bool penalty, bool sanitize, long* out) {
    for(int r=0;r<rows;++r) {
        GreedyPair best{-INFINITY,INT_MAX};
        // Same per-thread 16 values, 4096-column tiles, then a two-pass reduction.
        for(int tile=(vocab+4095)/4096-1;tile>=0;--tile) {
            GreedyPair chunk{-INFINITY,INT_MAX};
            for(int lane=255;lane>=0;--lane) for(int k=0;k<16;++k) {
                int i=tile*4096+k*256+lane;
                if(i>=vocab) continue;
                unsigned bits=unsigned(x[r*stride+i])<<16;
                float v; __builtin_memcpy(&v,&bits,4);
                bool stop=false;
                if(penalty && lengths[r/4]<minimum[r/4]) for(int s=0;s<ns;++s) stop |= stops[(r/4)*ns+s]==i;
                chunk=greedy_better(chunk,{greedy_value(v,sanitize,penalty,stop),i});
            }
            best=greedy_better(best,chunk);
        }
        out[r]=best.index;
    }
}
''')
    subprocess.run(['g++','-std=c++17','-shared','-fPIC','-O2','-fno-fast-math','-I'+str(ROOT/'kernels/fa3-verify'),str(source),'-o',str(library)],check=True)
    f=ctypes.CDLL(str(library)).run
    f.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_bool,ctypes.c_bool,ctypes.c_void_p]
    def run(x,stops,minimum,lengths,penalty,sanitize):
        out=np.empty(x.shape[0],np.int64)
        f(x.ctypes.data,*x.shape,x.strides[0]//2,stops.ctypes.data,stops.shape[1],minimum.ctypes.data,lengths.ctypes.data,penalty,sanitize,out.ctypes.data)
        return out
    return run


def numpy_control(x,stops,minimum,lengths,penalty,sanitize):
    # Materialized FP32 control: sanitize, zero penalty + where, add, argmax.
    y=(x.astype(np.uint32)<<16).view(np.float32)
    if sanitize: np.nan_to_num(y,copy=False,nan=-1e30,posinf=1e30,neginf=-1e30)
    if penalty:
        additive=np.zeros((len(minimum),x.shape[1]+1),np.float32)
        for r in range(len(minimum)): additive[r,stops[r]]=-np.inf
        additive=np.where((lengths<minimum)[:,None],additive[:,:x.shape[1]],np.float32(0))
        with np.errstate(invalid='ignore'): y+=np.repeat(additive,4,axis=0)
    return np.argmax(y,axis=1)


def verify_chain(tokens,candidates):
    b=len(tokens)//4; indices=np.full((b,4),-1,np.int32); length=np.ones(b,np.int32); bonus=np.zeros(b,np.int64)
    for r in range(b):
        indices[r,0]=r*4
        for k in range(3):
            if candidates[r,k+1]!=tokens[r*4+k]: break
            indices[r,k+1]=r*4+k+1; length[r]+=1
        bonus[r]=tokens[r*4+length[r]-1]
    return indices,length,bonus


@pytest.mark.parametrize('vocab',[151936,152064])
@pytest.mark.parametrize('b',range(1,49))
@pytest.mark.parametrize('padding',[False,True])
@pytest.mark.parametrize('sanitize',[False,True])
def test_shared_cuda_math_full_vocab_bit_contract(cpp_math,vocab,b,padding,sanitize):
    storage=np.zeros((4*b,vocab+64*padding),np.uint16); x=storage[:,:vocab]
    storage[:,vocab:]=0x7f80  # Poison padding with +Inf; it must be excluded.
    # BF16 values, including subnormals and NaN payload/sign positions.
    x[::4,0]=0x7fc1; x[::4,-1]=0xffc3
    x[1::4,vocab-1]=0x7f80; x[1::4,4096]=0xff80
    x[2::4,0]=0x8000
    x[3::4,4095]=0x4100; x[3::4,4096]=0x4100; x[3::4,-1]=0x4100
    x[3::4,1]=0x0001
    stops=np.tile(np.array([0,vocab-2,vocab-1,vocab],np.int64),(b,1))
    minimum=np.full(b,2,np.int32); lengths=np.full(b,1,np.int32)
    initial=None
    for state in ('initial','boundary','changed','restored','no_penalty'):
        lengths.fill(1 if state in ('initial','restored') else 2 if state=='boundary' else 3)
        minimum.fill(2)
        if state=='changed': minimum[::2]=4
        penalty=state!='no_penalty'
        expected=numpy_control(x,stops,minimum,lengths,penalty,sanitize)
        actual=cpp_math(x,stops,minimum,lengths,penalty,sanitize)
        np.testing.assert_array_equal(actual,expected)
        # Same target tokens feed the unchanged official verify; cover 0..3 accepted drafts.
        candidates=np.zeros((b,4),np.int64)
        for r in range(b):
            depth=r%4
            candidates[r,1:]=expected.reshape(b,4)[r,:3]
            if depth<3: candidates[r,depth+1]=-1
        for left,right in zip(verify_chain(actual,candidates),verify_chain(expected,candidates)): np.testing.assert_array_equal(left,right)
        if state=='initial': initial=actual.copy()
        if state=='restored': np.testing.assert_array_equal(actual,initial)


@pytest.mark.parametrize('vocab',[1,384,4095,4096,4097,151936,152064])
@pytest.mark.parametrize('sanitize',[False,True])
@pytest.mark.parametrize('penalty',[False,True])
def test_shared_math_subnormal_all_inf_all_nan(cpp_math,vocab,sanitize,penalty):
    x=np.zeros((4,vocab),np.uint16)
    x[0].fill(0xff80); x[1,min(101,vocab-1)]=0x0001
    x[2,0]=0x8000; x[3].fill(0x7fc3)
    stops=np.array([[vocab]],np.int64); minimum=np.array([2],np.int32); lengths=np.array([1],np.int32)
    np.testing.assert_array_equal(cpp_math(x,stops,minimum,lengths,penalty,sanitize),numpy_control(x,stops,minimum,lengths,penalty,sanitize))


@pytest.mark.parametrize('vocab',[151936,152064])
@pytest.mark.parametrize('padding',[0,128])
@pytest.mark.parametrize('sanitize',[False,True])
@pytest.mark.parametrize('penalty',[False,True])
def test_shared_math_winning_signed_subnormals_zeros_and_tile_ties(cpp_math,vocab,padding,sanitize,penalty):
    storage=np.full((8,vocab+padding),0x7f80,np.uint16)
    x=storage[:,:vocab]; x.fill(0xbf80)
    # Every special value determines the winner, rather than hiding below a
    # larger finite value. Tie positions cross both warp and tile boundaries.
    x[0].fill(0); x[0,4096]=0x0001
    x[1].fill(0x8002); x[1,4096]=0x8001
    x[2].fill(0x8000); x[2,vocab-1]=0
    x[3].fill(0); x[3,0]=0x8000
    x[4,[4095,4096,vocab-1]]=0x4100
    x[5,[255,256,vocab-1]]=0x4100
    x[6].fill(0xff80)
    x[7].fill(0x7fc3)
    stops=np.full((2,1),vocab,np.int64)
    minimum=np.full(2,2,np.int32); lengths=np.ones(2,np.int32)
    expected=numpy_control(x,stops,minimum,lengths,penalty,sanitize)
    np.testing.assert_array_equal(expected,[4096,4096,0,0,4095,255,0,0])
    np.testing.assert_array_equal(cpp_math(x,stops,minimum,lengths,penalty,sanitize),expected)


@pytest.mark.parametrize('vocab',[1,384,4095,4096,4097,151936,152064])
@pytest.mark.parametrize('padding',[0,128])
def test_last_vocab_stop_and_minimum_boundary(cpp_math,vocab,padding):
    storage=np.full((4,vocab+padding),0x7fc1,np.uint16)
    x=storage[:,:vocab]; x.fill(0xbf80)
    x[:,-1]=0x4100
    stops=np.array([[vocab-1,vocab]],np.int64)
    minimum=np.array([2],np.int32)
    for length in (1,2,3):
        lengths=np.array([length],np.int32)
        actual=cpp_math(x,stops,minimum,lengths,True,False)
        np.testing.assert_array_equal(actual,numpy_control(x,stops,minimum,lengths,True,False))
        np.testing.assert_array_equal(actual,np.full(4,0 if length<2 else vocab-1))


@pytest.mark.parametrize('vocab',[151936,152064])
def test_gate_reads_model_config_and_rejects_invalid_vocab(tmp_path,vocab):
    config=tmp_path/'config.json'; config.write_text(json.dumps({'vocab_size':vocab}))
    assert GATE['model_vocab'](tmp_path)==vocab
    for bad in (0,-1,True,'151936',2147479553):
        config.write_text(json.dumps({'vocab_size':bad}))
        with pytest.raises(AssertionError,match='invalid model vocab_size'): GATE['model_vocab'](tmp_path)


@pytest.fixture
def screen(tmp_path):
    now=time.time(); directory=tmp_path/'gate'; directory.mkdir()
    (directory/'inventory.json').write_text(json.dumps({'run_id':'123-456','host':'g4','gpus':{str(i+4):f'uuid-{i}' for i in range(4)}}))
    block={o:{'control':[.3]*GATE['ROUNDS'],'candidate':[.2]*GATE['ROUNDS']} for o in ('forward','reverse')}
    for rank in range(4):
        record={'schema':GATE['SCHEMA'],'run_id':'123-456','rank':rank,'vocab_size':151936,'uuid':f'uuid-{rank}','device':'NVIDIA H200','image_sha256':'a'*64,'binary_sha256':'b'*64,'passed':True,'elapsed_seconds':50,'finished_at':now,'replays':10000,'endurance_equal':True,'sanitize_modes':[False,True],'cases':{k:{'states':list(GATE['STATES']),'equal':True,'no_penalty_equal':True} for k in GATE['expected_cases']()},'timings_ms':{f'{b}:{p}':[copy.deepcopy(block) for _ in range(3)] for b in (44,46,48) for p in (0,1)}}
        (directory/f'rank-{rank}.json').write_text(json.dumps(record))
    env=tmp_path/'decode.env'; env.write_text('OTHER=keep\n'+GATE['KEY']+'=0\n')
    return directory,env,now+90


def aggregate(screen,workers_ok=True,deadline=None):
    return GATE['aggregate'](screen[0],screen[1],'123-456','a'*64,screen[2] if deadline is None else deadline,workers_ok,151936)


def test_gate_atomic_all_four(screen):
    assert aggregate(screen)['adopted']
    assert screen[1].read_text()=='OTHER=keep\n'+GATE['KEY']+'=1\n'
    assert not list(screen[1].parent.glob('*.tmp'))


@pytest.mark.parametrize('failure',['missing','stale','vocab','uuid','rank','duplicate','image','binary','late','elapsed','passed','case','bit','state','replays','endurance','sanitize','shape','block','order','margin','nan','timeout','workers','inventory','host'])
def test_gate_refusal_keeps_control(screen,failure):
    path=screen[0]/'rank-3.json'; r=json.loads(path.read_text()); original=screen[1].read_bytes()
    if failure=='missing': path.unlink()
    elif failure=='stale': r['run_id']='old'
    elif failure=='vocab': r['vocab_size']=152064
    elif failure in ('uuid','duplicate'): r['uuid']='foreign' if failure=='uuid' else 'uuid-0'
    elif failure=='rank': r['rank']=2
    elif failure in ('image','binary'): r[f'{failure}_sha256']='c'*64
    elif failure=='late': r['finished_at']=screen[2]+1
    elif failure=='elapsed': r['elapsed_seconds']=85
    elif failure=='passed': r['passed']=False
    elif failure=='case': r['cases'].pop('1:0')
    elif failure=='bit': r['cases']['1:0']['equal']=False
    elif failure=='state': r['cases']['1:0']['states'].pop()
    elif failure=='replays': r['replays']=9999
    elif failure=='endurance': r['endurance_equal']=False
    elif failure=='sanitize': r['sanitize_modes']=[False]
    elif failure=='shape': r['timings_ms'].pop('44:0')
    elif failure=='block': r['timings_ms']['44:0'].pop()
    elif failure=='order': r['timings_ms']['44:0'][0].pop('reverse')
    elif failure=='margin': r['timings_ms']['48:1'][2]['reverse']['candidate']=[.251]*5
    elif failure=='nan': r['timings_ms']['48:1'][2]['forward']['control']=[float('nan')]*5
    elif failure in ('inventory','host'):
        p=screen[0]/'inventory.json'; data=json.loads(p.read_text())
        if failure=='inventory': data['gpus']['4']='foreign'
        else: data['host']='g1'
        p.write_text(json.dumps(data))
    if failure!='missing': path.write_text(json.dumps(r))
    assert not aggregate(screen,workers_ok=failure!='workers',deadline=0 if failure=='timeout' else None)['adopted']
    assert screen[1].read_bytes()==original
    assert not list(screen[1].parent.glob('*.tmp'))


def shell_block():
    s=(ROOT/'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    return s[s.index('# Decode greedy gate begin'):s.index('# Decode greedy gate end.')]


@pytest.mark.parametrize('status',[1,124,137])
def test_shell_worker_failure_timeout_serves_control(tmp_path,status):
    for d in ('source','results','logs','scripts'): (tmp_path/d).mkdir()
    (tmp_path/'source/decode-kernels.env').write_text(GATE['KEY']+'=0\n')
    (tmp_path/'scripts'/SCRIPT.name).write_text(SCRIPT.read_text())
    (tmp_path/'model').mkdir(); (tmp_path/'model/config.json').write_text('{"vocab_size":151936}')
    preamble=f'''set -euo pipefail
ROOT={shlex.quote(str(tmp_path))}
ART="$ROOT/logs"
prefill_pp=4
TAIL=g4
MODEL="$ROOT/model"
SGLANG=unused
SLURM_JOB_ID=123
jq() {{ if [[ "$*" == *sha256* ]]; then printf '%064d\\n' 0; else printf true; fi; }}
mounts() {{ printf unused; }}
srun() {{ return {status}; }}
timeout() {{ shift 2; "$@"; }}
uv() {{ shift 5; {shlex.quote(sys.executable)} "$@"; }}
'''
    subprocess.run(['bash','-c',preamble+shell_block()+'\nprintf serving >"$ROOT/continued"\n'],check=True,capture_output=True,text=True)
    assert (tmp_path/'continued').read_text()=='serving'
    assert (tmp_path/'source/decode-kernels.env').read_text()==GATE['KEY']+'=0\n'
    assert not json.loads((tmp_path/'results/decode-greedy/selection.json').read_text())['adopted']


def test_wiring_build_and_time_contract():
    config=json.loads((ROOT/'configs/firmus-prefill-pp4.json').read_text())
    assert config['decode_greedy_verify_gate'] is True and config['prefill']['fa3_v3_gate'] is True
    assert 'patches/sglang/decode-greedy-verify.patch' in config['sglang_overlay_patches']
    shell=(ROOT/'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    assert shell.index('# Decode greedy gate begin')>shell.index('prefill-fa3-prepack/selection.json')
    assert shell.index('# Decode greedy gate end.')<shell.index('worker_pids+=("$!")',shell.index('# Decode greedy gate begin'))
    assert '--gres=gpu:8' in shell_block() and '-w "${TAIL}"' in shell_block()
    assert '85 srun' in shell_block() and '90 - $(date +%s)' in shell_block()
    assert GATE['WALL_SECONDS']>=1.5*60 and GATE['WALL_SECONDS']<=90
    assert '90 + 660' in shell_block()
    assert 'select-firmus-decode-greedy.py' in (ROOT/'scripts/provision-firmus-framework.sh').read_text()
    build=(ROOT/'kernels/fa3-verify/build.py').read_text()
    assert '"greedy_verify")' in build and '"--ftz=false", "--fmad=false"' in build
    assert 'f != "--use_fast_math"' in build
    assert 'kernels/fa3-verify' in (ROOT/'scripts/build-sglang-container.sh').read_text()
    patch=PATCH.read_text()
    assert 'is_target_verify()' in patch and 'torch.bfloat16' in patch
    assert 'p.len_output_tokens.clone()' in patch and 'self.greedy_stop_ids = self.greedy_stop_ids[keep_indices]' in patch
    assert 'ignore_eos' not in patch  # exact original stop union is reused
    # Full vocabulary, two passes, shared comparator and exact FP32 addition.
    source=(ROOT/'kernels/fa3-verify/greedy_verify.cu').read_text()
    assert 'k<16' in source and 'chunk*4096+k*256+threadIdx.x' in source
    assert 'greedy_verify_math.cuh' in source and 'greedy_finish<<<rows,256' in source


def test_ordered_overlay_applies_and_preserves_gemm_and_verify(tmp_path):
    import gzip
    with gzip.open(ROOT/'qwen-experiment-metrics/tests/fixtures/decode-greedy-control-source.json.gz','rt') as f: fixture=json.load(f)
    assert fixture['sglang_commit']=='fc275674a97d78553b22597ee72c8fd48f5420f4'
    for name,source in fixture['files'].items():
        p=tmp_path/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(source)
    subprocess.run(['git','apply','--check',str(PATCH)],cwd=tmp_path,check=True)
    subprocess.run(['git','apply',str(PATCH)],cwd=tmp_path,check=True)
    def function(source,name):
        tree=ast.parse(source)
        return ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name))
    for name,source in fixture['files'].items():
        updated=(tmp_path/name).read_text(); ast.parse(updated)
        if name.endswith('logits_processor.py'): assert function(source,'_compute_lm_head')==function(updated,'_compute_lm_head')
        if name.endswith('eagle_utils.py'):
            assert function(source,'verify_tree_greedy_func')==function(updated,'verify_tree_greedy_func')
            assert 'fused_target if fused_greedy else torch.argmax' in updated
    graph=(tmp_path/'python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py').read_text()
    assert 'next_token_logits_buffer=(torch.zeros' in graph  # buffers own storage for graph lifetime


@pytest.mark.parametrize('stage',['workers','aggregate'])
def test_shell_timeout_after_staged_update_never_enables(tmp_path,stage):
    for d in ('source','results','logs','scripts'): (tmp_path/d).mkdir()
    (tmp_path/'source/decode-kernels.env').write_text(GATE['KEY']+'=0\n')
    preamble=f'''set -euo pipefail
ROOT={shlex.quote(str(tmp_path))}
ART="$ROOT/logs"
prefill_pp=4
TAIL=g4
MODEL="$ROOT/model"
SGLANG=unused
SLURM_JOB_ID=123
jq() {{ if [[ "$*" == *sha256* ]]; then printf '%064d\\n' 0; else printf true; fi; }}
mounts() {{ printf unused; }}
timeout() {{
  if [[ "$1" == --kill-after=2 ]]; then return {124 if stage=='workers' else 0}; fi
  printf '{GATE['KEY']}=1\\n' >"$ROOT/source/decode-kernels.greedy.env"
  printf '{{"adopted":true}}\\n' >"$ROOT/results/decode-greedy/selection.json"
  return 124
}}
'''
    subprocess.run(['bash','-c',preamble+shell_block()+'\nprintf serving >"$ROOT/continued"\n'],check=True,capture_output=True,text=True)
    assert (tmp_path/'continued').read_text()=='serving'
    assert (tmp_path/'source/decode-kernels.env').read_text()==GATE['KEY']+'=0\n'
    assert not (tmp_path/'source/decode-kernels.greedy.env').exists()


def test_gate_skips_when_serving_budget_is_low(tmp_path):
    for d in ('source','results','logs'): (tmp_path/d).mkdir()
    (tmp_path/'source/decode-kernels.env').write_text(GATE['KEY']+'=0\n')
    script=f'''set -euo pipefail
ROOT={shlex.quote(str(tmp_path))}
prefill_pp=4
SECONDS=271
jq() {{ printf true; }}
timeout() {{ exit 99; }}
'''
    subprocess.run(['bash','-c',script+shell_block()],check=True)
    assert json.loads((tmp_path/'results/decode-greedy/selection.json').read_text())['reason']=='insufficient remaining serving budget'
    assert (tmp_path/'source/decode-kernels.env').read_text()==GATE['KEY']+'=0\n'


class FakeCuda:
    """Record capture/prepare/replay and defer device ops until graph replay."""
    def __init__(self):
        self.active=None
        self.events=[]
        self.graphs=[]

    def enqueue(self, operation):
        if self.active is None: operation()
        else: self.active.operations.append(operation)

    def CUDAGraph(self):
        cuda=self
        class Graph:
            def __init__(self):
                self.operations=[]
                self.captures=0
                self.replays=0

            def replay(self):
                cuda.events.append(('prepare',self,cuda.active))
                if cuda.active is not None:
                    raise RuntimeError('Cannot prepare for replay during capturing stage.')
                self.replays+=1
                cuda.events.append(('replay',self,None))
                for operation in self.operations: operation()
        graph=Graph()
        self.graphs.append(graph)
        return graph

    def graph(self, graph):
        cuda=self
        class Capture:
            def __enter__(self):
                assert cuda.active is None, 'nested capture'
                assert graph.captures==0, 'graph captured twice'
                graph.captures+=1
                cuda.events.append(('begin',graph,None))
                cuda.active=graph

            def __exit__(self, *args):
                cuda.events.append(('end',graph,None))
                cuda.active=None
        return Capture()

    def synchronize(self):
        assert self.active is None, 'synchronize during capture'
        self.events.append(('sync',None,None))

    def Event(self, *, enable_timing):
        assert enable_timing
        cuda=self
        class Event:
            def record(self):
                assert cuda.active is None

            def synchronize(self):
                cuda.synchronize()

            def elapsed_time(self, other):
                return 1.0
        return Event()


class FakeDeviceValue:
    def __init__(self, cuda, value=0):
        self.cuda=cuda
        self.value=value

    def __ne__(self, other):
        return lambda:self.value!=other.value

    def logical_or_(self, condition):
        self.cuda.enqueue(lambda:setattr(self,'value',self.value or condition()))

    def zero_(self):
        self.cuda.enqueue(lambda:setattr(self,'value',0))

    def item(self):
        assert self.cuda.active is None
        return self.value


@pytest.fixture
def graph_case(monkeypatch):
    cuda=FakeCuda()
    comparisons=[]
    def equal(left,right):
        assert cuda.active is None
        comparisons.append((left,right))
        return left.value==right.value
    torch=NS(cuda=cuda,bool=bool,zeros=lambda *a,**kw:FakeDeviceValue(cuda),any=lambda x:x,equal=equal)
    case=GATE['Case'].__new__(GATE['Case'])
    case.torch=torch; case.graphs={}; case.outputs={}; case.state='initial'
    case.logits=np.zeros((4,4)); case.b=1; case.vocab=4
    case.penalty=NS(_is_prepared=False)
    case.greedy=NS(GreedyWorkspace=lambda *_:NS(run=lambda _:FakeDeviceValue(cuda,case.outputs['candidate'][0].value)))
    case.corrupt_on=None
    def step(mode):
        # Capture binds mode/output storage; replay reads mutable device input.
        assert os.environ[GATE['KEY']]==('1' if mode=='candidate' else '0')
        outputs=tuple(FakeDeviceValue(cuda) for _ in range(4))
        case.outputs[mode]=outputs
        def device_step():
            value=0 if case.state=='restored' else GATE['STATES'].index(case.state) if case.state in GATE['STATES'] else -1
            if mode=='candidate' and cuda.active is None and case.graphs.get(mode) is not None and case.graphs[mode].replays==case.corrupt_on: value+=100
            for out in outputs: out.value=value
        cuda.events.append(('step',mode,cuda.active))
        cuda.enqueue(device_step)
    case.step=step
    scope=GATE['Case'].endurance.__globals__
    for key,value in (('BLOCKS',1),('ROUNDS',1),('REPS',2),('REPLAYS',5)): monkeypatch.setitem(scope,key,value)
    monkeypatch.setenv(GATE['KEY'],'0')
    monkeypatch.setenv('SGLANG_SANITIZE_NAN_LOGITS','0')
    return case,cuda,comparisons


def test_graph_capture_replay_order_and_distinct_control(graph_case):
    case,cuda,comparisons=graph_case
    for sanitize in (False,True):
        case.state='initial'; case.build_graphs(sanitize)
        control,candidate=case.graphs['control'],case.graphs['candidate']
        assert control is not candidate
        assert not set(map(id,case.outputs['control'])) & set(map(id,case.outputs['candidate']))
        for state in GATE['STATES']:
            case.state=state; case.equal()
        if not sanitize: case.timing()
        # Rebuild only after equal/timing has synchronized the previous replay.
        case.state='no_penalty'; case.build_graphs(sanitize); case.equal()
    case.state='initial'; case.build_graphs(True); case.equal()
    candidate=case.graphs['candidate']; before=candidate.replays
    case.endurance()
    checker=cuda.graphs[-1]
    assert checker is not candidate and checker is not case.graphs['control']
    assert candidate.replays-before==checker.replays==5
    assert all(graph.captures==1 for graph in cuda.graphs)
    assert all(active is None for event,_,active in cuda.events if event in ('prepare','replay'))
    assert [graph for event,graph,_ in cuda.events if event=='replay'][-10:]==[candidate,checker]*5
    assert all(left is not right for left,right in comparisons)
    # A wrong control graph must fail even though both graphs have been captured.
    case.graphs['control'].operations.append(lambda:setattr(case.outputs['control'][0],'value',999))
    with pytest.raises(AssertionError,match='argmax/index/length/bonus mismatch'): case.equal()


def test_endurance_sticky_flag_detects_one_transient_mismatch(graph_case):
    case,cuda,_=graph_case
    case.build_graphs(True); case.equal()
    case.corrupt_on=case.graphs['candidate'].replays+3
    with pytest.raises(AssertionError,match='transient endurance mismatch'): case.endurance()
    # Later replays restored all outputs; only the sticky flag proves the error.
    assert all(left.value==right.value for left,right in zip(case.outputs['control'],case.outputs['candidate']))
    assert cuda.graphs[-1].replays==5


def test_fake_cuda_rejects_original_nested_endurance_replay(graph_case):
    case,_,_=graph_case
    case.build_graphs(True); case.equal()
    with pytest.raises(RuntimeError,match='Cannot prepare for replay during capturing stage'):
        GATE['capture'](case.torch,case.graphs['candidate'].replay)


def test_sbatch_records_serving_greedy_dispatch_with_the_score():
    source=(ROOT/'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    score=source.index('test -s "${ROOT}/results/score.json"')
    record=source.index('results/decode-greedy-dispatch.txt')
    assert score<record<source.index('collect-eagle-acceptance.py')
    assert "'Decode greedy verify (using fused kernels|fallback).*'" in source  # backslash-r inside an ERE bracket matches a literal r; round 25 cut the line at 'g'

