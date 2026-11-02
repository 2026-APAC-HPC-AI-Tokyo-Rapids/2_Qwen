"""Round 23: W is admitted on CPU and adopted only by its independent GPU gate."""
import ast
import copy
import json
import os
import re
import runpy
import shlex
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_prefill_fa3_prepack import dispatch as old_dispatch  # noqa: F401 - fixture
from test_prefill_fa3_prepack import screen as prepack_screen  # noqa: F401 - fixture

ROOT=Path(__file__).resolve().parents[2]
G=runpy.run_path(str(ROOT/'scripts/select-firmus-prefill-fa3-prepack.py'))
PATCH=ROOT/'patches/sglang/prefill-v-writer.patch'


@pytest.fixture
def screen(prepack_screen):  # noqa: F811 - pytest fixture dependency
    directory,env,_,deadline=prepack_screen
    prior=[]
    for path in directory.glob('prefill-*/rank-*.json'):
        r=json.loads(path.read_text()); prior.append(copy.deepcopy(r))
        p=r['candidates']['P']; w=copy.deepcopy(p)
        for name,shape in w['shapes'].items():
            shape.update(baseline='P',baseline_fp32=copy.deepcopy(shape['control_fp32']),validation_orders=['forward','reverse'])
            for key in ('qk_bytes_equal','paged_kv_bytes_equal','native_v_bytes_equal','attention_bytes_equal','scale_bytes_equal','o_bytes_equal','contiguous_v_untouched'):
                shape[key]=True
            shape['samples_us']={order:{'control':[900.]*10,'candidate':[898. if name=='3994+3994' else 899.]*10} for order in ('forward','reverse')}
        w['endurance']['producer_bytes_equal']=True
        r.update(schema=G['W_SCHEMA'],run_id='123-w',elapsed_seconds=60,source_sha256={'kernel':'c'*64},candidates={'W':w})
        path.write_text(json.dumps(r))
    prior_path=directory/'prepack.json';prior_path.write_text(json.dumps({'schema':G['SCHEMA'],'adopted':True,'selected':'P3','gpus':prior}))
    env.write_text(G['KEY']+'='+G['DISPATCH']['P3']+'\nOTHER=kept\n')
    return directory,env,prior_path,deadline


def aggregate(screen,**kwargs):
    directory,env,prior,deadline=screen
    return G['aggregate_w'](directory,env,prior,'123-w',kwargs.get('deadline',deadline),kwargs.get('workers_ok',True),'a'*64)


def edit(screen,mutate):
    p=screen[0]/'prefill-2/rank-3.json';r=json.loads(p.read_text());mutate(r);p.write_text(json.dumps(r))


def test_w_success_uses_only_measured_p_fa3(screen):
    r=aggregate(screen)
    assert r['adopted'] and r['selected']=='W' and r['previous_selection']=='P3'
    assert screen[1].read_text()==G['KEY']+'='+G['DISPATCH']['W']+'\nOTHER=kept\n'
    assert r['summary']['cells']==84 and not r['rule']['W_P3_measured']
    assert not list(screen[0].glob('*.tmp'))


@pytest.mark.parametrize('failure',['missing','malformed','stale','worker','rank','uuid','foreign_uuid','late','elapsed','fatal','image','binary','source','incomplete','replay','row','variant','baseline','reference','qk_bytes_equal','paged_kv_bytes_equal','native_v_bytes_equal','attention_bytes_equal','scale_bytes_equal','o_bytes_equal','contiguous_v_untouched','order','numeric','single','endurance','endurance_producer','timing','nan','inf','representative','regression','workers','deadline','prior_failed','prior_incomplete','prior_image','prior_allocation','prior_P_failed','env'])
def test_w_refusal_keeps_p_or_p3_byte_for_byte(screen,failure):
    before=screen[1].read_bytes();path=screen[0]/'prefill-2/rank-3.json'
    if failure=='missing': path.unlink()
    elif failure=='malformed': path.write_text('{')
    elif failure.startswith('prior_'):
        prior=json.loads(screen[2].read_text())
        if failure=='prior_failed': prior['adopted']=False
        elif failure=='prior_incomplete': prior['gpus'].pop()
        elif failure=='prior_image': prior['gpus'][0]['image_sha256']='d'*64
        elif failure=='prior_allocation': prior['gpus'][0]['run_id']='999-prepack'
        else: prior['gpus'][0]['candidates']['P']['passed']=False
        screen[2].write_text(json.dumps(prior))
    elif failure=='env':
        screen[1].write_text('other\n');before=screen[1].read_bytes()
    elif failure not in ('workers','deadline'):
        def mutate(r):
            c=r['candidates']['W'];s=c['shapes']['3994+3994']
            if failure=='stale': r['run_id']='old'
            elif failure=='worker': r['worker']='prefill-0'
            elif failure=='rank': r['rank']=0
            elif failure=='uuid': r['uuid']='gpu-0-0'
            elif failure=='foreign_uuid': r['uuid']='foreign'
            elif failure=='late': r['finished_at']=screen[3]+1
            elif failure=='elapsed': r['elapsed_seconds']=G['W_SECONDS']
            elif failure=='fatal': r['healthy']=False
            elif failure=='image': r['image_sha256']='d'*64
            elif failure=='binary': r['binary_sha256']='d'*64
            elif failure=='source': r['source_sha256']={'kernel':'d'*64}
            elif failure=='incomplete': c['shapes'].pop('3900+4096')
            elif failure=='replay': s['replays']=1
            elif failure=='row': s['row']=G['ROWS']['P3']
            elif failure=='variant': s['variant']=G['VARIANTS']['P3']
            elif failure=='baseline': s['baseline']='P3'
            elif failure=='reference': s['baseline_fp32'][0]['relative_l2']=.02
            elif failure.endswith('equal') or failure=='contiguous_v_untouched': s[failure]=False
            elif failure=='order': s['validation_orders']=['forward']
            elif failure=='numeric': s['candidate_fp32'][0]['relative_l2']=.051
            elif failure=='single': c['singles'].pop('3584')
            elif failure=='endurance': c['endurance']['calls']=1199
            elif failure=='endurance_producer': c['endurance']['producer_bytes_equal']=False
            elif failure=='timing': s['samples_us']['reverse']['control'].pop()
            elif failure in ('nan','inf'): s['samples_us']['forward']['control'][0]=float(failure)
            elif failure=='representative': s['samples_us']['reverse']['candidate']=[898.001]*10
            elif failure=='regression': c['shapes']['3900+4096']['samples_us']={o:{'control':[900.]*10,'candidate':[901.001]*10} for o in ('forward','reverse')}
        edit(screen,mutate)
    r=aggregate(screen,workers_ok=failure!='workers',deadline=time.time()-1 if failure=='deadline' else screen[3])
    assert not r['adopted'] and screen[1].read_bytes()==before
    json.dumps(r,allow_nan=False)


def test_w_uuid_set_accepts_rescheduling(screen):
    a=screen[0]/'prefill-0/rank-0.json';b=screen[0]/'prefill-2/rank-3.json'
    ra=json.loads(a.read_text());rb=json.loads(b.read_text());ra['uuid'],rb['uuid']=rb['uuid'],ra['uuid']
    a.write_text(json.dumps(ra));b.write_text(json.dumps(rb))
    assert aggregate(screen)['adopted']


@pytest.mark.parametrize('order',['forward','reverse'])
def test_w_group_requires_positive_order_median(screen,order):
    for p in screen[0].glob('prefill-*/rank-*.json'):
        r=json.loads(p.read_text())
        for name,s in r['candidates']['W']['shapes'].items():
            if name!='3994+3994': s['samples_us'][order]['candidate']=[900.]*10
        p.write_text(json.dumps(r))
    assert not aggregate(screen)['adopted']


def test_w_atomic_failure_keeps_prior(screen,monkeypatch):
    before=screen[1].read_bytes()
    def fail(*_): raise OSError('rename')
    monkeypatch.setattr(Path,'replace',fail)
    assert not aggregate(screen)['adopted'] and screen[1].read_bytes()==before
    assert not list(screen[0].glob('*.tmp'))


@pytest.fixture
def dispatch(old_dispatch,tmp_path):  # noqa: F811 - pytest fixture dependency
    subprocess.run(['git','apply','--include=python/sglang/srt/layers/attention/prefill_fp8.py',str(PATCH)],cwd=tmp_path,check=True)
    tree=ast.parse((tmp_path/'python/sglang/srt/layers/attention/prefill_fp8.py').read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('prefill_candidate','pair_candidate')]
    exec(compile(tree,'<W routing>','exec'),old_dispatch)  # noqa: S102 - actual pure-host routing
    return old_dispatch


@pytest.mark.parametrize('lengths,expected',[((3994,3994),15),((3900,4096),15),((4096,3900),15),((3994,),11),((3584,),11),((4096,),11),((3899,4096),11),((4096,4097),11),((3994,)*3,11),(None,11)])
def test_w_dispatch_cannot_select_p3(dispatch,monkeypatch,lengths,expected):
    monkeypatch.setenv(G['KEY'],G['DISPATCH']['W'])
    assert dispatch['pair_candidate'](lengths)['id']==expected
    assert dispatch['prefill_candidate']()['id']==11


def model_fragment():
    # Build the real pre-W block from its owning overlay and apply W to that file.
    patch=(ROOT/'patches/sglang/prefill-fa3-vsum.patch').read_text().split('+++ b/python/sglang/srt/models/qwen3_moe.py',1)[1]
    text='\n'.join(l[1:] for l in patch.splitlines() if l.startswith((' ','+')))
    return '                selected = pair_candidate'+text.split('                selected = pair_candidate',1)[1]


@pytest.mark.parametrize('mode',['P','P3','W'])
@pytest.mark.parametrize('lengths,fixed',[((3994,3994),True),((3900,4096),True),((3994,),True),((3899,4096),True),((3994,)*3,True),((3994,3994),False)])
def test_w_model_removes_only_w_contiguous_v_and_pack(dispatch,monkeypatch,tmp_path,mode,lengths,fixed):
    # Reconstruct only the model text needed by the actual W hunks, preserving line numbers.
    section=PATCH.read_text().split('+++ b/python/sglang/srt/models/qwen3_moe.py',1)[1].split('--- a/',1)[0]
    lines=[]
    for hunk in re.split(r'(?=^@@ )',section,flags=re.MULTILINE)[1:]:
        first,*body=hunk.splitlines();start=int(re.match(r'@@ -(\d+)',first)[1])
        lines.extend(['']*(start-1-len(lines)))
        lines.extend(l[1:] for l in body if l.startswith((' ','-')) or l=='')
    fragment_lines=model_fragment().splitlines()
    index=next(i for i,l in enumerate(lines) if l.startswith('                selected ='))
    lines[index:index+len(fragment_lines)]=fragment_lines
    subprocess.run(['git','init','-q',str(tmp_path/'model')],check=True)
    path=tmp_path/'model/python/sglang/srt/models/qwen3_moe.py';path.parent.mkdir(parents=True);path.write_text('\n'.join(lines)+'\n')
    subprocess.run(['git','apply','--include=python/sglang/srt/models/qwen3_moe.py',str(PATCH)],cwd=tmp_path/'model',check=True)
    patched=path.read_text();fragment=model_fragment()
    # Use the entire contiguous native dispatch region from the patched model;
    # fill unchanged fallback lines from the owning overlay.
    full=patched.split('                selected =',1)[1]
    full='                selected ='+full
    marker='                else:\n'
    if marker not in full:
        full=full.rstrip()+'\n'+fragment[fragment.index(marker):]
    calls=[];key=object()
    monkeypatch.setenv(G['KEY'],G['DISPATCH'][mode])
    monkeypatch.setitem(sys.modules,'sglang.srt.layers.attention.decode_fa3_verify',SimpleNamespace(load=lambda:calls.append('load')))
    monkeypatch.setitem(sys.modules,'sglang.kernels.ops.attention.rope',SimpleNamespace(apply_qknorm_rope_fp8_prefill_w=lambda *a:calls.append('W')))
    scope={**dispatch,'os':os,'self':SimpleNamespace(attn=SimpleNamespace(fixed_prefill_output=fixed)), 'forward_batch':SimpleNamespace(extend_seq_lens_cpu=lengths),'torch':SimpleNamespace(empty=lambda *a,**k:'image',empty_like=lambda *a:calls.append('V allocation'),uint8='uint8',ops=SimpleNamespace(apac_fa3_verify=SimpleNamespace(prefill_prepack=lambda *a:calls.append('pack')))), 'k':SimpleNamespace(device='cuda'),'_PREFILL_QKNORM':'fp32','rope_args':(),'kv_args':(),'contiguous_k':key,'contiguous_v':key,'get_attn_backend':lambda:SimpleNamespace(forward_metadata=SimpleNamespace(cu_seqlens_q='cu')),'apply_qknorm_rope_fp8_prefill_fp32':lambda *a:calls.append('fp32'),'apply_qknorm_rope_fp8_prefill':lambda *a:calls.append('old')}
    # Stitch sparse hunks by removing gaps; they cover the complete native branch.
    full='\n'.join(l for l in full.splitlines() if l.strip())
    exec(compile(textwrap.dedent(full),'<W model>','exec'),scope)  # noqa: S102 - real model route
    eligible=fixed and len(lengths)==2 and all(3900<=n<=4096 for n in lengths)
    assert calls==(['load','W'] if eligible and mode=='W' else (['V allocation','load','fp32','pack'] if eligible else ['V allocation','fp32']))
    if eligible and mode=='W': assert scope['contiguous_v'] is key


def shell_block():
    return '# W gate begin'+(ROOT/'platforms/firmus/framework-comparison/sglang.sbatch').read_text().split('# W gate begin',1)[1].split('# W gate end.',1)[0]


@pytest.mark.parametrize('stage',['workers','aggregate','budget','prior'])
def test_w_timeout_or_no_budget_keeps_p3_and_serves(tmp_path,stage):
    for name in ('source','scripts','logs','results/prefill-fa3-prepack'): (tmp_path/name).mkdir(parents=True,exist_ok=True)
    pin=G['KEY']+'='+G['DISPATCH']['P3']+'\n'
    (tmp_path/'source/prefill-kernels.env').write_text(pin)
    (tmp_path/'scripts/select-firmus-prefill-fa3-prepack.py').write_text((ROOT/'scripts/select-firmus-prefill-fa3-prepack.py').read_text())
    (tmp_path/'results/prefill-fa3-prepack/selection.json').write_text('{}')
    preamble=f'''set -euo pipefail
ROOT={shlex.quote(str(tmp_path))}
ART="$ROOT/logs"
prefill_pp=4
HEAD=head
TAIL=tail
SLURM_JOB_ID=123
SGLANG=unused
SECONDS={171 if stage=='budget' else 0}
jq() {{ if [[ "$*" == *sha256* ]]; then printf '%064d\\n' 0; elif [[ "$*" == *adopted* ]]; then return {1 if stage=='prior' else 0}; else printf 'true\\n'; fi; }}
mounts() {{ printf 'unused\\n'; }}
timeout() {{ if [[ "$1" == --kill-after=2 ]]; then return {0 if stage=='aggregate' else 124}; fi; {'return 124;' if stage=='aggregate' else 'shift 2; "$@";'} }}
uv() {{ shift 5; {shlex.quote(sys.executable)} "$@"; }}
'''
    subprocess.run(['bash','-c',preamble+shell_block()+'\nprintf continued >"$ROOT/continued"\n'],check=True,capture_output=True,text=True)
    assert (tmp_path/'continued').read_text()=='continued'
    assert (tmp_path/'source/prefill-kernels.env').read_text()==pin
    selection=tmp_path/'results/prefill-fa3-v-writer/selection.json'
    if stage!='aggregate': assert not json.loads(selection.read_text())['adopted']


def test_w_wiring_budget_and_default_off(tmp_path):
    c=json.loads((ROOT/'configs/firmus-prefill-pp4.json').read_text())
    assert c['prefill']['fa3_v_writer_gate'] is False
    assert c['sglang_overlay_patches'][-1]=='patches/sglang/prefill-v-writer.patch'
    assert c['container_sha256'].startswith('c7804efb')
    shell=(ROOT/'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    assert shell.index('selection.json" --env-output',shell.index('# P is revalidated'))<shell.index('# W gate begin')<shell.index('# Decode greedy gate begin')
    assert '100 + 90 + 660' in shell_block() and G['W_SECONDS']==95 and G['W_WALL']==100
    assert 793+G['W_WALL']<1020
    assert '--w true' in shell_block() and '--p3 ' not in shell_block()
    for name in ('source','results'): (tmp_path/name).mkdir()
    c['prefill']['fa3_prepack_gate']=False;c['prefill']['fa3_v3_gate']=False;c['prefill']['fa3_v_writer_gate']=True
    with pytest.raises(AssertionError,match='W gate requires prepack'):
        runpy.run_path(str(ROOT/'scripts/prepare-firmus-pp4-settings.py'))['prepare'](c,ROOT,tmp_path)


@pytest.mark.parametrize('corrupt',[None,'q','k','kc','vc','image','attention','scale','O','v'])
def test_w_validation_executes_both_orders_and_detects_bytes(monkeypatch,corrupt):
    torch=pytest.importorskip('torch')
    common={key:torch.zeros((2,1,1)) for key in ('q','k','kc','vc','v')}
    common.update(rows=260,lengths=(130,130),packed=torch.randn(260,2),rope_args=(None,)*11+(torch.arange(260),))
    state={'image':torch.zeros(2,dtype=torch.uint8),'output':{},'o_outputs':{},'composite':{},'writers':{}}
    trace=[]
    for mode in ('P','W'):
        # Column-major scales exercise the same dtype-view restriction as serving.
        state['output'][mode]=(torch.empty((2,2,1)),torch.empty((2,2)).T)
        state['o_outputs'][mode]=torch.empty(2)
        def writer(mode=mode):
            value=float(common['packed'][0].sum())
            for key in ('q','k','kc','vc'): common[key].fill_(value)
            if mode=='P': common['v'].fill_(value)
            state['image'].fill_(int(abs(value)*17)%256)
        state['writers'][mode]=writer
        def call(mode=mode):
            trace.append(mode);state['writers'][mode]()
            for tensor in (*state['output'][mode],state['o_outputs'][mode]): tensor.fill_(common['q'][0,0,0])
            if mode=='W' and corrupt:
                mapping={**common,'image':state['image'],'attention':state['output'][mode][0],'scale':state['output'][mode][1],'O':state['o_outputs'][mode]}
                mapping[corrupt].add_(1)
            return state['output'][mode]
        state['composite'][mode]=call
    class Torch:
        cuda=SimpleNamespace(synchronize=lambda:None)
        ops=SimpleNamespace(apac_fa3_verify=SimpleNamespace(prefill_dump_native=lambda v,dump,cu,n:dump.copy_(state['image'])))
        def __getattr__(self,key): return getattr(torch,key)
    class Graph:
        def __init__(self,call): self.call=call
        def replay(self): self.call()
    checks={'prefill_selection_capture':Graph,'prefill_fp32_reference':lambda c:([0],torch.zeros(1)), 'prefill_assert_errors':lambda *a:{'relative_l2':.01,'max_scaled_error':.02}}
    common['cu']=None
    monkeypatch.setitem(G['validate_w'].__globals__,'host_image',lambda t,c,image:image.clone())
    if corrupt:
        with pytest.raises(AssertionError): G['validate_w'](Torch(),checks,common,state)
    else:
        result=G['validate_w'](Torch(),checks,common,state)
        assert result['validation_orders']==['forward','reverse'] and len(result['candidate_fp32'])==3
        assert trace==['P','W','W','P']*3


def test_w_allows_exact_one_us_regression(screen):
    for p in screen[0].glob('prefill-*/rank-*.json'):
        r=json.loads(p.read_text())
        r['candidates']['W']['shapes']['3900+4096']['samples_us']={o:{'control':[900.]*10,'candidate':[901.]*10} for o in ('forward','reverse')}
        p.write_text(json.dumps(r))
    assert aggregate(screen)['adopted']


@pytest.mark.parametrize('order',['forward','reverse'])
def test_w_order_regression_cannot_hide_in_pooled(screen,order):
    def mutate(r):
        s=r['candidates']['W']['shapes']['3900+4096']
        s['samples_us'][order]['candidate']=[901.001]*10
    edit(screen,mutate)
    assert not aggregate(screen)['adopted']


def test_w_inverse_exhaustive_against_independent_forward_map(tmp_path):
    added='\n'.join(l[1:] for l in PATCH.read_text().splitlines() if l.startswith('+') and not l.startswith('+++'))
    function='int qkv_w_logical_offset'+added.split('int qkv_w_logical_offset',1)[1].split('\n}',1)[0]+'\n}'
    path=tmp_path/'inverse.cpp';binary=tmp_path/'inverse'
    path.write_text('#include <cstdio>\n'+function+'\nint main(){for(int x=0;x<12288;x++)std::printf("%d\\n",qkv_w_logical_offset(x));}\n')
    subprocess.run(['g++','-O2',str(path),'-o',str(binary)],check=True,capture_output=True)
    inverse=[int(x) for x in subprocess.check_output([str(binary)],text=True).splitlines()]
    assert set(inverse)==set(range(12288))
    for logical in range(12288):
        physical=0
        for bit,mask in enumerate((256,32,64,144,512,1024,2048,1,4,8,2,16,4096,8192)):
            physical^=((logical>>bit)&1)*mask
        assert inverse[physical]==logical


def test_w_cpu_admission_preserves_old_source_math_and_bytes():
    evidence=json.loads((ROOT/'research/runtime-performance/prefill-v-writer-r23-revised-cpu-validation-20261102.json').read_text())
    assert evidence['shared_inline_qk_body']['byte_equal_to_original_body']
    assert evidence['shared_inline_qk_body']['calls_from_old_and_W']==2
    assert evidence['W_V_branch_matches_admitted_diagnostic'] and evidence['W_inverse_matches_production_pack']
    for r in evidence['resources'].values():
        assert r['registers']==48 and r['stack_bytes']==r['spill_store_bytes']==r['spill_load_bytes']==0 and not r['C75xx']
    for r in evidence['occupancy_model']:
        assert r['ctas_per_sm']==r['register_limit']==5 and r['thread_limit']==8
    assert all(r['encoded_equal'] for group in evidence['unchanged_JIT_pack_encoded_SASS'].values() for r in group.values())
    assert {r['row'] for r in evidence['FA3_encoded_SASS']}=={'control-n192','P','P3'}
    assert all(r['encoded_equal_to_round17'] for r in evidence['FA3_encoded_SASS'])
    byte=evidence['CPU_byte_validation']
    assert byte['bf16_patterns']==65536 and byte['bijection_coordinates']==12288
    assert byte['helper_vs_cuda_nosat_equal'] and byte['helper_vs_old_lane_repair_equal'] and byte['helper_vs_independent_rne_equal']
    assert len(byte['cases'])==7
    assert evidence['GPU_byte_equality'] is evidence['GPU_composite_us'] is None
