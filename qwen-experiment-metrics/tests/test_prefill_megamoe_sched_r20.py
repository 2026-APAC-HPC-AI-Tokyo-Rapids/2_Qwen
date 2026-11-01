"""Round20 scheduling model, strict UUID gate, capture and timeout contracts."""
import copy
import json
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

ROOT = Path(__file__).resolve().parents[2]
MODEL = runpy.run_path(str(ROOT/'scripts/model-prefill-megamoe-schedule.py'))
TIMING = runpy.run_path(str(ROOT/'scripts/run-prefill-megamoe-timing.py'))
PREPACK = runpy.run_path(str(ROOT/'scripts/select-firmus-prefill-fa3-prepack.py'))
CAPTURE = runpy.run_path(str(ROOT/'scripts/capture-prefill-routing-pp4.py'))


def build():
    return {'control_sass_equal':True,'candidate_header_sha256':'c'*64,'before':{'spill_store_load_bytes':[['0','0']]},'candidate':{'spill_store_load_bytes':[['0','0']],'c75xx':[]}}


def gpu_record(worker,rank,candidate=True):
    case = {'byte_equal':True,'finite':True,'control_fp32':[{'relative_l2':.01,'max_scaled_error':.02}]*2,'samples_us':[{'round':i,**{order:{'control':2500.,**({'L':2490.} if candidate else {})} for order in ('forward','reverse')}} for i in range(8)]}
    return {'schema':TIMING['SCHEMA'],'run_id':'123-pass','worker':f'prefill-{worker}','rank':rank,'uuid':f'uuid-{worker}-{rank}','device':'NVIDIA H200','image_sha256':'a'*64,'finished_at':time.time(),'elapsed_seconds':35.,'candidate':candidate,'complete':True,'routing_orders':[list(TIMING['routing_order'](i)) for i in range(8)],'build':build(),'cases':{name:copy.deepcopy(case) for name in TIMING['ROUTINGS']},'endurance':{'calls':600,'bytes_equal':True,'finite':True,'completed':True,'inputs':2,'routings':list(TIMING['ROUTINGS'])}}


def screen(tmp_path,candidate=True):
    directory = tmp_path/'timing'; directory.mkdir()
    uuids=[]
    for worker in range(3):
        (directory/f'prefill-{worker}').mkdir()
        for rank in range(4):
            record = gpu_record(worker,rank,candidate)
            uuids.append(record['uuid'])
            (directory/f'prefill-{worker}'/f'rank-{rank}.json').write_text(json.dumps(record))
    p = tmp_path/'P.json'; p.write_text(json.dumps({'adopted':True,'selected':'P','gpus':[{'uuid':u} for u in uuids[::-1]]}))
    env = tmp_path/'serving.env'; env.write_text('SGLANG_PREFILL_FA3_VERIFY='+PREPACK['DISPATCH']['P']+'\nOTHER=kept\n')
    return directory,p,env


def aggregate(tmp_path,args,candidate=True,deadline=None):
    return TIMING['aggregate'](*args,tmp_path/'selection.json','123-pass',deadline or time.time()+60,'a'*64,candidate)


def test_gate_matches_uuid_sets_and_appends_only_l(tmp_path):
    args=screen(tmp_path); before=args[2].read_text()
    result=aggregate(tmp_path,args)
    assert result['adopted'] is True and len(result['gpus'])==12
    assert args[2].read_text()==before+TIMING['KEY']+'=1\n'
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('problem',['missing','stale','uuid','duplicate','worker','rank','image','header','late','incomplete','order','nan','below_forward','below_reverse','bits','finite','reference','reference_nan','endurance','endurance_bytes','endurance_finite','spill','c75','sass','wrong_candidate','P','P_uuid','deadline'])
def test_gate_refuses_every_incomplete_or_bad_gpu_without_touching_p(tmp_path,problem):
    args=screen(tmp_path); path=args[0]/'prefill-2/rank-3.json';r=json.loads(path.read_text());before=args[2].read_bytes()
    if problem=='missing':path.unlink()
    elif problem=='stale':r['run_id']='old'
    elif problem=='uuid':r['uuid']='other'
    elif problem=='duplicate':r['uuid']='uuid-0-0'
    elif problem=='worker':r['worker']='prefill-0'
    elif problem=='rank':r['rank']=0
    elif problem=='image':r['image_sha256']='b'*64
    elif problem=='header':r['build']['candidate_header_sha256']='d'*64
    elif problem=='late':r['finished_at']=time.time()+1000
    elif problem=='incomplete':r['complete']=False
    elif problem=='order':r['routing_orders'][1].reverse()
    elif problem=='nan':r['cases']['edges']['samples_us'][0]['forward']['L']=float('nan')
    elif problem.startswith('below_'):
        order=problem.split('_')[1]
        for sample in r['cases']['edges']['samples_us']:sample[order]['L']=2490.1
    elif problem=='bits':r['cases']['balanced']['byte_equal']=False
    elif problem=='finite':r['cases']['balanced']['finite']=False
    elif problem=='reference':r['cases']['balanced']['control_fp32'][0]['relative_l2']=.03
    elif problem=='reference_nan':r['cases']['balanced']['control_fp32'][0]['max_scaled_error']=float('nan')
    elif problem=='endurance':r['endurance']['calls']=599
    elif problem=='endurance_bytes':r['endurance']['bytes_equal']=False
    elif problem=='endurance_finite':r['endurance']['finite']=False
    elif problem=='spill':r['build']['candidate']['spill_store_load_bytes']=[['4','4']]
    elif problem=='c75':r['build']['candidate']['c75xx']=['C7514']
    elif problem=='sass':r['build']['control_sass_equal']=False
    elif problem=='wrong_candidate':r['candidate']=False
    elif problem in ('P','P_uuid'):
        p=json.loads(args[1].read_text());p['selected']='S' if problem=='P' else 'P';p['gpus'][0]['uuid']='other';args[1].write_text(json.dumps(p))
    if problem!='missing':path.write_text(json.dumps(r))
    result=aggregate(tmp_path,args,deadline=time.time()-1 if problem=='deadline' else None)
    assert result['adopted'] is False and result['complete'] is False and result['reason']
    assert args[2].read_bytes()==before


def test_control_observation_does_not_publish_l(tmp_path):
    args=screen(tmp_path,False);before=args[2].read_bytes()
    result=aggregate(tmp_path,args,False)
    assert result['complete'] is True and result['adopted'] is False
    assert args[2].read_bytes()==before


def test_cpu_model_covers_every_tile_and_keeps_expert_order():
    cases=MODEL['fixtures']()
    maxima={'balanced':(47,47),'historical-routing':(50,49),'edges':(50,49)}
    for name,case in cases.items():
        groups=MODEL['assign'](case['expert_rows'])
        got=[(e,m,n) for group in groups for e,m,p,v in group for n in range(12)]
        expected=[(e,m,n) for e,count in enumerate(case['expert_rows']) for m in range((count+127)//128) for n in range(12)]
        assert sorted(got)==expected and len(set(got))==len(got)
        for group in groups:
            assert [(e,m) for e,m,p,v in group]==sorted((e,m) for e,m,p,v in group)
            assert all(p==sum((count+127)//128 for count in case['expert_rows'][:e])+m and v==min(128,case['expert_rows'][e]-m*128) for e,m,p,v in group)
        assert (case['control']['mma']['max'],case['candidate']['mma']['max'])==maxima[name]
        assert sum(case['candidate']['cost07']['histogram'].values())==132
        assert case['candidate']['cost07']['max'] <= case['control']['cost07']['max']
    # Stable tie -> groups 0..10; two consecutive zero-expert gaps remain safe.
    assert [g[0][:2] for g in MODEL['assign']([0,0,1408]+[0]*125)]==[(2,m) for m in range(11)]


def test_only_schedule_changes_not_numeric_body_and_preparation_is_timed():
    patch=(ROOT/'patches/deep-gemm-sched/expert-greedy.patch').read_text()
    removed=[l[1:] for l in patch.splitlines() if l.startswith('-') and not l.startswith('---')]
    added=[l[1:] for l in patch.splitlines() if l.startswith('+') and not l.startswith('+++')]
    assert all('scheduler' in l or 'l1_func' in l or 'n_block_idx' in l or 'kTailPair' in l or 'continue' in l or '}' in l or 'sm90_fp8_mega_moe_impl(' in l for l in removed)
    assert any('__reduce_min_sync' in l for l in added) and any('<= 64 ? 7 : 10' in l for l in added)
    source=(ROOT/'scripts/run-prefill-megamoe-timing.py').read_text()
    assert 'begin.record(); graphs[mode].replay(); end.record()' in source
    assert 'torch.equal(a.view(torch.uint8),b.view(torch.uint8))' in source
    assert "range(ENDURANCE)" in source and 'expected[name,replay]' in source
    assert 'fp8_mega_moe_diag(' not in source and 'torch.cuda._sleep' not in source
    host=(ROOT/'patches/deep-gemm-host/sm90-prefill-fc1-balanced.patch').read_text()
    assert 'and num_tokens == 7988 and prefill_split_accum and num_ranks == 1' in host
    assert 'if constexpr (not Diagnostic)' in host and 'include_parser->get_hash_value(code)' in host
    assert 'DG_SM90_PREFILL_FC1_BALANCED' not in (ROOT/'scripts/prepare-firmus-pp4-settings.py').read_text()


def capture_module(monkeypatch,torch):
    monkeypatch.setitem(sys.modules,'torch',torch)
    patch=(ROOT/'patches/sglang/prefill-routing-capture-pp4.patch').read_text()
    code=patch.split('+++ b/python/sglang/srt/layers/moe/prefill_routing_capture_pp4.py\n',1)[1]
    code='\n'.join(l[1:] for l in code.splitlines() if l.startswith('+'))
    scope={};exec(compile(code,'capture.py','exec'),scope)  # noqa: S102 - execute the production capture with CPU tensor collaborators
    return scope


def test_capture_does_nothing_during_measurement_even_when_configured(monkeypatch):
    class Forbidden:
        def __getattr__(self,name):raise AssertionError('CUDA during measurement: '+name)
    c=capture_module(monkeypatch,Forbidden())
    c['ENABLED']=True;c['DIRECTORY']='must-not-be-accessed'
    c['CaptureState']=lambda *_:pytest.fail('state/IO during measurement')
    assert c['begin_forward'](object(),NS(rids=['ordinary-request']),1) is False
    c['record'](0,object(),7988);c['end_forward']()
    assert c['_state'] is None


class Array:
    def __init__(self,array):self.array=np.array(array)
    @property
    def shape(self):return self.array.shape
    def __getitem__(self,key):return Array(self.array[key])
    def detach(self):return self
    def to(self,**kwargs):
        assert kwargs=={'dtype':'i32','copy':True}
        return Array(self.array.astype(np.int32))
    def cpu(self):return self
    def numpy(self):return self.array


def test_capture_keeps_pp_subset_boundaries_and_copy_before_reuse(tmp_path,monkeypatch):
    directory=tmp_path/'results/prefill-routing-capture/prefill-0';directory.mkdir(parents=True)
    (tmp_path/'source').mkdir();(tmp_path/'source/prefill-kernels.env').write_text('P env');(tmp_path/'source/firmus-framework.json').write_text(json.dumps({'images':{'sglang':{'sha256':'a'*64}}}));(tmp_path/'source/provenance.json').write_text(json.dumps({'commit':'source'}))
    p=tmp_path/'results/prefill-fa3-prepack';p.mkdir();(p/'selection.json').write_text(json.dumps({'adopted':True,'selected':'P'}))
    torch=NS(int32='i32',cuda=NS(current_device=lambda:1,get_device_properties=lambda _:NS(uuid='uuid')),stack=lambda seq:Array(np.stack([x.array for x in seq])))
    c=capture_module(monkeypatch,torch);c['DIRECTORY']=str(directory);c['ENABLED']=True
    runner=NS(ps=NS(pp_rank=1),layer_info=NS(start_layer=24,end_layer=48))
    batch=NS(rids=['r20-routing-a','r20-routing-b'],extend_seq_lens_cpu=[2,3],extend_prefix_lens_cpu=[0,0])
    for sample in range(2):
        assert c['begin_forward'](runner,batch,100+sample)
        for layer in range(24,48):
            ids=Array(np.full((8,8),layer))
            c['record'](layer,ids,5)
            assert np.all(ids.array==layer)
            ids.array[:]=-1  # workspace reuse must not corrupt the dedicated copy
        c['end_forward']()
        with np.load(directory/f'pp-1-sample-{sample}.npz') as data:
            meta=json.loads(str(data['metadata']))
            assert data['ids'].shape==(24,5,8) and data['ids'].dtype==np.int32
            assert (data['ids'][:,0,0]==np.arange(24,48)).all()
            assert meta['gpu_uuid']=='uuid' and meta['forward_id']==100+sample and meta['request_lengths']==[2,3]
    assert not c['begin_forward'](runner,batch,102)


def test_same_process_and_config_build_wiring(tmp_path):
    config=json.loads((ROOT/'configs/firmus-prefill-pp4.json').read_text())
    assert all(config['prefill'][key] is True for key in ('megamoe_fc1_balance_gate','megamoe_timing','routing_capture')) and config['prefill']['megamoe_diag'] is False
    assert config['container_sha256'].startswith('5a3df8c4') and config['images']['sglang']['sha256']==config['container_sha256']
    shell=(ROOT/'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    combined=shell.split('# Reserve the complete P',1)[1].split('# Default off. Prepack/sum',1)[0]
    assert shell.index('unset DG_SM90_PREFILL_FC1_BALANCED') < shell.index('source "$(dirname "$0")/prefill-kernels.env"')
    assert combined.count('torchrun')==1 and 'run-prefill-p-gates.py' in combined
    assert '+ 150 -' in combined and '+ 155' in combined and '--workers-ok 1' in combined
    wrapper=(ROOT/'scripts/run-prefill-p-gates.py').read_text()
    assert wrapper.index("prepack['run']")<wrapper.index("timing['run']") and 'a.deadline-69' in wrapper
    block=shell.split('# Separate routing pass:',1)[1].split('if [[ "${PROFILE}"',1)[0]
    assert 'timeout --signal=KILL 15 srun' in block and '|| true' in block
    assert shell.index('test -s "${ROOT}/results/score.json"')<shell.index('# Separate routing pass:')
    staging=(ROOT/'scripts/build-sglang-container.sh').read_text();block=staging[staging.index('# The private diagnostic header'):staging.index('base_reference=')]
    context=tmp_path/'context';(context/'containers').mkdir(parents=True)
    subprocess.run(['bash','-euc',block],env={**os.environ,'GITHUB_WORKSPACE':str(ROOT),'context':str(context)},check=True)
    assert (context/'deep-gemm-sched/expert-greedy.patch').read_bytes()==(ROOT/'patches/deep-gemm-sched/expert-greedy.patch').read_bytes()
    assert (context/'containers/build-prefill-megamoe-sched.py').is_file()
    docker=(ROOT/'containers/Dockerfile').read_text()
    assert 'RUN /opt/sglang/venv/bin/python /opt/apac/build-prefill-megamoe-sched.py' in docker
    provision=(ROOT/'scripts/provision-firmus-framework.sh').read_text()
    for name in ('run-prefill-p-gates.py','run-prefill-megamoe-timing.py','capture-prefill-routing-pp4.py','select-firmus-prefill-megamoe.py'):
        assert name in provision


@pytest.mark.parametrize('late_status',['failure','timeout'])
def test_shell_late_megamoe_failure_preserves_finished_p_and_continues(tmp_path,late_status):
    root=tmp_path/'runtime';commands=tmp_path/'commands';commands.mkdir()
    for name in ('source','scripts','logs','results/prefill-fa3-pair'):(root/name).mkdir(parents=True,exist_ok=True)
    data=tmp_path/'fragments';data.mkdir()
    old=runpy.run_path(str(ROOT/'qwen-experiment-metrics/tests/test_prefill_fa3_prepack.py'))
    fixture=old['screen'].__wrapped__(data)
    (root/'source/prefill-kernels.env').write_text(old['PIN'])
    (root/'source/firmus-framework.json').write_text(json.dumps({'prefill':{'megamoe_timing':True,'megamoe_fc1_balance_gate':False},'images':{'sglang':{'sha256':'a'*64}}}))
    (root/'results/prefill-fa3-pair/selection.json').write_bytes(fixture[2].read_bytes())
    for name in ('select-firmus-prefill-fa3-prepack.py','run-prefill-megamoe-timing.py'):(root/'scripts'/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    # The mock srun persists the completed P result, then fails/hangs only in
    # the later MegaMoE phase. The shell's actual timeout and aggregate run.
    worker_code=f'''#!{sys.executable}
import sys,json,time
from pathlib import Path
a=sys.argv
worker=a[a.index('--worker')+1];run_id=a[a.index('--run-id')+1]
out=Path({str(root)!r})/'results/prefill-fa3-prepack'/worker;out.mkdir(parents=True,exist_ok=True)
for rank in range(4):
 r=json.loads((Path({str(data)!r})/worker/f'rank-{{rank}}.json').read_text());r.update(run_id=run_id,finished_at=time.time());(out/f'rank-{{rank}}.json').write_text(json.dumps(r))
{'time.sleep(3)' if late_status=='timeout' else 'sys.exit(1)'}
'''
    (commands/'srun').write_text(worker_code);(commands/'srun').chmod(0o755)
    (commands/'timeout').write_text('#!/bin/bash\nif [[ "$1" == --kill-after=2 ]]; then shift 2; exec /usr/bin/timeout --kill-after=.02 .3 "$@"; fi\nshift 2\nexec /usr/bin/timeout 5 "$@"\n');(commands/'timeout').chmod(0o755)
    (commands/'uv').write_text('#!/bin/bash\nshift 5\nexec '+shlex.quote(sys.executable)+' "$@"\n');(commands/'uv').chmod(0o755)
    source=(ROOT/'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    block='# Reserve the complete P'+source.split('# Reserve the complete P',1)[1].split('# Default off. Prepack/sum',1)[0]
    preamble='ROOT="$1"; ART="$ROOT/logs"; HEAD=head; TAIL=tail; SLURM_JOB_ID=123; SGLANG=image; prefill_pp=4; mounts() { echo unused; }; '
    result=subprocess.run(['bash','-euc',preamble+block+'\ntouch "$ROOT/serving-reached"','fixture',str(root)],env={**os.environ,'PATH':str(commands)+os.pathsep+os.environ['PATH']},capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stderr
    assert (root/'serving-reached').exists()
    assert PREPACK['DISPATCH']['P'] in (root/'source/prefill-kernels.env').read_text()
    assert TIMING['KEY'] not in (root/'source/prefill-kernels.env').read_text()
    assert json.loads((root/'results/prefill-fa3-prepack/selection.json').read_text())['adopted'] is True
    assert json.loads((root/'results/prefill-megamoe-timing/selection.json').read_text())['complete'] is False


def test_routing_pass_failure_never_rewrites_official_result(tmp_path):
    for name in ('source','results','logs'):(tmp_path/name).mkdir()
    (tmp_path/'source/firmus-framework.json').write_text('{"prefill":{"routing_capture":true}}')
    (tmp_path/'results/score.json').write_text('official')
    source=(ROOT/'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    block='# Separate routing pass:'+source.split('# Separate routing pass:',1)[1].split('if [[ "${PROFILE}"',1)[0]
    preamble='ROOT="$1"; ART="$ROOT/logs"; HEAD=head; SGLANG=image; profile_only=false; benchmark_status=7; mounts() { echo unused; }; srun() { return 1; }; export -f srun; timeout() { shift 2; "$@"; }; '
    subprocess.run(['bash','-euc',preamble+block+'\n[[ "$benchmark_status" == 7 ]]; touch "$ROOT/continued"','fixture',str(tmp_path)],check=True)
    assert (tmp_path/'results/score.json').read_text()=='official' and (tmp_path/'continued').exists()


def test_byte_oracle_detects_signed_zero_instead_of_float_equality():
    class Tensor:
        def __init__(self,values):self.data=np.asarray(values,dtype=np.float32)
        def view(self,dtype):return self.data.view(dtype)
    torch=NS(uint8=np.uint8,equal=lambda a,b:np.array_equal(a,b))
    assert TIMING['bytes_equal'](torch,Tensor([1.,0.]),Tensor([1.,0.]))
    assert not TIMING['bytes_equal'](torch,Tensor([1.,0.]),Tensor([1.,-0.]))


@pytest.mark.parametrize('busy_peer',[False,True])
def test_capture_controller_checks_all_dp_peers_before_paired_dispatch(tmp_path,monkeypatch,busy_peer):
    (tmp_path/'source').mkdir();p=tmp_path/'results/prefill-fa3-prepack';p.mkdir(parents=True)
    (p/'selection.json').write_text(json.dumps({'adopted':True,'selected':'P','gpus':[{'uuid':f'gpu-{i}'} for i in range(12)]}))
    (tmp_path/'source/worker-endpoints.tsv').write_text('prefill-0 head 30100 1\nprefill-1 head 30101 1\nprefill-2 tail 30102 1\ndecode-0 tail 30200 4\n')
    calls=[]
    def request(url,body,deadline):
        calls.append((url,body))
        if '/flush_cache' in url:return 'Cache flushed.'
        if '/v1/loads' in url:
            ranks=4 if ':30200/' in url else 1
            return json.dumps({'loads':[{'num_running_reqs':int(busy_peer and r==1),'num_waiting_reqs':0,'disaggregation':dict.fromkeys(('prefill_bootstrap_queue_reqs','prefill_inflight_queue_reqs','decode_prealloc_queue_reqs','decode_transfer_queue_reqs','decode_retracted_queue_reqs'),0)} for r in range(ranks)]})
        assert url.endswith('/generate')
        assert body['rid'][0].startswith('r20-routing-') and len(body['input_ids'])==4 and all(len(ids)==3994 for ids in body['input_ids'])
        return '{}'
    function=CAPTURE['run']
    monkeypatch.setitem(function.__globals__,'request',request)
    monkeypatch.setitem(function.__globals__,'summarize',lambda directory:[{'gpu_uuid':f'gpu-{i}'} for i in range(12)])
    function(tmp_path,time.time()+14)
    status=json.loads((tmp_path/'results/prefill-routing-capture/status.json').read_text())
    generated=[(url,body) for url,body in calls if url.endswith('/generate')]
    if busy_peer:
        assert status['complete'] is False and 'outstanding DP' in status['error']
        assert generated==[]
    else:
        assert status['complete'] is True and len(generated)==6
        for i in range(0,6,2):
            assert generated[i][1]==generated[i+1][1]  # exactly matched P/D bootstrap rooms and inputs
