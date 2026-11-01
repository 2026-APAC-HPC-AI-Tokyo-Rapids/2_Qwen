import hashlib
import json
import os
import runpy
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = runpy.run_path(str(ROOT/'scripts/run-prefill-megamoe-diag.py'))
SUMMARY = runpy.run_path(str(ROOT/'scripts/summarize-prefill-megamoe-diag.py'))


def fixture_case():
    rows = []
    for cta in range(132):
        for warp in range(12):
            events = [[1000,1,0,0,0,0,0,0]]
            if warp >= 4:
                if warp in (4,8):
                    for phase in (1,2):
                        start = phase*1000+1000
                        # Last CTA has a tail and is the last math finisher.
                        tail = cta==131
                        context = [phase,phase,64 if tail else 128,7,19]
                        events += [[start,2,*context,3],[start+500,3,*context,0],[start+600+int(tail)*100,4,*context,0]]
                events += [[4000,16,0,0,0,0,0,0],[4500,17,0,0,0,0,0,0],[4600,18,0,0,0,0,0,0]]
            events.append([5000,15,0,0,0,0,0,0])
            rows.append({'cta':cta,'warp':warp,'sm':cta%132,'events':events,'wait_ns':{'fc1_full':100,'fc2_full':100,'fc1_empty':500,'fc2_empty':500}})
    return {'samples_us':[{'order':['control','diagnostic'] if i%2==0 else ['diagnostic','control'],'control':10,'diagnostic':10.1} for i in range(6)],'byte_equal':True,'trace':rows,'expert_rows':[500]*128}


def fixture_build():
    return {'serving_sass_equal':True,'diagnostic':{'spill_store_load_bytes':[['0','0']],'c75xx':[]}}


def test_summary_reports_nested_waits_tail_and_overhead():
    result = SUMMARY['summarize_case'](fixture_case(),fixture_build())
    assert result['instrumented_warp_span_us']==4
    assert result['cta_wg_distributions']['fc1_mainloop_us']['median']==.5
    assert result['cta_wg_distributions']['fc1_full_wait_fraction_within_mainloop']['median']==.2
    assert result['timing']['impact_fraction']==pytest.approx(.01)
    assert result['fine_intervals_usable_for_adoption'] is True
    assert result['tail']['last_math_tiles'][0]['cta']==131
    assert result['tail']['last_math_ctas_have_any_fc2_tail'] is True
    assert result['tail']['last_fc2_tail_minus_last_full_us']==.1
    assert result['tail']['fc2_true_m64_tiles']==2
    assert 'total_phase_sum' not in result


@pytest.mark.parametrize('problem',['overhead','spill','schedule','bytes','order','nan','tile','clock'])
def test_summary_rejects_or_qualifies_damaged_evidence(problem):
    case,build = fixture_case(),fixture_build()
    if problem=='overhead':
        for r in case['samples_us']:r['diagnostic']=11
    if problem=='spill':build['diagnostic']['spill_store_load_bytes']=[['4','4']]
    if problem=='schedule':build['diagnostic']['c75xx']=['C7514']
    if problem=='bytes':case['byte_equal']=False
    if problem=='order':case['samples_us'][1]['order']=['control','diagnostic']
    if problem=='nan':case['samples_us'][1]['control']=float('nan')
    if problem=='tile':case['trace'][4]['events'].pop(2)
    if problem=='clock':case['trace'][4]['events'][2][0]=1
    if problem in ('overhead','spill','schedule'):
        assert SUMMARY['summarize_case'](case,build)['fine_intervals_usable_for_adoption'] is False
    else:
        with pytest.raises(AssertionError):SUMMARY['summarize_case'](case,build)


def test_half_and_scale_stamps_are_nested_and_kept_separate():
    case = fixture_case()
    row = case['trace'][4]
    context = [1,1,128,7,19]
    detail = [[2010+i*10,event,*context,1 if event in (11,12,19,20) else 2] for i,event in enumerate((7,8,9,10,11,19,20,12,13,6))]
    # Drain key must match; input arg=2 also means half 0 of K=1.
    row['events'][2:2] = detail
    result = SUMMARY['summarize_case'](case,fixture_build())
    metrics = result['sampled_intervals_us']
    assert metrics['fc1/full/accumulator_wait/half0']['median']==.01
    assert metrics['fc1/full/rescale/half0']['median']==.01
    assert metrics['fc1/full/sampled_full_wait']['median']==.01
    assert metrics['fc1/full/scale_prepare_including_full_wait']['median']==.03
    assert metrics['fc1/full/scale_prepare_excluding_full_wait']['median']==.02
    assert metrics['fc1/full/chunk_drain']['median']==.01


def test_routing_is_exactly_7988_and_has_requested_expert_edges():
    cases = RUNNER['routing_cases'](np,ROOT/'kernels/probes/megamoe-local-ep1-routing.npz')
    assert set(cases)=={'balanced','historical-routing','edges'}
    for ids in cases.values():
        assert ids.shape==(7988,8) and ids.min()>=0 and ids.max()<128
        assert np.all(np.diff(np.sort(ids,axis=1),axis=1)>0)
    assert np.bincount(cases['edges'].ravel(),minlength=128)[:7].tolist()==[0,1,64,65,127,128,129]
    data=np.load(ROOT/'kernels/probes/megamoe-local-ep1-routing.npz')
    assert np.array_equal(cases['historical-routing'],data['logical_ids'][:,0,0,:3994,:].reshape(7988,8))


def test_private_trace_parser_never_accepts_overflow_or_missing_warps(monkeypatch):
    function = RUNNER['parse_trace']
    for key,value in {'CTAS':2,'WARPS':2,'WORDS':64,'CAPACITY':4}.items():monkeypatch.setitem(function.__globals__,key,value)
    data = np.zeros((2,2,64),dtype=np.uint64)
    data[:,:,0]=RUNNER['MAGIC'];data[:,:,1]=2;data[:,:,3]=7988
    data[:,:,32:40]=[10,1,0,0,0,0,0,0]
    data[:,:,40:48]=[20,15,0,0,0,0,0,0]
    # Finder searches only before the next warp header in the production large stride.
    monkeypatch.setitem(function.__globals__,'MAGIC',RUNNER['MAGIC'])
    # Use production stride here to avoid several headers in the search window.
    monkeypatch.setitem(function.__globals__,'WORDS',RUNNER['WORDS'])
    large=np.zeros((2,2,RUNNER['WORDS']),dtype=np.uint64);large[:,:,:64]=data
    assert len(function(np,large.ravel()))==4
    large[1,1,4]=1
    with pytest.raises(AssertionError,match='overflow'):function(np,large.ravel())
    large[1,1,4]=0;large[1,1,0]=0
    with pytest.raises(AssertionError,match='missing'):function(np,large.ravel())


def test_aggregate_requires_all_twelve_same_run_unique_gpus(tmp_path):
    fn=SUMMARY['summarize']
    for worker in range(3):
        path=tmp_path/f'prefill-{worker}';path.mkdir()
        for rank in range(4):
            data={'schema':SUMMARY['SCHEMA'],'run_id':'r18','rank':rank,'tokens':7988,'uuid':f'gpu{worker}-{rank}','passed':True,'routing_is_current_serving':False,'build':fixture_build(),'cases':{k:fixture_case() for k in ('balanced','historical-routing','edges')}}
            (path/f'rank-{rank}.json').write_text(json.dumps(data))
    assert fn(tmp_path,'r18')['complete'] is True
    assert fn(tmp_path,'different')['complete'] is False
    (tmp_path/'prefill-2/rank-3.json').unlink()
    result=fn(tmp_path,'r18')
    assert result['complete'] is False and len(result['gpus'])==11
    assert result['adoption'] is None


@pytest.mark.parametrize('status',[0,1,124,137,'timeout'])
def test_real_shell_failure_timeout_and_summary_failure_keep_p(tmp_path,status):
    shell=(ROOT/'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    block=shell[shell.index('# Independent diagnostic-only step.'):shell.index('srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4')]
    root=tmp_path/'runtime';(root/'source').mkdir(parents=True);(root/'logs').mkdir()
    (root/'source/firmus-framework.json').write_text(json.dumps({'prefill':{'megamoe_diag':True}}))
    env=root/'source/prefill-kernels.env';env.write_text('SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen-pipe+m192n96-pair-prepack\nDG_SM90_PREFILL_SPLIT_ACCUM=1\n')
    p_json=root/'source/p-selection.json';p_json.write_text('{"selected":"P","adopted":true}\n')
    before=(env.read_bytes(),p_json.read_bytes())
    commands=tmp_path/'bin';commands.mkdir()
    # Mock at the srun boundary; the production timeout command and waits execute.
    code = 'sleep 5' if status=='timeout' else f'exit {status}'
    (commands/'srun').write_text(f'#!/bin/bash\nprintf "%s\\n" "$*" >>"{tmp_path}/calls"\n{code}\n')
    (commands/'uv').write_text('#!/bin/bash\nexit 1\n')
    if status=='timeout':
        (commands/'timeout').write_text('#!/bin/bash\nshift 2\nexec /usr/bin/timeout --kill-after=.01 .05 "$@"\n')
    for p in commands.iterdir():p.chmod(0o755)
    result=subprocess.run(['bash','-euc','ROOT="$1"; ART="$ROOT/logs"; HEAD=head; TAIL=tail; SGLANG=image; SLURM_JOB_ID=18; prefill_pp=4; prefill_fa3_prepack_deadline=$(($(date +%s)+180)); mounts() { echo fixture; }; '+block+'\ntouch "$ROOT/serving-reached"','fixture',str(root)],env={**os.environ,'PATH':str(commands)+os.pathsep+os.environ['PATH']},capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stderr
    assert (root/'serving-reached').exists()
    assert (env.read_bytes(),p_json.read_bytes())==before
    calls=(tmp_path/'calls').read_text().splitlines()
    assert len(calls)==3 and all('--nproc-per-node=4' in c for c in calls)
    assert sum('-w head' in c for c in calls)==2 and sum('-w tail' in c for c in calls)==1
    assert '--env-output' not in block and 'source/prefill-kernels.env' not in block
    assert 'megamoe_diag_remaining > 3' in block and '+ 45 -' in block
    assert 'prefill_fa3_prepack_deadline - $(date +%s) - 5' in block


def test_build_and_dispatch_are_explicitly_isolated(tmp_path):
    config=json.loads((ROOT/'configs/firmus-prefill-pp4.json').read_text())
    assert config['prefill']['megamoe_diag'] is True
    assert all('deep-gemm-diag' not in p for p in config['deep_gemm_overlay_patches'])
    host=(ROOT/'patches/deep-gemm-host/sm90-prefill-diagnostic.patch').read_text()
    assert 'template <bool Diagnostic = false>' in host
    assert 'if constexpr (Diagnostic)' in host
    assert 'mega::fp8_mega_moe<true>' in host
    assert 'get_env' not in host
    assert 'sm90-prefill-diagnostic.patch' in (ROOT/'patches/deep-gemm-host/series').read_text().split()
    build=(ROOT/'containers/build-prefill-megamoe-diag.py').read_text()
    assert "assert original.read_bytes() == before" in build
    assert "assert baseline['sass_encoding_sha256'] == ordinary['sass_encoding_sha256']" in build
    assert "generated.write_bytes(control_header)" in build
    docker=(ROOT/'containers/Dockerfile').read_text()
    assert 'COPY deep-gemm-diag /opt/apac/deep-gemm-diag' in docker
    assert 'RUN /opt/sglang/venv/bin/python /opt/apac/build-prefill-megamoe-diag.py' in docker
    staging=(ROOT/'scripts/build-sglang-container.sh').read_text()
    block=staging[staging.index('# The private diagnostic header'):staging.index('base_reference=')]
    context=tmp_path/'context';(context/'containers').mkdir(parents=True)
    subprocess.run(['bash','-euc',block],env={**os.environ,'GITHUB_WORKSPACE':str(ROOT),'context':str(context)},check=True)
    for p in config['deep_gemm_overlay_patches']:
        assert (context/Path(p).relative_to('patches')).read_bytes()==(ROOT/p).read_bytes()
    assert (context/'deep-gemm-diag/trace.cuh').read_bytes()==(ROOT/'patches/deep-gemm-diag/trace.cuh').read_bytes()
    provision=(ROOT/'scripts/provision-firmus-framework.sh').read_text()
    for name in ('run-prefill-megamoe-diag.py','summarize-prefill-megamoe-diag.py','megamoe-local-ep1-routing.npz'):
        assert name in provision
    # No other caller of the diagnostic API is present in serving code/patches.
    for directory in ('patches/sglang','kernels/prefill'):
        assert all('fp8_mega_moe_diag' not in p.read_text() for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.patch','.py','.cu','.cuh'))


def test_p_gate_and_image_pin_are_unchanged():
    # Content hashes taken from main beb5640, including its complete P/S gate.
    expected='34bd6827b644f8408e146008781524a761e11d2f61de0adf7d8f40bafacae01f'
    assert hashlib.sha256((ROOT/'scripts/select-firmus-prefill-fa3-prepack.py').read_bytes()).hexdigest()==expected
    config=json.loads((ROOT/'configs/firmus-prefill-pp4.json').read_text())
    assert config['images']['sglang']['sha256']=='5a3df8c485a5ac939761b07a9ca82d7c38d864200250591ec6d9485dba778bb2'


@pytest.mark.parametrize('budget',['disabled','expired'])
def test_missing_or_exhausted_prepack_envelope_skips_without_touching_p(tmp_path,budget):
    shell=(ROOT/'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    block=shell[shell.index('# Independent diagnostic-only step.'):shell.index('srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4')]
    (tmp_path/'source').mkdir();(tmp_path/'logs').mkdir()
    (tmp_path/'source/firmus-framework.json').write_text('{"prefill":{"megamoe_diag":true}}')
    env=tmp_path/'source/prefill-kernels.env';env.write_text('P\n')
    setup='prefill_fa3_prepack_deadline=1;' if budget=='expired' else ''
    subprocess.run(['bash','-euc','ROOT="$1"; ART="$ROOT/logs"; prefill_pp=4; SLURM_JOB_ID=18; HEAD=h; TAIL=t; '+setup+block,'fixture',str(tmp_path)],check=True)
    assert env.read_text()=='P\n'
    result=json.loads((tmp_path/'results/prefill-megamoe-diag/status.json').read_text())
    assert result['complete'] is False and result['adoption'] is None
