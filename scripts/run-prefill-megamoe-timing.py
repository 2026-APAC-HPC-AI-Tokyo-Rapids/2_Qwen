"""Uninstrumented complete MegaMoE timing, called inside P's resident process."""
import json
import math
import os
import runpy
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

SCHEMA = 'prefill-megamoe-timing/v1'
KEY = 'DG_SM90_PREFILL_FC1_BALANCED'
ROUTINGS = ('balanced', 'historical-routing', 'edges')
ROUNDS, REPS, ENDURANCE = 8, 4, 600


def bytes_equal(torch, a, b):
    return torch.equal(a.view(torch.uint8),b.view(torch.uint8))


def read_json(path):
    def reject(value):
        raise ValueError('nonfinite JSON: '+value)
    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            reject(value)
        return number
    return json.loads(path.read_text(),parse_constant=reject,parse_float=finite)


def routing_order(round_index):
    order = ROUTINGS[round_index % 3:] + ROUTINGS[:round_index % 3]
    return order if round_index % 2 == 0 else order[::-1]


def timing_summary(samples, candidate):
    assert len(samples) == ROUNDS
    assert [s['round'] for s in samples] == list(range(ROUNDS))
    savings, medians = {}, {}
    for order in ('forward', 'reverse', 'pooled'):
        medians[order] = {}
        for mode in ('control', 'L') if candidate else ('control',):
            values = [s[o][mode] for s in samples for o in (('forward','reverse') if order == 'pooled' else (order,))]
            assert all(type(x) in (int,float) and math.isfinite(x) and x > 0 for x in values), 'invalid timing'
            medians[order][mode] = statistics.median(values)
        if candidate:
            savings[order] = medians[order]['control'] - medians[order]['L']
    return {'medians_us': medians, 'savings_us': savings}


def validate_record(record, candidate):
    assert record['complete'] is True and record['candidate'] is candidate, 'incomplete/wrong mode'
    assert record['routing_orders'] == [list(routing_order(i)) for i in range(ROUNDS)], 'routing order'
    assert set(record['cases']) == set(ROUTINGS), 'missing routing'
    build = record['build']
    assert build['control_sass_equal'] is True and not build['candidate']['c75xx']
    assert all(int(a)<=int(b) for a,b in zip(build['candidate']['spill_store_load_bytes'][0],build['before']['spill_store_load_bytes'][0],strict=True)), 'spills'
    for case in record['cases'].values():
        assert case['byte_equal'] is True and case['finite'] is True, 'output bits/nonfinite'
        for metric in case['control_fp32']:
            assert all(type(metric[k]) in (int,float) and math.isfinite(metric[k]) and 0 <= metric[k] <= limit for k,limit in (('relative_l2',.025),('max_scaled_error',.08))), 'control reference'
        assert len(case['control_fp32']) == 2, 'input replays'
        timing = timing_summary(case['samples_us'], candidate)
        if candidate:
            assert all(v >= 10 for v in timing['savings_us'].values()), 'below 10 us margin'
    e = record['endurance']
    assert e['calls'] >= ENDURANCE and e['bytes_equal'] is True and e['finite'] is True and e['completed'] is True and e['inputs'] == 2 and set(e['routings']) == set(ROUTINGS), 'endurance'


def aggregate(directory, p_selection, env_output, output, run_id, deadline, image_sha256, candidate):
    result = {'schema': SCHEMA, 'complete': False, 'adopted': False, 'candidate': candidate, 'gpus': [], 'run_id':run_id, 'minimum_saving_us':10}
    try:
        p = read_json(p_selection)
        assert p['adopted'] is True and p['selected'] == 'P', 'P was not adopted'
        identities = {r['uuid'] for r in p['gpus']}
        assert len(identities) == len(p['gpus']) == 12
        seen = set(); headers = set()
        for worker in range(3):
            for rank in range(4):
                r = read_json(directory/f'prefill-{worker}'/f'rank-{rank}.json')
                result['gpus'].append(r)
                assert r['schema'] == SCHEMA and r['run_id'] == run_id and r['worker'] == f'prefill-{worker}' and r['rank'] == rank, 'stale/identity'
                assert r['uuid'] in identities and r['uuid'] not in seen and 'H200' in r['device'], 'GPU UUID set'
                assert r['image_sha256'] == image_sha256 and len(image_sha256) == 64, 'image'
                assert r['finished_at'] <= deadline and 0 < r['elapsed_seconds'] < 150, 'late fragment'
                validate_record(r, candidate)
                seen.add(r['uuid']); headers.add(r['build']['candidate_header_sha256'])
        assert seen == identities and len(headers) == 1, 'incomplete/inconsistent GPUs'
        assert time.time() <= deadline, 'aggregate deadline'
        if candidate:
            lines = env_output.read_text().splitlines(keepends=True)
            assert not any(line.startswith(KEY+'=') for line in lines), 'unexpected prior L selection'
            # P's environment is already committed. Only the L switch is appended.
            tmp = env_output.with_suffix('.megamoe.tmp')
            try:
                tmp.write_text(''.join(lines)+KEY+'=1\n')
                assert time.time() <= deadline, 'aggregate deadline'
                tmp.replace(env_output)
            finally:
                tmp.unlink(missing_ok=True)
        result.update(complete=True, adopted=candidate, reason='all twelve UUIDs passed' if candidate else 'control observation complete')
    except (AssertionError, KeyError, TypeError, ValueError, OSError) as exc:
        result['reason'] = f'{type(exc).__name__}: {exc}'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def run(output, fixture, run_id, deadline, image_sha256, candidate):
    started = time.monotonic()
    rank = int(os.environ['LOCAL_RANK'])
    output.mkdir(parents=True, exist_ok=True)
    record = {'schema':SCHEMA, 'run_id':run_id, 'worker':output.name, 'rank':rank, 'image_sha256':image_sha256, 'candidate':candidate, 'complete':False, 'cases':{}, 'routing_orders':[list(routing_order(i)) for i in range(ROUNDS)]}
    def save():
        record.update(elapsed_seconds=time.monotonic()-started, finished_at=time.time())
        (output/f'rank-{rank}.json').write_text(json.dumps(record, allow_nan=False)+'\n')
    save()
    try:
        run_rank(record, fixture, rank, deadline, candidate, save)
        record['complete'] = True
    except Exception:  # noqa: BLE001 - independent EP1 peers keep their fragments
        record['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        os.environ[KEY] = '0'
        save()


def run_rank(record, fixture, rank, deadline, candidate, save):
    from datetime import timedelta

    import deep_gemm
    import numpy as np
    import torch
    import torch.distributed as dist
    helper = runpy.run_path(str(Path(__file__).with_name('run-prefill-megamoe-diag.py')))
    reference = runpy.run_path(str(Path(__file__).with_name('select-firmus-prefill-megamoe.py')))
    tokens, hidden_size, intermediate, experts, topk = 7988, 4096, 1536, 128, 8
    torch.set_num_threads(1); torch.cuda.set_device(rank)
    torch.cuda.empty_cache()  # P's local validation buffers are no longer live.
    record.update(uuid=str(torch.cuda.get_device_properties(rank).uuid), device=torch.cuda.get_device_name(rank), build=json.loads(Path('/opt/apac/prefill-megamoe-sched-build.json').read_text()), fixture_sha256=__import__('hashlib').sha256(fixture.read_bytes()).hexdigest(), tokens=tokens, routing_is_current_serving=False, seed=201101, input_scales=[.25,10.])
    assert torch.cuda.get_device_capability() == (9,0) and torch.cuda.get_device_properties(rank).multi_processor_count == 132
    device = torch.device('cuda',rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=40), device_id=device)
    assert dist.get_world_size() == 4
    singles = [dist.new_group([r]) for r in range(4)]
    os.environ.update(DG_SM90_PREFILL_2WG='1', DG_SM90_PREFILL_LOCAL_EP1='1', DG_SM90_PREFILL_4STAGE='1', DG_SM90_PREFILL_LEAN='1', DG_SM90_PREFILL_SPLIT_ACCUM='1', DG_SM90_PREFILL_TAIL_PAIR='0')
    cap = 8448
    size, slicer = deep_gemm._C.get_symm_buffer_size_for_sm90_mega_moe(1,experts,cap,topk,hidden_size,intermediate,True,'swiglu')
    buffer = torch.zeros(size, dtype=torch.int8, device=device)
    views = [deep_gemm._from_dlpack_if_needed(v,dtype) for v,dtype in zip(slicer(buffer),(torch.float8_e4m3fn,None,None,None,torch.float8_e4m3fn,None,torch.float8_e4m3fn,None),strict=True)]
    buf = SimpleNamespace(buffer=buffer,group=singles[rank],handle=SimpleNamespace(buffer_ptrs=[buffer.data_ptr()]),num_max_tokens_per_rank=cap,num_experts=experts,num_topk=topk)
    rng = torch.Generator(device=device).manual_seed(201101)
    w13 = (torch.randn(experts,2*intermediate,hidden_size,dtype=torch.bfloat16,device=device,generator=rng)*4).to(torch.float8_e4m3fn)
    w2 = (torch.randn(experts,hidden_size,intermediate,dtype=torch.bfloat16,device=device,generator=rng)*4).to(torch.float8_e4m3fn)
    s13 = torch.rand(experts,2*intermediate//128,hidden_size//128,device=device,generator=rng)*.005+.005
    s2 = torch.rand(experts,hidden_size//128,intermediate//128,device=device,generator=rng)*.005+.005
    l1,l2 = deep_gemm.transform_weights_for_mega_moe_sm90((w13,s13),(w2,s2))
    hidden = [torch.randn(tokens,hidden_size,dtype=torch.bfloat16,device=device,generator=rng)*scale for scale in (.25,10.)]
    weights = torch.rand(tokens,topk,device=device,generator=rng); weights /= weights.sum(-1,keepdim=True)
    cpu_cases = helper['routing_cases'](np,fixture)
    ids = {name:torch.as_tensor(value,dtype=torch.int32,device=device) for name,value in cpu_cases.items()}
    modes = ('control','L') if candidate else ('control',)
    out = {mode:torch.empty_like(hidden[0]) for mode in modes}
    def dispatch(name, replay):
        deep_gemm.mega_moe_pre_dispatch_sm90(hidden[replay],ids[name],weights,*views[:4],num_tokens=tokens,group_size=128,routed_scaling_factor=1.)
    def call(mode):
        os.environ[KEY] = '1' if mode == 'L' else '0'
        deep_gemm.fp8_mega_moe(out[mode],l1,l2,buf,recipe=(128,128,128),activation='swiglu',fast_math=True)
    def equal(a,b):
        return bytes_equal(torch,a,b)
    def verify(mode,target,case):
        case['byte_equal'] = case['byte_equal'] and equal(out[mode],target)
        case['finite'] = case['finite'] and bool(torch.isfinite(out[mode]).all())
        assert case['byte_equal'] and case['finite'], 'byte/finite replay'
    graphs = {}
    expected = {}
    try:
        for name in ROUTINGS:
            case = record['cases'][name] = {'samples_us':[], 'byte_equal':True, 'finite':True, 'control_fp32':[], 'expert_rows':np.bincount(cpu_cases[name].ravel(),minlength=experts).tolist()}
            for replay in range(2):
                dispatch(name,replay)
                call('control'); torch.cuda.synchronize()
                expected[name,replay] = out['control'].clone()
                ref = reference['reference_rows'](torch,hidden[replay][:32],ids[name][:32],weights[:32],w13,s13,w2,s2,0)
                a,b = reference['errors'](torch,out['control'][:32],ref)
                assert math.isfinite(a) and math.isfinite(b) and a <= .025 and b <= .08, 'control reference'
                case['control_fp32'].append({'relative_l2':a,'max_scaled_error':b})
                for mode in modes:
                    call(mode); torch.cuda.synchronize()
                    verify(mode,expected[name,replay],case)
            dispatch(name,0)
            for mode in modes:
                if mode not in graphs:
                    # Host selection is captured separately per mode. The GPU scheduler
                    # executes on every replay and is inside both event endpoints.
                    call(mode); torch.cuda.synchronize()
                    graphs[mode] = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graphs[mode]):
                        for _ in range(REPS): call(mode)
            save()
        del w13,w2
        for round_index in range(ROUNDS):
            assert time.time() < deadline-11, 'reserved endurance/exit time exhausted'
            for name in routing_order(round_index):
                dispatch(name,round_index%2)
                pair = {'round':round_index}
                for order,sequence in (('forward',modes),('reverse',modes[::-1])):
                    pair[order] = {}
                    for mode in sequence:
                        begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record(); graphs[mode].replay(); end.record(); end.synchronize()
                        elapsed = begin.elapsed_time(end)*1000/REPS
                        assert math.isfinite(elapsed) and elapsed > 0, 'invalid event timing'
                        pair[order][mode] = elapsed
                        verify(mode,expected[name,round_index%2],record['cases'][name])
                record['cases'][name]['samples_us'].append(pair)
            save()
        # 600 candidate calls, with routing and input changes, on the same workspace.
        mode = 'L' if candidate else 'control'
        for iteration in range(ENDURANCE):
            assert time.time() < deadline-5, 'endurance deadline'
            name,replay = ROUTINGS[iteration%3],(iteration//3)%2
            dispatch(name,replay); call(mode)
            assert equal(out[mode],expected[name,replay]) and bool(torch.isfinite(out[mode]).all()), 'endurance bytes/finite'
        record['endurance'] = {'calls':ENDURANCE,'bytes_equal':True,'finite':True,'completed':True,'inputs':2,'routings':list(ROUTINGS)}
    finally:
        save()
        dist.destroy_process_group()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--p-selection', type=Path, required=True)
    p.add_argument('--env-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--image-sha256', required=True)
    p.add_argument('--candidate', type=int, choices=(0,1), required=True)
    a = p.parse_args()
    aggregate(a.directory,a.p_selection,a.env_output,a.output,a.run_id,a.deadline,a.image_sha256,bool(a.candidate))
