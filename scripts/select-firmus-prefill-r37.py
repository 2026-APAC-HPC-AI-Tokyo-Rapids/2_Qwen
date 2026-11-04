"""One resident group per worker; publish each candidate before interpreter teardown.

Only published bytes, timings, identity and deadlines decide adoption. Wrapper
exit codes remain evidence, including a timeout after a healthy publication.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

SCHEMA = 'firmus-prefill-r37/v1'
KEYS = {'A': 'DG_SM90_PREFILL_DENSE_FIRST_K', 'B': 'SGLANG_PREFILL_QKNORM_FIXED', 'C': 'SGLANG_PREFILL_ONEPASS_DIV448'}
CALLS, ROUNDS, WORKER_SECONDS, WALL_SECONDS = 1200, 12, 95, 105
STATES = ('initial', 'boundary', 'changed', 'restored')
CASES = {'A': ('7936', '7988', '7989', '8192'), 'B': tuple(f'{n}/{bits}' for n in (31,32,33,7988,7989) for bits in (32,64)), 'C': tuple(map(str, (1,63,64,65,7988,7989,8192)))}
COMPONENTS = {'A': ('qkv', 'o'), 'B': ('writer',), 'C': ('onepass',)}


def read(path):
    def reject(value):
        raise ValueError(f'nonfinite JSON: {value}')
    return json.loads(path.read_text(), parse_constant=reject)


def medians(samples):
    assert set(samples) == {'forward', 'reverse'}, 'timing orders'
    rows = {}
    for order, row in samples.items():
        assert set(row) == {'control', 'candidate'}, 'timing modes'
        for values in row.values():
            assert len(values) == ROUNDS and all(type(v) in (int,float) and math.isfinite(v) and v > 0 for v in values), 'timing samples'
        rows[order] = {m: statistics.median(v) for m,v in row.items()}
    rows['pooled'] = {m: statistics.median(samples['forward'][m] + samples['reverse'][m]) for m in ('control','candidate')}
    return rows


def speed_rule(timings):
    assert len(timings) == 12, 'twelve GPUs required'
    rows = [medians(s) for s in timings]
    savings = {o: [r[o]['control'] - r[o]['candidate'] for r in rows] for o in ('forward','reverse','pooled')}
    fractions = [v/r['pooled']['control'] for v,r in zip(savings['pooled'],rows,strict=True)]
    summary = {'saving_median_us': {o: statistics.median(v) for o,v in savings.items()}, 'positive_gpus': sum(v > 0 for v in savings['pooled']), 'worst_fraction': min(fractions)}
    assert all(v > 0 for v in summary['saving_median_us'].values()), 'nonpositive pooled/order median'
    assert summary['positive_gpus'] >= 9, 'fewer than nine positive GPUs'
    assert summary['worst_fraction'] >= -.01, 'GPU loss exceeds one percent'
    return summary


def combined_projection_samples(timing):
    # Sum paired samples within each GPU, then pool the two orders within that GPU.
    return {o: {m: [sum(values) for values in zip(*(timing[p][o][m] for p in COMPONENTS['A']), strict=True)] for m in ('control','candidate')} for o in ('forward','reverse')}


def check_record(r, candidate, worker, rank, run_id, image, identity, deadline):
    assert r['schema'] == SCHEMA and r['run_id'] == run_id and r['candidate'] == candidate, 'stale candidate'
    assert r['worker'] == worker and r['rank'] == rank and r['uuid'].removeprefix('GPU-') in identity and 'H200' in r['device'], 'wrong GPU/rank'
    assert r['image_sha256'] == image and r['healthy'] is True, 'wrong image or unhealthy'
    assert 0 < r['elapsed_seconds'] < WORKER_SECONDS and r['finished_at'] <= deadline - (WALL_SECONDS-WORKER_SECONDS), 'late worker'
    assert set(r['exact']) == set(CASES[candidate]), 'missing exact case'
    for case in r['exact'].values():
        assert set(case) == set(STATES) and all(v is True for v in case.values()), 'intermediate/final byte mismatch'
    e = r['endurance']
    assert e['calls'] == CALLS and e['states'] == ['initial','changed'] and e['calls_per_state'] == [600,600], 'incomplete changed-state endurance'
    assert e['buffers'] == 2, 'missing changed output buffer'
    assert e['every_call_equal'] is True and e['restored_equal'] is True, 'endurance byte mismatch'
    assert set(r['timing']) == set(COMPONENTS[candidate]), 'missing timed component'
    for samples in r['timing'].values():
        medians(samples)
    assert len(r['source_sha256']) == 64 and len(r['snapshot_sha256']) == 64, 'missing source/input provenance'


def aggregate(directory, env_output, prior_path, run_id, image, deadline, exit_codes):
    result = {'schema': SCHEMA, 'run_id': run_id, 'worker_exit_codes': exit_codes, 'candidates': {}, 'rule': {'calls_per_candidate': CALLS, 'rounds_per_order': ROUNDS, 'worker_seconds': WORKER_SECONDS, 'total_seconds': WALL_SECONDS}}
    lines = env_output.read_text().splitlines()
    for key in KEYS.values():
        assert [s for s in lines if s.startswith(key+'=')] == [key+'=0'], 'expected unique disabled flag'
    for candidate, key in KEYS.items():
        decision = result['candidates'][candidate] = {'adopted': False, 'gpus': []}
        try:
            assert time.time() <= deadline, 'aggregate budget expired'
            prior = read(prior_path)
            # Slurm can reassign worker/rank GPUs between gate steps. Match all twelve UUIDs.
            identities = {g['uuid'].removeprefix('GPU-') for g in prior['gpus']}
            assert len(identities) == len(prior['gpus']) == 12, 'incomplete prerequisite inventory'
            assert all(g['run_id'].split('-',1)[0] == run_id.split('-',1)[0] for g in prior['gpus']), 'wrong allocation'
            sources = set(); seen = set()
            for worker in range(3):
                for rank in range(4):
                    name = f'prefill-{worker}'
                    r = read(directory/name/f'{candidate}-rank-{rank}.json')
                    decision['gpus'].append(r)
                    check_record(r,candidate,name,rank,run_id,image,identities,deadline)
                    assert r['uuid'] not in seen, 'duplicate GPU'
                    seen.add(r['uuid'])
                    sources.add((r['source_sha256'], r['snapshot_sha256']))
            assert seen == identities and len(sources) == 1, 'inconsistent source/snapshot or missing GPU'
            if candidate == 'A':
                # Individual projections may be flat, but neither may lose over 1% on any GPU.
                for r in decision['gpus']:
                    for samples in r['timing'].values():
                        v = medians(samples)
                        assert all(v[o]['candidate'] <= v[o]['control'] * 1.01 for o in v), 'projection loss exceeds one percent'
                timing = [combined_projection_samples(r['timing']) for r in decision['gpus']]
            else:
                timing = [r['timing'][COMPONENTS[candidate][0]] for r in decision['gpus']]
            decision['timing_summary'] = speed_rule(timing)
            assert time.time() <= deadline, 'aggregate budget expired'
            decision.update(adopted=True, reason='twelve published healthy results passed byte/endurance and pooled speed rules')
        except (OSError,AssertionError,KeyError,ValueError,TypeError) as error:
            decision['reason'] = f'{type(error).__name__}: {error}'
    # Deadline also guards the atomic serving environment change.
    if time.time() > deadline:
        for decision in result['candidates'].values():
            decision.update(adopted=False,reason='aggregate budget expired')
    values = {key: str(int(result['candidates'][c]['adopted'])) for c,key in KEYS.items()}
    temporary = env_output.with_name(env_output.name+'.r37.tmp')
    try:
        temporary.write_text('\n'.join(next((k+'='+v for k,v in values.items() if s.startswith(k+'=')),s) for s in lines)+'\n')
        assert time.time() <= deadline or all(v == '0' for v in values.values()), 'aggregate budget expired'
        temporary.replace(env_output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def dense_snapshot(tensors):
    """Match SM90 TP1 Fp8LinearMethod's parameter allocation and loader copies.

    Upstream fc275674: quantization/fp8.py:558-594 allocates row-major E4M3
    weights and FP32 block scales; linear.py:1017-1033,1143-1161 packs Q/K/V
    on dim 0 and parameter.py:247-274 copies into those parameters. There is
    no standalone conversion helper: the cast happens in Tensor.copy_ during
    loading. On SM90 model_loader/utils.py:267-272 disables UE8M0 requant.
    """
    import torch

    def loaded(projection, suffix):
        raw = tensors[f'self_attn.{projection}.{suffix}']
        dtype = torch.float32 if suffix == 'weight_scale_inv' else torch.float8_e4m3fn
        return torch.empty(raw.shape, dtype=dtype, device=raw.device).copy_(raw)

    qkv = tuple(torch.cat([loaded(p, s) for p in ('q_proj','k_proj','v_proj')], dim=0) for s in ('weight','weight_scale_inv'))
    o = tuple(loaded('o_proj', s) for s in ('weight','weight_scale_inv'))
    return [qkv, o]


def snapshot_weights(model):
    """Read only the first layer's small dense/norm/router tensors, never the MoE weights."""
    import torch
    from safetensors import safe_open
    index = read(model/'model.safetensors.index.json')['weight_map']
    prefix = 'model.layers.0.'
    names = [prefix+'self_attn.'+p+'.'+suffix for p in ('q_proj','k_proj','v_proj','o_proj') for suffix in ('weight','weight_scale_inv')]
    names += [prefix+s for s in ('self_attn.q_norm.weight','self_attn.k_norm.weight','post_attention_layernorm.weight','mlp.gate.weight')]
    tensors = {}
    digest = hashlib.sha256()
    for file in sorted({index[name] for name in names}):
        with safe_open(model/file, framework='pt', device='cpu') as f:
            for name in sorted(n for n in names if index[n] == file):
                tensor = f.get_tensor(name)
                digest.update(name.encode()); digest.update(tensor.view(torch.uint8).numpy().tobytes())
                tensors[name.removeprefix(prefix)] = tensor.cuda()
    dense = dense_snapshot(tensors)
    config = read(model/'config.json'); eps = config['rms_norm_eps']
    angles = torch.arange(8192,device='cuda').float()[:,None] * (config['rope_theta'] ** (-torch.arange(0,128,2,device='cuda').float()/128))
    trig = torch.cat((angles.cos(),angles.sin()),dim=-1).contiguous()
    return {'A': dense, 'B': (tensors['self_attn.q_norm.weight'],tensors['self_attn.k_norm.weight'],trig,eps), 'C': (tensors['post_attention_layernorm.weight'],tensors['mlp.gate.weight'].to(torch.bfloat16),eps)}, digest.hexdigest()


def capture(torch, call):
    call(); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph


def equal(torch, left, right):
    assert len(left) == len(right)
    return all(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)) for a,b in zip(left,right,strict=True))


def exact(torch, case, key):
    record = {}; initial = None
    for state in STATES:
        case.state('initial' if state == 'restored' else state)
        values = {}
        for mode in ('control','candidate'):
            os.environ[key] = str(int(mode == 'candidate'))
            case.step(mode)
            values[mode] = [x.clone() for x in (*case.outputs(mode),*case.final(mode))]
        assert equal(torch,values['control'],values['candidate']), f'{state}: intermediate/final mismatch'
        if state == 'initial':
            initial = values['candidate']
        if state == 'restored':
            assert equal(torch,initial,values['candidate']), 'restore mismatch'
        record[state] = True
    return record


def endurance(torch, case, key):
    # Control is reused per state; each candidate still runs 1200 times across two output buffers.
    initial = None
    failed = torch.zeros((),device='cuda',dtype=torch.bool)
    outputs = [case.out['candidate'], [x.clone() for x in case.out['candidate']]]
    os.environ[key] = '0'
    control = capture(torch,lambda: case.step('control'))
    os.environ[key] = '1'
    graphs = []
    for buffer in outputs:
        case.out['candidate'] = buffer
        graphs.append(capture(torch,lambda: case.step('candidate')))
    for state in ('initial','changed'):
        case.state(state); control.replay()
        refs = [x.clone() for x in case.outputs('control')]
        if state == 'initial':
            initial = refs
        checkers = []
        for buffer in outputs:
            case.out['candidate'] = buffer
            actual = case.outputs('candidate')
            def check(refs=refs, actual=actual):
                for a,b in zip(refs,actual,strict=True):
                    failed.logical_or_(torch.any(a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)))
            checkers.append(capture(torch,check))
        failed.zero_()
        for i in range(CALLS//2):
            graphs[i%2].replay(); checkers[i%2].replay()
        torch.cuda.synchronize()
        assert not failed.item(), 'transient endurance mismatch'
    case.state('initial')
    for i, buffer in enumerate(outputs):
        graphs[i].replay(); torch.cuda.synchronize()
        case.out['candidate'] = buffer
        assert equal(torch,initial,case.outputs('candidate')), 'endurance restore mismatch'
    case.out['candidate'] = outputs[0]
    return {'calls': CALLS, 'buffers': 2, 'states': ['initial','changed'], 'calls_per_state': [600,600], 'every_call_equal': True, 'restored_equal': True}


def timing(torch, case, key, projection=None):
    graphs = {}
    for mode in ('control','candidate'):
        os.environ[key] = str(int(mode == 'candidate'))
        step = (lambda mode=mode: case.step(mode,projection)) if projection is not None else (lambda mode=mode: case.step(mode))
        graphs[mode] = capture(torch,step)
    samples = {o: {m: [] for m in graphs} for o in ('forward','reverse')}
    # Alternate both orders every round; twelve samples for each order, on ordinary kernels.
    for _ in range(ROUNDS):
        for order,modes in (('forward',('control','candidate')),('reverse',('candidate','control'))):
            for mode in modes:
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(8):
                    graphs[mode].replay()
                end.record(); end.synchronize()
                samples[order][mode].append(start.elapsed_time(end)*1000/8)
    return samples


def publish(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n'); temporary.replace(path)
    print(json.dumps(record,allow_nan=False),flush=True)


def prepare_compilers(ops, weights, rank):
    """Overlap only cold CPU JIT work; join before endurance or any event timing."""
    from concurrent.futures import ThreadPoolExecutor

    import deep_gemm  # noqa: F401 - finish A imports before starting B/C
    import torch
    import triton
    import triton.language  # noqa: F401 - finish shared imports on main
    from sglang.kernels.ops.attention.rope import _jit_rope_fp8_q_module
    from sglang.kernels.ops.moe.prefill_moe_input import (
        _onepass_moe_input,
        _onepass_moe_input_div448,
    )
    from sglang.kernels.ops.quantization import fp8_kernel  # noqa: F401 - A state/final

    def writer():
        torch.cuda.set_device(rank)
        return _jit_rope_fp8_q_module()

    def onepass():
        torch.cuda.set_device(rank)
        case = ops.Onepass(7988, weights['C'])
        r,q,scale,ids,tw = case.out['candidate']
        # All boundary rows map to one of these production scalar specialization keys:
        # equal-to-one, divisible-by-sixteen, or ordinary i32. Warmup never launches.
        for kernel in (_onepass_moe_input, _onepass_moe_input_div448):
            for rows in (1,64,7988):
                kernel.warmup(case.x,r,case.w,case.g,q,scale,ids,tw,rows,case.eps,BLOCK_M=64,num_warps=8,num_stages=3,grid=(128,))
        torch.cuda.synchronize()

    pool = ThreadPoolExecutor(max_workers=2)
    return pool, [pool.submit(writer), pool.submit(onepass)]


def run(args):
    import torch
    from sglang.srt.layers import prefill_r37 as ops
    started = time.monotonic(); rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 4
    torch.set_num_threads(1); torch.cuda.set_device(rank); torch.manual_seed(37004)
    args.output.mkdir(parents=True,exist_ok=True)
    common = {'schema': SCHEMA, 'run_id': args.run_id, 'worker': args.output.name, 'rank': rank, 'image_sha256': args.image_sha256, 'uuid': str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-'), 'device': torch.cuda.get_device_name(rank)}
    weights, digest = snapshot_weights(args.model_path)
    import deep_gemm
    import sglang
    source = hashlib.sha256()
    package = Path(sglang.__file__).parent
    for path in (Path(ops.__file__), package/'kernels/jit/csrc/elementwise/rope.cuh', package/'kernels/ops/moe/prefill_moe_input.py', Path(deep_gemm.__file__).parent/'include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh'):
        source.update(path.name.encode()); source.update(path.read_bytes())
    common.update(snapshot_sha256=digest,source_sha256=source.hexdigest())
    # B/C CPU compilation overlaps A's cold compilation and byte checks.
    pool, pending = prepare_compilers(ops, weights, rank)
    compiler_errors = {}
    for candidate, factory in (('A',ops.Dense),('B',ops.Writer),('C',ops.Onepass)):
        r = {**common,'candidate':candidate,'healthy':False,'exact':{},'timing':{}}
        key = KEYS[candidate]
        try:
            if candidate in compiler_errors:
                raise compiler_errors[candidate]
            for case_name in CASES[candidate]:
                assert time.time() < args.deadline, 'worker budget expired'
                rows = int(case_name.split('/')[0])
                case = factory(rows,weights[candidate],torch.int32 if case_name.endswith('/32') else torch.int64) if candidate == 'B' else factory(rows,weights[candidate])
                r['exact'][case_name] = exact(torch,case,key)
                del case
            if candidate == 'A' and pending:
                for name, future in zip(('B','C'), pending, strict=True):
                    try:
                        future.result()
                    except Exception as error:  # noqa: BLE001 - preserve independent compile failure
                        compiler_errors[name] = error
                pool.shutdown(wait=True)
                pending = []
            case = factory(7988,weights[candidate])
            r['endurance'] = endurance(torch,case,key)
            case.state('initial')
            for i,component in enumerate(COMPONENTS[candidate]):
                r['timing'][component] = timing(torch,case,key,i if candidate == 'A' else None)
            torch.cuda.synchronize()
            r['healthy'] = True
            del case
        except Exception as error:  # noqa: BLE001 - persist failure independently for each candidate
            r.update(healthy=False,error=f'{type(error).__name__}: {error}')
        finally:
            if candidate == 'A' and pending:
                for name, future in zip(('B','C'), pending, strict=True):
                    try:
                        future.result()
                    except Exception as error:  # noqa: BLE001 - preserve independent compile failure
                        compiler_errors[name] = error
                pool.shutdown(wait=True)
                pending = []
            os.environ[key] = '0'
        r.update(elapsed_seconds=time.monotonic()-started,finished_at=time.time())
        publish(args.output/f'{candidate}-rank-{rank}.json',r)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--run-id',required=True)
    p.add_argument('--image-sha256',required=True); p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--model-path',type=Path); p.add_argument('--aggregate',type=Path)
    p.add_argument('--env-output',type=Path); p.add_argument('--pair-selection',type=Path)
    p.add_argument('--worker-exit-codes',type=int,nargs=3,default=[])
    args = p.parse_args()
    if args.aggregate:
        result = aggregate(args.aggregate,args.env_output,args.pair_selection,args.run_id,args.image_sha256,args.deadline,args.worker_exit_codes)
        publish(args.output,result)
    else:
        run(args)


if __name__ == '__main__':
    main()
