"""Read-only reconstruction of request scores, one real trace window, and bounds."""
import argparse
import collections
import datetime
import hashlib
import json
import math
import pathlib
import re
import sqlite3
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifacts', required=True, type=pathlib.Path, help='Directory containing raw/<run>/ and prefill-2.sqlite')
parser.add_argument('--output', required=True, type=pathlib.Path)
args = parser.parse_args()
ROOT = args.artifacts


def logged_periods(run):
    """Recover logged PP0 periods, including the token-count convention used there.

    The logger's 8064 page-padded tokens belong in this inverse of its throughput
    field. They do not replace 7988 model rows in FLOP or payload accounting.
    """
    result = []
    for worker in range(3):
        path = next((ROOT / 'raw' / str(run)).rglob(f'prefill-{worker}.log'))
        samples = []
        # Concurrent worker progress bars can interleave non-UTF8 bytes. The
        # selected ASCII Prefill records remain intact; hash the untouched log.
        for line in path.read_text(errors='replace').splitlines():
            if ' PP0] Prefill batch, #new-seq: 2,' not in line:
                continue
            tokens = int(re.search(r'#new-token: (\d+)', line)[1])
            throughput = float(re.search(r'input throughput \(token/s\): ([0-9.]+)', line)[1])
            if throughput > 0:
                samples.append(1000 * tokens / throughput)
        result.append({'worker': worker, 'samples': len(samples), 'median_ms': statistics.median(samples),
                       'log_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


def analyze_run(run):
    period_evidence = logged_periods(run)
    periods = [r['median_ms'] for r in period_evidence]
    directory = next((ROOT / 'raw' / str(run)).rglob('score.json')).parent
    score = json.loads((directory / 'score.json').read_text())
    exported = json.loads((directory / 'profile_export_aiperf.json').read_text())
    records = []
    with (directory / 'profile_export.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row['metadata'].get('phase_kind') == 'profiling':
                records.append(row)
    value = lambda row, name: row['metrics'][name]['value']
    first = min(row['metadata']['request_start_ns'] for row in records)
    bad_ttft = [r for r in records if value(r, 'time_to_first_token') > 3000]
    bad_itl = [r for r in records if value(r, 'inter_token_latency') > 20]
    good = sum(value(r, 'good_request_count') for r in records)
    # The official parser consumes AIPerf's goodput, whose denominator includes
    # the profiling phase's endpoints. Request-span benchmark_duration differs
    # by a few milliseconds and must not silently replace that denominator.
    rounded_duration = (datetime.datetime.fromisoformat(exported['end_time']) - datetime.datetime.fromisoformat(exported['start_time'])).total_seconds()
    duration = good / exported['goodput']['avg']
    assert abs(duration-rounded_duration) < 2e-6
    recovered_score = exported['goodput']['avg'] * 200
    assert abs(recovered_score - score['good_output_tokens_per_second']) < 1e-8
    rate = sum(2000 / t for t in periods)
    lo, hi = 0.0, min(periods) / 2
    for _ in range(60):
        mid = (lo + hi) / 2
        modeled = recovered_score * sum(2000 / (t-mid) for t in periods) / rate
        if modeled < 10000:
            lo = mid
        else:
            hi = mid
    return {
        'run': run, 'official_score': recovered_score, 'duration_s': duration,
        'request_span_duration_s': exported['benchmark_duration']['avg'],
        'requests': len(records), 'good_requests': good, 'bad_ttft': len(bad_ttft),
        'bad_itl': len(bad_itl), 'bad_both': sum(r in bad_itl for r in bad_ttft),
        'bad_ttft_request_start_range_s': [min((r['metadata']['request_start_ns']-first)/1e9 for r in bad_ttft), max((r['metadata']['request_start_ns']-first)/1e9 for r in bad_ttft)],
        'additional_good_requests_at_fixed_duration': math.ceil(50*duration)-good,
        'measured_pp_periods_ms': periods, 'conditional_prefill_capacity_rps': rate,
        'period_evidence': period_evidence,
        'uniform_period_reduction_for_10k_ms': hi,
        'assumptions': 'Fixed good fraction/drain/Decode response; all three saturated Prefill pipelines improve by the same number of milliseconds. A conditional capacity model, not a score prediction.'
    }


def category(name):
    if 'mega_moe_impl' in name:
        return 'MegaMoE'
    if 'apac_fa3_private' in name:
        return 'FA3'
    if 'sm90_fp8_gemm_1d2d_impl' in name and '(unsigned int)9216, (unsigned int)4096' in name:
        return 'QKV'
    if 'sm90_fp8_gemm_1d2d_impl' in name and '(unsigned int)4096, (unsigned int)8192' in name:
        return 'O'
    if 'qknorm_rope_prefill_fp32_kernel' in name:
        return 'QK_post'
    if name == 'norm_quant':
        return 'norm_quant'
    if name == '_onepass_moe_input':
        return 'onepass_moe_input'
    return 'other'


def trace():
    c = sqlite3.connect(f'file:{ROOT}/prefill-2.sqlite?mode=ro', uri=True)
    assert c.execute('pragma quick_check').fetchone()[0] == 'ok'
    rows = [(s, e, category(n), gx) for s, e, n, gx in c.execute('select k.start,k.end,s.value,k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where deviceId=2 order by start')]
    qkv = [r for r in rows if r[2] == 'QKV']
    assert len(qkv) == 45 * 24
    selected = list(range(2, 12)) + [15, 19, 27, 31, 35, 36, 37, 38]
    windows = []
    for i in selected:
        start, end = qkv[24*i][0], qkv[24*(i+1)][0]
        assert 216277184 <= start < end <= 5232131438
        intervals = []
        totals = collections.defaultdict(int)
        for s, e, name, gx in rows:
            a, b = max(start, s), min(end, e)
            if b > a:
                intervals.append((a, b))
                totals[name] += b-a
        merged = []
        for s, e in sorted(intervals):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(e, merged[-1][1])
            else:
                merged.append([s, e])
        union = sum(e-s for s, e in merged)
        kernel_sum = sum(totals.values())
        windows.append({'batch': i, 'start_ns': start, 'end_ns': end,
                        'period_ms': (end-start)/1e6,
                        'kernel_ms': {k:v/1e6 for k,v in totals.items()},
                        'overlap_ms': (kernel_sum-union)/1e6,
                        'no_kernel_ms': (end-start-union)/1e6})
    representative = next(x for x in windows if x['batch'] == 37)
    representative['memcpy'] = [dict(zip(['bytes','us','label','source_device','destination_device'], row)) for row in c.execute('select m.bytes,(m.end-m.start)/1e3,e.label,m.srcDeviceId,m.dstDeviceId from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where m.deviceId=2 and m.start>=? and m.end<=? and m.bytes>1000000', (representative['start_ns'],representative['end_ns']))]
    return {'run':40964714831, 'not_current_serving_profile':True,
            'device_ordinal':2, 'physical_gpu':3, 'selected_windows':windows,
            'median_period_ms':statistics.median(x['period_ms'] for x in windows),
            'median_no_kernel_ms':statistics.median(x['no_kernel_ms'] for x in windows),
            'representative':representative}


def bounds():
    length, tokens, hidden, intermediate = 3994, 7988, 4096, 1536
    peak = 132*8192*1.4e9
    flops = {'MegaMoE':6*tokens*8*hidden*intermediate,
             'FA3':2*2*length*(length+1)*64*128,
             'QKV':2*tokens*hidden*9216, 'O':2*tokens*8192*hidden}
    tensor = tokens*hidden*2
    kv = 94*length*4*128*2
    return {'scenario':'Conditional H200 dense FP8 at 1.40 GHz; no sparsity multiplier',
            'peak_flops':peak, 'hbm_bytes_per_second_reference':4.8e12,
            'per_layer':{k:{'useful_flops':f,'tensor_only_floor_ms':1000*f/peak} for k,f in flops.items()},
            'moe_cold_expert_weight_bytes_per_layer':3*hidden*intermediate*128,
            'pp_tensor_bytes':tensor, 'nvlink_one_direction_reference_bytes_per_second':450e9,
            'pp_boundary_tensor_counts':[3,4,4],
            'pp_boundary_floor_ms':[1000*n*tensor/450e9 for n in [3,4,4]],
            'pp_total_bytes_per_pair':11*tensor,
            'kv_cache_bytes_per_request_unpadded':kv,
            'kv_payload_50GBps_reference_ms':1000*kv/50e9,
            'kv_reference_is_not_measured_network_bandwidth':True,
            'accounting':'Use max(F/P,B/W) per overlapping resource domain; do not sum four pipeline stage times for throughput, or add already-overlapped transfers to kernel durations.'}


result = {'runs':[analyze_run(run) for run in (41150387984, 41166990387, 41201412459)], 'profile':trace(), 'bounds':bounds()}
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'runs':result['runs'],'profile':{k:v for k,v in result['profile'].items() if k!='selected_windows'},'bounds':result['bounds']},indent=2))
