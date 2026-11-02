#!/usr/bin/env python3
"""PP4 pair-cycle accounting from a read-only Nsight SQLite export.

Cycles rotate from every Lth target MegaMoE to the next. They are not request
boundaries. In a steady pair window this includes one forward tail/prologue.
No timestamps or clocks from other runs enter the accounting.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import datetime as dt
import itertools
import json
import re
import sqlite3
import statistics
from pathlib import Path

CATEGORIES = ('MegaMoE', 'FA3', 'QKV/O GEMM', 'writer/pack', 'norm', 'router/gate', 'MoE input', 'PP send/recv or copy', 'lm_head/draft', 'other')
GATE_US = {'MegaMoE': (2334, 2500), 'FA3': (621, 621), 'QKV GEMM': (537, 537), 'O GEMM': (459, 459), 'writer/pack': (76, 76)}


def records(db, table):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
        return []
    return [dict(r) for r in db.execute(f'SELECT * FROM "{table}"')]


def merge(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def duration(intervals):
    return sum(b - a for a, b in merge(intervals))


def distribution(values):
    if not values:
        return {'count': 0, 'median': None, 'p10': None, 'p90': None, 'max': None}
    ordered = sorted(values)
    def quantile(p):
        pos = (len(ordered) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return {'count': len(values), 'median': statistics.median(values), 'p10': quantile(.1), 'p90': quantile(.9), 'max': max(values)}


def target_qkv(name):
    return 'gemm_1d2d' in name and bool(re.search(r'(?:unsigned int\)?\s*)?9216.*4096', name))


def category(name):
    lower = name.lower()
    if 'mega_moe_impl' in name:
        return 'MegaMoE'
    if 'nccl' in lower or 'sendrecv' in lower:
        return 'PP send/recv or copy'
    if 'gemm_1d2d' in name:
        return 'QKV/O GEMM'
    if any(x in lower for x in ('qknorm_rope', 'prepack', 'v_pack', 'pack_v')):
        return 'writer/pack'
    if any(x in lower for x in ('pre_dispatch', 'moe_input')):
        return 'MoE input'
    if any(x in lower for x in ('flashattn', 'prepare_varlen', 'combine_fixed')):
        return 'FA3'
    if any(x in lower for x in ('rmsnorm', 'norm_quant', 'group_quant')):
        return 'norm'
    if any(x in lower for x in ('router', 'gate', 'nvjet', 'splitkreduce')):
        return 'router/gate'
    return 'other'


def load(db):
    db.row_factory = sqlite3.Row
    strings = dict(db.execute('SELECT id,value FROM StringIds'))
    kernels = records(db, 'CUPTI_ACTIVITY_KIND_KERNEL')
    for row in kernels:
        row['name'] = strings[row['demangledName']]
        row['category'] = category(row['name'])
        row['kind'] = 'kernel'
    copies = records(db, 'CUPTI_ACTIVITY_KIND_MEMCPY')
    for row in copies:
        row.update(name='memcpy', category='PP send/recv or copy', kind='copy')
    memsets = records(db, 'CUPTI_ACTIVITY_KIND_MEMSET')
    for row in memsets:
        row.update(name='memset', category='other', kind='copy')
    copies += memsets
    apis = records(db, 'CUPTI_ACTIVITY_KIND_RUNTIME') + records(db, 'CUPTI_ACTIVITY_KIND_DRIVER')
    by_correlation = collections.defaultdict(list)
    for row in apis:
        row['name'] = strings[row['nameId']]
        # Nsight GlobalId: low 24 bits are thread ID, the upper bits identify process/VM.
        row['globalPid'] = row['globalTid'] & ~0xffffff
        by_correlation[row['globalPid'], row['correlationId']].append(row)
    for event in kernels + copies:
        candidates = by_correlation.get((event.get('globalPid'), event.get('correlationId')), [])
        prior = [r for r in candidates if r['start'] <= event['start']]
        event['api'] = max(prior, key=lambda r: r['start']) if prior else None
    return kernels, copies, apis


def tail_categories(kernels, stage):
    """Identify the PP3 tail from its LM-head GEMM through next target QKV.

    Known c7804efb/cuBLAS names only. Unrecognised names remain visible in other.
    NCCL is retained as transfer even in the tail. This is not shape inference.
    """
    if stage != 3:
        return
    in_tail = False
    for row in kernels:
        if re.search(r'nvjet[^ ]*(?:512x|384x)|lm_head', row['name']):
            in_tail = True
        if target_qkv(row['name']):
            in_tail = False
        if in_tail and row['category'] != 'PP send/recv or copy':
            row['category'] = 'lm_head/draft'


def capture_window(db, log, guard_ns):
    epoch = records(db, 'TARGET_INFO_SESSION_START_TIME')[0]['utcEpochNs']
    text = log.read_text()
    def stamp(label):
        match = re.search(rf'^{label} utc=(\S+)', text, re.MULTILINE)
        if not match:
            raise ValueError(f'missing {label} in {log}')
        return round(dt.datetime.fromisoformat(match[1].replace('Z', '+00:00')).timestamp() * 1e9)
    return stamp('capture_active') - epoch + guard_ns, stamp('stop_requested') - epoch - guard_ns


def telemetry_samples(paths, gpu_map, epoch, timezone_hours):
    """Explicit logical-device order from worker gpu-map; historical maps supplied separately."""
    uuids = [row[0].strip().removeprefix('GPU-') for row in csv.reader(gpu_map.open()) if row]
    output = collections.defaultdict(list)
    zone = dt.timezone(dt.timedelta(hours=timezone_hours))
    for path in paths:
        with path.open() as handle:
            for row in csv.reader(handle):
                if not row or row[0].strip() == 'timestamp':
                    continue
                identity = row[2].strip().removeprefix('GPU-')
                if identity not in uuids:
                    continue
                time = dt.datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=zone)
                output[uuids.index(identity)].append({'timestamp': round(time.timestamp() * 1e9) - epoch, 'sm_clock_mhz': float(row[4].split()[0]), 'power_w': float(row[3].split()[0]), 'throttle_reasons': row[6].strip()})
    for values in output.values():
        values.sort(key=lambda r: r['timestamp'])
    return output


def gap_evidence(a, b, events, kernels, apis):
    event_starts, event_rows, kernel_starts, kernel_rows, kernel_ends, ended_rows = events
    i = bisect.bisect_right(kernel_ends, a) - 1
    before = ended_rows[i] if i >= 0 else None
    i = bisect.bisect_left(kernel_starts, b)
    after = kernel_rows[i] if i < len(kernel_rows) else None
    i = bisect.bisect_left(event_starts, b)
    following = event_rows[i] if i < len(event_rows) else None
    api = following.get('api') if following else None
    waits = [r for r in apis if before and r['globalPid'] == before.get('globalPid') and r['start'] < b and r['end'] > a and any(x in r['name'] for x in ('Synchronize', 'WaitEvent', 'WaitValue'))]
    cause = 'unresolved dependency'
    exposed = None
    if following and following['kind'] == 'copy':
        cause = 'transfer wait candidate'
    elif waits:
        cause = 'synchronization wait candidate'
    elif api and api['start'] > a:
        cause = 'late launch candidate'
        exposed = min(b, api['start']) - a
    elif api and api['end'] <= a:
        cause = 'submitted before gap; dependency or device wait'
    return {'start_ns': a, 'end_ns': b, 'duration_us': (b-a)/1e3, 'before_kernel': before.get('demangledName', before['name']) if before else None, 'after_kernel': after.get('demangledName', after['name']) if after else None, 'cause': cause, 'pre_api_idle_us': exposed / 1e3 if exposed is not None else None, 'next_api': {k: api[k] for k in ('name', 'start', 'end', 'correlationId', 'globalPid')} if api else None, 'synchronization_apis': [r['name'] for r in waits]}


def account(lo, hi, kernels, copies, apis, samples):
    ks = [r for r in kernels if r['start'] < hi and r['end'] > lo]
    cs = [r for r in copies if r['start'] < hi and r['end'] > lo]
    events = ks + cs
    spans = [(max(lo, r['start']), min(hi, r['end'])) for r in events]
    busy = merge(spans)
    kernel_spans = merge((max(lo, r['start']), min(hi, r['end'])) for r in ks)
    all_events = sorted(kernels + copies, key=lambda r: r['start'])
    ended = sorted(kernels, key=lambda r: r['end'])
    index = ([r['start'] for r in all_events], all_events, [r['start'] for r in kernels], kernels, [r['end'] for r in ended], ended)
    processes = {r.get('globalPid') for r in kernels}
    syncs = [r for r in apis if r['globalPid'] in processes and r['start'] < hi and r['end'] > lo and any(x in r['name'] for x in ('Synchronize', 'WaitEvent', 'WaitValue'))]
    gaps = []
    cursor = lo
    for a, b in busy + [(hi, hi)]:
        if a > cursor:
            gaps.append(gap_evidence(cursor, a, index, kernels, syncs))
        cursor = max(cursor, b)
    summed = collections.Counter()
    by_category = collections.defaultdict(list)
    changes = collections.defaultdict(list)
    for row in events:
        a, b = max(lo, row['start']), min(hi, row['end'])
        by_category[row['category']].append((a, b))
        if row['kind'] == 'kernel':
            summed[row['category']] += (b-a)/1e3
        changes[a].append((row['category'], 1))
        changes[b].append((row['category'], -1))
    active = collections.Counter()
    exclusive = collections.Counter()
    prev = lo
    for timestamp, updates in sorted(changes.items()):
        labels = [k for k, v in active.items() if v > 0]
        if labels:
            exclusive[labels[0] if len(labels) == 1 else 'overlap'] += (timestamp-prev)/1e3
        for label, delta in updates:
            active[label] += delta
        prev = timestamp
    compute_ns = duration((max(lo, r['start']), min(hi, r['end'])) for r in ks if r['category'] != 'PP send/recv or copy')
    # Kernel-conditioned samples only. Missing samples stay null, including short cycles.
    starts = [a for a, _ in kernel_spans]
    def running(t):
        i = bisect.bisect_right(starts, t) - 1
        return i >= 0 and t < kernel_spans[i][1]
    measured = [s for s in samples if running(s['timestamp'])]
    launches = [r for r in ks if r.get('api')]
    waits = [(r['start'] - r['api']['end'])/1e3 for r in launches]
    causes = collections.Counter()
    for gap in gaps:
        causes[gap['cause']] += gap['duration_us']
    return {'gap_cause_us': dict(causes), 'pre_api_idle_lower_bound_us': sum(g['pre_api_idle_us'] for g in gaps if g['pre_api_idle_us'] is not None), 'start_ns': lo, 'end_ns': hi, 'period_us': (hi-lo)/1e3, 'summed_kernel_us': {k: summed.get(k, 0) for k in CATEGORIES}, 'category_union_us': {k: duration(v)/1e3 for k, v in by_category.items()}, 'exclusive_busy_us': dict(exclusive), 'compute_kernel_union_us': compute_ns/1e3, 'nccl_only_us': (duration(kernel_spans)-compute_ns)/1e3, 'kernel_union_us': duration(kernel_spans)/1e3, 'copy_only_us': (duration(spans)-duration(kernel_spans))/1e3, 'gpu_idle_us': sum(g['duration_us'] for g in gaps), 'gaps': gaps, 'gap_distribution_us': distribution([g['duration_us'] for g in gaps]), 'launch_to_gpu_after_api_us': distribution(waits), 'launch_api_matched_kernels': len(launches), 'kernel_count': len(ks), 'sm_clock_mhz_median': statistics.median(s['sm_clock_mhz'] for s in measured) if measured else None, 'power_w_median': statistics.median(s['power_w'] for s in measured) if measured else None, 'nvml_samples_during_kernels': len(measured), 'throttle_reasons': dict(collections.Counter(s['throttle_reasons'] for s in measured))}


def analyze(database, window=None, layers=(24, 24, 24, 22), samples=None, devices=None):
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as db:
        kernels, copies, apis = load(db)
    results = {}
    for stage, device in enumerate(devices if devices is not None else sorted({r['deviceId'] for r in kernels})):
        ks = sorted((r for r in kernels if r['deviceId'] == device), key=lambda r: r['start'])
        cs = [r for r in copies if r['deviceId'] == device]
        tail_categories(ks, stage)
        moes = [r for r in ks if r['category'] == 'MegaMoE']
        anchors = moes[::layers[stage]]
        cycles = []
        for first, second in itertools.pairwise(anchors):
            lo, hi = first['start'], second['start']
            if window and (lo < window[0] or hi > window[1]):
                continue
            # Without HTTP capture boundaries, discard first/last two rotations (flush tails).
            if not window and (first in anchors[:2] or second in anchors[-2:]):
                continue
            counted = [r for r in ks if lo <= r['start'] < hi]
            target = [r for r in counted if r['category'] == 'QKV/O GEMM']
            if len(target) != 2 * layers[stage]:
                raise ValueError(f'PP{stage}: rotated cycle has {len(target)} dense calls, expected {2 * layers[stage]}; incomplete/mixed forward')
            cycle = account(lo, hi, ks, cs, apis, (samples or {}).get(stage, []))
            cycle['category_kernel_counts'] = dict(collections.Counter(r['category'] for r in counted))
            cycle['writer_grids'] = [list(grid) for grid in sorted({(r.get('gridX'), r.get('gridY')) for r in counted if 'qknorm_rope' in r['name']})]
            cycle['gate_ratios'] = {}
            for label, (low, high) in GATE_US.items():
                selected = [r for r in counted if (target_qkv(r['name']) if label == 'QKV GEMM' else r['category'] == 'QKV/O GEMM' and not target_qkv(r['name']) if label == 'O GEMM' else r['category'] == label)]
                mean = sum((r['end']-r['start'])/1e3 for r in selected)/layers[stage] if selected else None
                cycle['gate_ratios'][label] = {'call_count': len(selected), 'serving_us_per_target_layer': mean, 'r19_gate_us': [low, high], 'ratio_range': [mean/high, mean/low] if mean is not None else None}
            cycles.append(cycle)
        if not cycles:
            raise ValueError(f'PP{stage}: no complete cycles in steady window')
        results[str(stage)] = {'device_id': device, 'layers': layers[stage], 'period_us': distribution([r['period_us'] for r in cycles]), 'gpu_idle_us': distribution([r['gpu_idle_us'] for r in cycles]), 'kernel_union_us': distribution([r['kernel_union_us'] for r in cycles]), 'compute_kernel_union_us': distribution([r['compute_kernel_union_us'] for r in cycles]), 'nccl_only_us': distribution([r['nccl_only_us'] for r in cycles]), 'nccl_or_copy_union_us': distribution([r['category_union_us'].get('PP send/recv or copy', 0) for r in cycles]), 'cycles': cycles}
    return {'schema': 'pp4-prefill-profile/v1', 'kernel_names': {str(r['demangledName']): r['name'] for r in kernels}, 'database': str(database), 'window_ns': window, 'gap_kernel_reference': 'before_kernel/after_kernel are StringIds, resolved by kernel_names.', 'cycle_definition': 'Rotated L-target-MegaMoE intervals; not request IDs; one steady pair forward worth of work. Validate 2L target dense calls. Does not prove pair shape; verify writer grids/server logs.', 'accounting': 'period = kernel_union + copy_only + gpu_idle. Summed category durations overlap; exclusive_busy separates overlap. NCCL occupancy is transfer, never idle. Gap causes are evidence candidates, not proof of causality; queued kernels may wait for stream/remote dependencies. API-to-kernel delay includes queued work, not just host launch delay.', 'clock_power': 'NVML samples matched by explicit ordered worker UUID map and capture UTC epoch, conditioned on kernel union. SM clock and power are 10 Hz samples, not per-kernel energy or a counterfactual estimate of throttling cost. Nsight GPC raw values are not converted or mapped by undocumented typeId bits.', 'r19_comparison': 'Historical isolated gates differ in routing/clock/source. Ratios are diagnostic, not speedup claims. No gate clocks are substituted into serving. Power-cap milliseconds remain unidentifiable without matched same-shape clock evidence.', 'stages': results}


def markdown(result):
    lines = ['# PP4 Prefill profile', '', result['cycle_definition'], '', '| PP | cycles | period median ms | kernel union ms | idle median us |', '|---|---:|---:|---:|---:|']
    for stage, data in result['stages'].items():
        lines.append(f"| {stage} | {len(data['cycles'])} | {data['period_us']['median']/1e3:.3f} | {data['kernel_union_us']['median']/1e3:.3f} | {data['gpu_idle_us']['median']:.3f} |")
    lines += ['', '| category median kernel us/cycle | ' + ' | '.join('PP'+s for s in result['stages']) + ' |', '|---|' + '---:|'*len(result['stages'])]
    for label in CATEGORIES:
        values = [statistics.median(c['summed_kernel_us'][label] for c in stage['cycles']) for stage in result['stages'].values()]
        lines.append('| ' + label + ' | ' + ' | '.join(f'{v:.2f}' for v in values) + ' |')
    lines += ['', '| PP | pre-API idle lower bound median us | kernel-conditioned SM MHz | power W |', '|---|---:|---:|---:|']
    for stage, data in result['stages'].items():
        def median_value(key, cycles=data['cycles']):
            values = [c[key] for c in cycles if c[key] is not None]
            return f'{statistics.median(values):.2f}' if values else 'unknown'
        lines.append(f"| {stage} | {median_value('pre_api_idle_lower_bound_us')} | {median_value('sm_clock_mhz_median')} | {median_value('power_w_median')} |")
    lines += ['', 'Gap neighbours are StringIds resolved by kernel_names in JSON. Per-cycle overlap, API evidence, NVML samples and r19 ratios are also in JSON. Pre-API idle is a lower bound from late-launch candidates; unresolved gaps are not assigned zero cost or claimed as graph savings. Inspect dependencies before a CUDA graph decision at 150–250 us/cycle. Missing clocks/power are null.', '', result['accounting'], '', result['clock_power'], '', result['r19_comparison'], '']
    return '\n'.join(lines)


def mapped_devices(db, gpu_map):
    """Join physical PCI addresses; Nsight's UUID may be anonymized on export."""
    def pci(value):
        return value.lower().lstrip('0')
    addresses = [pci(row[1].strip()) for row in csv.reader(gpu_map.open())]
    gpus = records(db, 'TARGET_INFO_GPU')
    devices = []
    for address in addresses:
        matches = [r['id'] for r in gpus if pci(r['busLocation']) == address]
        if len(matches) != 1:
            raise ValueError(f'nonunique/missing Nsight GPU for PCI address {address}')
        devices.append(matches[0])
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError('expected four unique GPUs in logical PP order')
    return devices


def validate_runtime(runtime, name, database):
    """Cheap in-allocation coverage checks. Full attribution runs after collection."""
    gpu_map = runtime / f'profile/gpu-map-{name}.csv'
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        window = capture_window(db, runtime / 'logs/profile-capture.log', 500_000_000)
        if not 5e9 <= window[1] - window[0] <= 10e9:
            raise ValueError('expected 5–10 seconds inside acknowledged steady capture')
        devices = mapped_devices(db, gpu_map)
        epoch = records(db, 'TARGET_INFO_SESSION_START_TIME')[0]['utcEpochNs']
        samples = telemetry_samples(sorted((runtime / 'logs').glob('gpu-telemetry-*.csv')), gpu_map, epoch, 0)
        kernels, copies, _ = load(db)
    coverage = {}
    for stage, device in enumerate(devices):
        ks = [r for r in kernels if r['deviceId'] == device and window[0] <= r['start'] < window[1]]
        if not ks or max(r['end'] for r in ks) - min(r['start'] for r in ks) < window[1]-window[0]-1e9:
            raise ValueError(f'missing sustained kernels on device {device}')
        if not any(r['deviceId'] == device and r['name'] == 'memcpy' and window[0] <= r['start'] < window[1] for r in copies):
            raise ValueError(f'missing memcpy in steady capture on device {device}')
        # Round 24 (41423181527) saw 4-7 one-request forwards (gridX 125) among about 70 pairs per worker.
        writers = [r for r in ks if 'qknorm_rope' in r['name']]
        pairs = sum(250 <= r['gridX'] <= 256 for r in writers)
        singles = sum(120 <= r['gridX'] <= 128 for r in writers)
        if not writers or any(r['gridY'] != 72 for r in writers) or pairs+singles != len(writers) or pairs < .8*len(writers):
            raise ValueError('writer grid does not identify a steady two-request forward')
        matched = sum(r['api'] is not None for r in ks)
        if matched < len(ks)*.9:
            raise ValueError('missing launch API correlation for more than 10% of steady kernels')
        spans = merge((r['start'], r['end']) for r in ks)
        starts = [a for a, _ in spans]
        measured = []
        for row in samples.get(stage, []):
            i = bisect.bisect_right(starts, row['timestamp'])-1
            if i >= 0 and row['timestamp'] < spans[i][1]:
                measured.append(row)
        if len(measured) < 20:
            raise ValueError('insufficient kernel-conditioned NVML clock/power samples')
        coverage[str(stage)] = {'kernel_count': len(ks), 'api_matched_kernels': matched, 'nvml_samples_during_kernels': len(measured), 'nvml_sample_interval_ms': distribution([(b['timestamp']-a['timestamp'])/1e6 for a, b in itertools.pairwise(measured)])}
    return {'window_ns': window, 'devices_in_pp_order': devices, 'coverage': coverage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture-log', type=Path)
    parser.add_argument('--guard-seconds', type=float, default=.5)
    parser.add_argument('--gpu-map', type=Path)
    parser.add_argument('--telemetry', type=Path, nargs='*', default=[])
    parser.add_argument('--telemetry-timezone-hours', type=int, default=0)
    args = parser.parse_args()
    window, samples, devices = None, None, None
    with sqlite3.connect(f'file:{args.database}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        if args.capture_log:
            window = capture_window(db, args.capture_log, round(args.guard_seconds*1e9))
        if args.gpu_map:
            devices = mapped_devices(db, args.gpu_map)
        if args.telemetry:
            if not args.gpu_map:
                parser.error('--telemetry requires --gpu-map in logical PP order')
            epoch = records(db, 'TARGET_INFO_SESSION_START_TIME')[0]['utcEpochNs']
            samples = telemetry_samples(args.telemetry, args.gpu_map, epoch, args.telemetry_timezone_hours)
    result = analyze(args.database, window, samples=samples, devices=devices)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    args.output.with_suffix('.md').write_text(markdown(result))


if __name__ == '__main__':
    main()
