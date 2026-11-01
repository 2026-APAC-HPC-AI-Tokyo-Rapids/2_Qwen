"""Describe current split MegaMoE diagnostics; never select a serving variant.

Globaltimer ns only. CTA medians are distributions, not additive kernel phases.
Stage waits and sampled half/scale/drain intervals are inside the mainloop.
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

SCHEMA = 'prefill-megamoe-diag/v1'


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    return {'median':statistics.median(values), 'p95':values[math.ceil(.95*len(values))-1], 'max':max(values), 'n':len(values)}


def summarize_case(case, build):
    samples = case['samples_us']
    assert len(samples) == 6 and case['byte_equal'] is True
    assert all(row['order'] == (['control','diagnostic'] if i%2 == 0 else ['diagnostic','control']) for i,row in enumerate(samples))
    assert all(math.isfinite(row[k]) and row[k]>0 for row in samples for k in ('control','diagnostic'))
    control = statistics.median(row['control'] for row in samples)
    diag = statistics.median(row['diagnostic'] for row in samples)
    impact = diag/control-1
    spilled = any(int(a) or int(b) for a,b in build['diagnostic']['spill_store_load_bytes'])
    usable = abs(impact) <= .02 and not spilled and not build['diagnostic']['c75xx']
    trace = case['trace']
    assert len(trace) == 132*12
    assert {(r['cta'],r['warp']) for r in trace} == {(c,w) for c in range(132) for w in range(12)}
    metrics, detail = defaultdict(list), defaultdict(list)
    tiles, entries, exits, ends = [], [], [], []
    per_cta = []
    for row in trace:
        events = row['events']
        assert events[0][1] == 1 and events[-1][1] == 15
        assert all(len(e) == 8 for e in events)
        assert all(b[0] >= a[0] for a,b in pairwise(events))
        entries.append(events[0][0]); exits.append(events[-1][0])
        if row['warp'] < 4:
            for phase in (1,2):
                metrics[f'producer_warp{row["warp"]}_fc{phase}_empty_wait_us'].append(row['wait_ns'][f'fc{phase}_empty']/1000)
            continue
        by_id = {e[1]:e[0] for e in events if e[1] in (16,17,18)}
        assert {16,17,18} <= by_id.keys()
        metrics['precombine_grid_wait_us'].append((by_id[17]-by_id[16])/1000)
        metrics['dispatch_handshake_us'].append((by_id[18]-by_id[17])/1000)
        metrics['combine_to_warp_exit_us'].append((events[-1][0]-by_id[18])/1000)
        if row['warp'] not in (4,8):
            continue
        groups = defaultdict(dict)
        for e in events:
            if e[1] in (2,3,4):
                assert e[1] not in groups[e[3]], 'duplicate tile event'
                groups[e[3]][e[1]] = e
        totals = defaultdict(float)
        for group in groups.values():
            assert set(group) == {2,3,4}, 'incomplete tile'
            a,b,c = [group[k] for k in (2,3,4)]
            assert a[2:7] == b[2:7] == c[2:7]
            phase,rows = a[2], a[4]
            assert phase in (1,2) and 0 < rows <= 128
            main,epi = (b[0]-a[0])/1000,(c[0]-b[0])/1000
            totals[f'fc{phase}_mainloop_us'] += main
            totals[f'fc{phase}_epilogue_us'] += epi
            tiles.append({'cta':row['cta'], 'wg':(row['warp']-4)//4, 'phase':phase, 'rows':rows, 'active_wg_rows':min(64,max(0,rows-((row['warp']-4)//4)*64)), 'expert':a[5], 'pool_block':a[6], 'n_block':a[7], 'start_ns':a[0], 'end_ns':c[0], 'mainloop_us':main, 'epilogue_us':epi})
        for phase in (1,2):
            total = totals.get(f'fc{phase}_mainloop_us')
            if total:
                full = row['wait_ns'][f'fc{phase}_full']/1000
                totals[f'fc{phase}_full_wait_us'] = full
                totals[f'fc{phase}_full_wait_fraction_within_mainloop'] = full/total
        per_cta.append({'cta':row['cta'], 'wg':(row['warp']-4)//4, 'totals':dict(totals)})
        for key,value in totals.items():
            metrics[key].append(value)
        # Match sample endpoints within one CTA/WG/tile. Never join separate calls.
        pending = {}
        sampled_full = {}
        for e in events:
            time,event,phase,tile,rows,_expert,_pool,arg = e
            key = (phase,tile,arg)
            pairs = {8:(7,'accumulator_wait'), 9:(8,'rescale'), 10:(9,'issue'), 12:(11,'scale_prepare_including_full_wait'), 6:(13,'chunk_drain'), 20:(19,'sampled_full_wait')}
            if event in pairs:
                begin,label = pairs[event]
                start = pending.get((begin,*key))
                if start is not None:
                    # Half args encode k*2+h. Preserve half/chunk and tail stratum.
                    bucket = f'fc{phase}/{"tail" if rows<128 else "full"}/{label}'
                    if event in (8,9,10):
                        bucket += f'/half{arg%2}'
                    detail[bucket].append((time-start)/1000)
                    if event == 20:
                        sampled_full[key] = time-start
                    if event == 12 and key in sampled_full:
                        exclusive = time-start-sampled_full[key]
                        assert exclusive >= 0, 'sampled full wait exceeds scale interval'
                        detail[f'fc{phase}/{"tail" if rows<128 else "full"}/scale_prepare_excluding_full_wait'].append(exclusive/1000)
                    if event == 6:
                        chunk_start = pending.get((5,*key))
                        if chunk_start is not None:
                            detail[f'fc{phase}/{"tail" if rows<128 else "full"}/chunk{arg}_span'].append((time-chunk_start)/1000)

            pending[(event,*key)] = time
        ends.append((max(e[0] for e in events if e[1]==4),row['cta'],row['warp']))
    last = max(t['end_ns'] for t in tiles)
    last_tiles = [t for t in tiles if t['end_ns'] == last]
    last_ctas = {t['cta'] for t in last_tiles}
    tail = [t for t in tiles if t['phase']==2 and t['rows']<128]
    full = [t for t in tiles if t['phase']==2 and t['rows']==128]
    last_tail = max((t['end_ns'] for t in tail), default=None)
    last_full = max((t['end_ns'] for t in full), default=None)
    tail_ctas = {t['cta'] for t in tail}
    tail_analysis = {'last_math_tiles':last_tiles, 'last_math_ctas_have_any_fc2_tail':bool(last_ctas & tail_ctas), 'fc2_tail_tiles':len(tail), 'fc2_true_m64_tiles':sum(t['rows']<=64 for t in tail), 'last_fc2_tail_minus_last_full_us':None if last_tail is None or last_full is None else (last_tail-last_full)/1000, 'last_fc2_tail_to_last_math_end_us':None if last_tail is None else (last-last_tail)/1000, 'last_fc2_tail_to_last_warp_exit_us':None if last_tail is None else (max(exits)-last_tail)/1000, 'fc2_tail_inactive_wg_visits':sum(t['active_wg_rows']==0 for t in tail), 'last_tail_ctas_also_last_exit_ctas':bool({t['cta'] for t in tail if t['end_ns']==last_tail} & {r['cta'] for r in trace if r['events'][-1][0]==max(exits)}), 'tail_mainloop_us':distribution([t['mainloop_us'] for t in tail]), 'full_mainloop_us':distribution([t['mainloop_us'] for t in full]), 'interpretation':'Completion association only; no counterfactual M64 speedup. Row/WG skip, expert, CTA work and clock can confound duration.'}
    return {'timing':{'control_us':control, 'diagnostic_us':diag, 'impact_fraction':impact, 'paired_impact_fraction':distribution([r['diagnostic']/r['control']-1 for r in samples]), 'forward_impact_fraction':statistics.median(r['diagnostic']/r['control']-1 for r in samples if r['order'][0]=='control'), 'reverse_impact_fraction':statistics.median(r['diagnostic']/r['control']-1 for r in samples if r['order'][0]=='diagnostic')}, 'fine_intervals_usable_for_adoption':usable, 'qualification':'Diagnostic association only. Reject fine attribution if absolute event-time overhead exceeds 2 percent, CPU spill or C75xx; small total impact alone does not prove every interval undisturbed.', 'instrumented_warp_span_us':(max(exits)-min(entries))/1000, 'last_warp_exit':[{k:r[k] for k in ('cta','warp','sm')} for r in trace if r['events'][-1][0]==max(exits)], 'cta_wg_distributions':{k:distribution(v) for k,v in metrics.items()}, 'sampled_intervals_us':{k:distribution(v) for k,v in detail.items()}, 'cta_wg_totals':per_cta, 'tail':tail_analysis, 'expert_rows':case['expert_rows']}


def summarize(directory, run_id):
    rows, failures = [], []
    uuids = set()
    for worker in range(3):
        for rank in range(4):
            path = directory/f'prefill-{worker}'/f'rank-{rank}.json'
            try:
                data = json.loads(path.read_text())
                assert data['schema'] == SCHEMA and data['passed'] is True
                assert data['run_id'] == run_id and data['rank']==rank and data['tokens']==7988
                assert data['routing_is_current_serving'] is False
                assert data['uuid'] not in uuids, 'duplicate physical GPU'
                uuids.add(data['uuid'])
                assert data['build']['serving_sass_equal'] is True
                assert set(data['cases']) == {'balanced','historical-routing','edges'}
                cases = {k:summarize_case(v,data['build']) for k,v in data['cases'].items()}
                rows.append({'worker':worker, 'rank':rank, 'uuid':data['uuid'], 'cases':cases})
            except (OSError,ValueError,KeyError,IndexError,AssertionError,TypeError) as exc:
                failures.append({'worker':worker,'rank':rank,'error':str(exc) or type(exc).__name__})
    return {'schema':'prefill-megamoe-diag-summary/v1', 'run_id':run_id, 'complete':not failures and len(rows)==12, 'adoption':None, 'gpus':rows, 'failures':failures, 'interpretation':'Globaltimer ns, same-call timestamps only. Per-CTA/WG totals and medians are not a decomposition to add into kernel time. Full/empty waits are nested mainloop diagnostics. Fixed historical routing is not current serving; true M64 upper bound remains unknown.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(summarize(a.directory,a.run_id),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
