"""Read CPU-exported Nsight SQLite and archived Torch spans; no GPU execution.

Usage: python3 analyze-38664373784.py SQLITE_DIR ARCHIVE_ROOT
Durations are recorded evidence, not hardware utilization or removable costs.
"""
import collections
import bisect
import gzip
import json
import sqlite3
import statistics
import sys
from pathlib import Path


def union(intervals):
    end = None
    total = 0
    for a, b in sorted(intervals):
        total += b - max(a, end if end is not None else a) if end is None or b > end else 0
        end = max(b, end if end is not None else b)
    return total


def analyze(directory, archive):
    result = {}
    for worker in ("prefill-0", "prefill-1", "prefill-2", "decode-0"):
        c = sqlite3.connect(f"file:{directory / (worker + '.sqlite')}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        origin = c.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
        start, end = 1792040417278964454 - origin, 1792040422297356859 - origin
        rows = [dict(r) for r in c.execute("SELECT k.*,s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName")]
        sync = [dict(r) for r in c.execute("SELECT r.*,s.value AS name FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE '%Synchronize%'")]
        diagnostics = [dict(r) for r in c.execute("SELECT DISTINCT severity,text,globalPid,timestamp,timestampType FROM DIAGNOSTIC_EVENT WHERE severity>1")]
        ranks = []
        for device in range(4):
            allrows = [r for r in rows if r['deviceId'] == device]
            selected = [r for r in allrows if r['start'] < end and r['end'] > start]
            mega = [r for r in selected if 'sm90_fp8_mega_moe_impl' in r['name']]
            durations = [(min(r['end'], end) - max(r['start'], start)) for r in selected]
            mega_full = [(r['end'] - r['start']) / 1000 for r in mega if r['start'] >= start and r['end'] <= end]
            pid = allrows[0]['globalPid']
            ranks.append({'device': device, 'pid': (pid >> 24) & 0xffffff,
                          'all_kernel_count': len(allrows), 'active_kernel_count': len(selected),
                          'recorded_union_pct': 100 * union([(max(start,r['start']),min(end,r['end'])) for r in selected]) / (end-start),
                          'mega_duration_share_pct': 100 * sum(min(end,r['end'])-max(start,r['start']) for r in mega) / sum(durations),
                          'mega_complete_count': len(mega_full), 'mega_mean_us': statistics.mean(mega_full),
                          'mega_p50_us': statistics.median(mega_full), 'mega_p95_us': sorted(mega_full)[int((len(mega_full)-1)*.95)],
                          'cuda_loss_warning': any(d['globalPid']==pid and 'Not all CUDA events' in d['text'] for d in diagnostics)})
        geometries = {tuple(r[k] for k in ('gridX','gridY','gridZ','blockX','blockY','blockZ','registersPerThread','staticSharedMemory','dynamicSharedMemory','name')) for r in rows if 'sm90_fp8_mega_moe_impl' in r['name']}
        gpu_intervals = {}
        for rank in ranks:
            intervals = sorted((r['start'],r['end']) for r in rows if r['deviceId']==rank['device'])
            gpu_intervals[rank['pid']] = (intervals,[i[0] for i in intervals],max(b-a for a,b in intervals))
        stages = []
        for path in sorted((archive / 'profile/profile/torch' / worker).glob('*.gz')):
            trace = json.load(gzip.open(path, 'rt'))
            annotations = [e for e in trace['traceEvents'] if e.get('cat')=='user_annotation' and e.get('name') in ('draft_extend','scheduler.get_next_batch_to_run','gloo:all_gather') and 'dur' in e]
            groups = collections.defaultdict(list)
            for e in annotations:
                a = trace['baseTimeNanoseconds'] + round(e['ts']*1000) - origin
                b = a + round(e['dur']*1000)
                if a < start or b > end:
                    continue
                waits = [(max(a,r['start']),min(b,r['end'])) for r in sync if ((r['globalTid']>>24)&0xffffff)==e['pid'] and r['start']<b and r['end']>a]
                intervals, starts, longest = gpu_intervals[e['pid']]
                gpu = [(max(a,s),min(b,t)) for s,t in intervals[bisect.bisect_left(starts,a-longest):bisect.bisect_left(starts,b)] if t>a]
                groups[e['name']].append(((b-a)/1e6,union(waits)/1e6,union(gpu)/1e6))
            for name, values in groups.items():
                stages.append({'trace':path.name,'stage':name,'complete_spans_in_active_window':len(values),
                               'mean_span_ms':statistics.mean(v[0] for v in values),
                               'mean_overlapping_cuda_sync_ms':statistics.mean(v[1] for v in values),
                               'mean_overlapping_recorded_gpu_kernel_ms':statistics.mean(v[2] for v in values)})
        result[worker] = {'active_relative_ns':[start,end], 'ranks':ranks,
                          'mega_geometries': [dict(zip(('gridX','gridY','gridZ','blockX','blockY','blockZ','registersPerThread','staticSharedMemory','dynamicSharedMemory','name'), g)) for g in sorted(geometries)],
                          'cpu_stages':stages,'diagnostics':diagnostics}
        c.close()
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()),indent=2))
