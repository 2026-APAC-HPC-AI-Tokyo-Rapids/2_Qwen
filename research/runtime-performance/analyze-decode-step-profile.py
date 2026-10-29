"""Per-step Decode kernel time by category from an nsys SQLite export.

Usage: analyze-decode-step-profile.py <nsys-decode-0.sqlite> <output.json>

A step starts at a TARGET_VERIFY CUDA-graph launch holding 94 MegaMoE calls and
ends at the next one. Kernels up to the last MegaMoE are the 94 layers; the rest
of that graph is the verify tail (LM head); then eager kernels (sampling, draft
extend) and the draft CUDA graph. n = verify rows / 4 requests per rank. For each
category: mean us per step at each n and a least-squares slope over n.
"""
import sqlite3, sys, collections, statistics, json
db = sys.argv[1]
c = sqlite3.connect(db)
def cat(name, sec):
    if sec != 'verify-layers': return sec
    for key, label in (('mega_moe_impl', 'megamoe'), ('pre_dispatch', 'pre_dispatch'), ('add_rmsnorm_group_quant', 'norm_quant'), ('FusedAddRMSNorm', 'post_attn_norm'), ('qknorm_rope', 'qknorm_rope'), ('combine_fixed', 'fa3_combine'), ('router_triton', 'router_topk'), ('splitKreduce', 'gate'), ('nvjet', 'gate'), ('gemm_1d2d', 'dense'), ('device_kernel', 'fa3_verify')):
        if key in name: return label
    return 'other'
per = collections.defaultdict(lambda: collections.defaultdict(list))
for dev in range(4):
    rows = list(c.execute("select k.start,k.end,s.value,k.gridX,k.graphId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where deviceId=? order by start", (dev,)))
    moe_graphs = collections.Counter(r[4] for r in rows if 'mega_moe_impl' in r[2] and r[4] is not None)
    starts = []; i = 0
    while i < len(rows):
        g = rows[i][4]
        if g is not None and moe_graphs.get(g):
            j = i
            while j < len(rows) and rows[j][4] == g: j += 1
            if sum('mega_moe_impl' in rows[k][2] for k in range(i, j)) == 94: starts.append((i, j))
            i = j
        else: i += 1
    for (a, b), (a2, _) in zip(starts, starts[1:]):
        n = next(r[3] for r in rows[a:b] if 'add_rmsnorm_group_quant' in r[2]) // 4
        last_moe = max(k for k in range(a, b) if 'mega_moe_impl' in rows[k][2])
        tot = collections.defaultdict(float); prev = rows[a][0]
        for k in range(a, a2):
            r = rows[k]
            sec = ('verify-layers' if k <= last_moe else 'verify-tail') if k < b else ('eager' if r[4] is None else 'draft-graph')
            key = cat(r[2], sec)
            tot[key] += (r[1] - r[0]) / 1e3
            tot['idle:' + (sec if sec != 'verify-layers' else 'layers')] += max(0, r[0] - prev) / 1e3
            prev = max(prev, r[1])
        tot['period'] = (rows[a2][0] - rows[a][0]) / 1e3
        for key, v in tot.items(): per[n][key].append(v)
keys = sorted({k for d in per.values() for k in d})
ns = sorted(per)
out = {}
print('n:', ns, 'steps:', [len(per[n]['period']) for n in ns])
for key in keys:
    xs, ys = [], []
    for n in ns:
        v = per[n].get(key, [])
        v = v + [0.0] * (len(per[n]['period']) - len(v))
        xs += [n] * len(v); ys += v
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    at = {n: statistics.mean(per[n].get(key, [0.0]) + [0.0] * (len(per[n]['period']) - len(per[n].get(key, [])))) for n in ns}
    out[key] = {'slope_us_per_request': slope, 'intercept_us': my - slope * mx, 'mean_us_by_n': at}
    print(f"{key:18s} slope {slope:8.1f} us/req  intercept {my - slope*mx:9.1f}  n40 {at.get(40,0):8.1f}  n42 {at.get(42,0):8.1f}  n44 {at.get(44,0):8.1f}")
json.dump(out, open(sys.argv[2], 'w'), indent=1)
