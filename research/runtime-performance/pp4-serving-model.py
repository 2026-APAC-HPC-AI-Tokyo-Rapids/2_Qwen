#!/usr/bin/env python3
"""Closed-loop model of the Firmus P3(PP4)/D1 serving path under the official AIPerf window.

Usage: python3 THIS.py OUTPUT.json   (CPU only, about 2 minutes on 8 cores)

Calibrated on run 40883389891 (9392.85) and checked out of sample on 40825417205
(8952.69, depth 2 and in-stream PyNccl: loop ratios V0 below). The same client,
router hold and Decode lineage (scratchpad/sys/des.py) reproduces EP4 run
40630160781 (9235.28) at 9230. Research note 2026-10-30-pp4-structure-model.md.

Client: C slots, all issued in the first 0.08 s, re-issued 4 ms after each completion until 120 s;
score = 200 * good / (last completion). Router: least outstanding Prefill pipeline, 8 ms serial
tokenization per pipeline. Prefill pipeline w: microbatches of up to `mb` requests; the stage-0 GPU
starts one every P[w] (gaussian jitter); a microbatch is formed lead_r*P before its start; the last
stage finishes it pipe_r*P after the start; Decode can admit it `kvready` later; the Prefill notice
that releases the router's held first token comes release_r*P (+release_extra) after the last stage.
lead_r, pipe_r, release_r and kvready come from the PP4 loop timing model of the previous worker
(codex-scratch/pl/des/loopsim.py; depth 1, stage-0 admission, last stage result first, copy engine).
Decode: 4 DP ranks in lockstep, step = s_ref + b * (n - n_ref) with n the mean running per rank
(fit on the 40883389891 Decode log: 37.4 ms at 41.9, local slope 0.72 ms per request), cap per rank,
per-request acceptance alpha ~ N(0.61, 0.02), chain of K = 3 drafts.
"""
import heapq
import random
import statistics as st


def run(C=225, P=(0.1208, 0.1202, 0.1239), jit=0.02, mb=2, lead_r=1.015, pipe_r=3.907, release_r=4.041, release_extra=0.0,
        kvready=0.230, adm=0.005, adm_burst=0.03, start_cut=0.0, launch=0.027, s_ref=37.4e-3, n_ref=41.9, b=0.72e-3,
        alpha=0.61, alpha_sd=0.02, K=3, tok_ms=8e-3, cap_per_rank=44, window=120.0, seed=1, burst_spread=0.08,
        worker_start=0.06, idle_tick=0.005, step_scale=1.0, n_floor=30.0):
    rng = random.Random(seed)
    W, DR = len(P), 4
    reqs = []
    ev = []
    cnt = [0]
    w_out = [0] * W
    tok_free = [0.0] * W

    def push(t, kind, data):
        cnt[0] += 1
        heapq.heappush(ev, (t, kind, cnt[0], data))

    def issue(t):
        i = len(reqs)
        al = min(0.95, max(0.3, rng.gauss(alpha, alpha_sd)))
        burst = i < C
        reqs.append(dict(issue=t, alpha=al, first=None, end=None, burst=burst))
        w = min(range(W), key=lambda x: (w_out[x], rng.random()))
        w_out[w] += 1
        ta = max(t, tok_free[w]) + tok_ms
        tok_free[w] = ta
        push(ta + (adm_burst if burst else adm), 1, (i, w))

    for j in range(C):
        issue(burst_spread * j / C)
    queue = [[] for _ in range(W)]
    last_gs = [None] * W
    form_pending = [False] * W
    for w in range(W):
        push(worker_start, 2, w)
        form_pending[w] = True
    dec_wait = []
    dec_run = [[] for _ in range(DR)]
    push(0.0, 3, None)
    last = [0.0]
    steps = []

    def finish(rq):
        rq['pending_end'] = False
        rq['end'] = max(rq['dend'], rq['first'] + 1e-3)
        last[0] = max(last[0], rq['end'])
        if rq['end'] < window:
            issue(rq['end'] + 0.004)

    while ev:
        t, kind, _, data = heapq.heappop(ev)
        if t > window + 40:
            break
        if kind == 1:
            i, w = data
            queue[w].append(i)
            if not form_pending[w]:
                form_pending[w] = True
                push(t + idle_tick, 2, w)
        elif kind == 2:
            w = data
            form_pending[w] = False
            if not queue[w]:
                continue
            Pw = P[w]
            batch = queue[w][:mb]
            del queue[w][:mb]
            Pk = Pw * (1.0 + rng.gauss(0.0, jit)) if jit else Pw
            gs = t + launch if last_gs[w] is None else max(t + launch, last_gs[w] + Pk)
            last_gs[w] = gs
            done = gs + pipe_r * Pw
            rel = release_r * Pw + release_extra
            for i in batch:
                rq = reqs[i]
                rq['te'] = done
                w_out[w] -= 1
                dec_wait.append((done + (max(0.03, kvready - start_cut) if rq['burst'] else kvready), i))
                rq['notice'] = done + rel - (start_cut if rq['burst'] else 0.0)
                push(rq['notice'], 4, i)
            form_pending[w] = True
            push(max(t + 0.001, gs + Pw - lead_r * Pw), 2, w)
        elif kind == 4:
            rq = reqs[data]
            if 'dout' in rq and rq['first'] is None:
                rq['first'] = max(rq['dout'], rq['notice'] + 0.01)
                if rq.get('pending_end'):
                    finish(rq)
        elif kind == 3:
            dec_wait.sort()
            keep = []
            for (rt, i) in dec_wait:
                if rt <= t:
                    rr = min(range(DR), key=lambda x: len(dec_run[x]))
                    if len(dec_run[rr]) < cap_per_rank:
                        reqs[i]['tok'] = 1
                        dec_run[rr].append(i)
                        continue
                keep.append((rt, i))
            dec_wait = keep
            n = sum(len(x) for x in dec_run)
            if n == 0:
                push(t + 0.002, 3, None)
                continue
            nm = n / DR
            # Below n_floor the step follows the low-load line through (0, 0.45 s_ref) (graph sizes shrink).
            s = s_ref + b * (nm - n_ref) if nm >= n_floor else (0.45 * s_ref + nm / n_floor * (s_ref + b * (n_floor - n_ref) - 0.45 * s_ref))
            s *= step_scale
            if 20 <= t <= 110:
                steps.append((nm, s))
            tend = t + s
            for rr in range(DR):
                nxt = []
                for i in dec_run[rr]:
                    rq = reqs[i]
                    if rq.get('dout') is None:
                        rq['dout'] = tend
                    al = rq['alpha']
                    k2 = 1
                    while k2 <= K and rng.random() < al:
                        k2 += 1
                    rq['tok'] += k2
                    if rq['first'] is None and rq['notice'] <= tend:
                        rq['first'] = max(rq['dout'], rq['notice'] + 0.01)
                    if rq['tok'] >= 200:
                        rq['dend'] = tend
                        if rq['first'] is not None:
                            finish(rq)
                        else:
                            rq['pending_end'] = True
                    else:
                        nxt.append(i)
                dec_run[rr] = nxt
            push(tend, 3, None)
    done = [r for r in reqs if r['end'] is not None]
    good = ttf = itf = 0
    itls = []
    for r in done:
        ttft = r['first'] - r['issue']
        itl = (r['end'] - r['first']) / 199
        itls.append(itl)
        ok = ttft <= 3.0 and itl <= 0.020
        good += ok
        ttf += ttft > 3.0
        itf += (ttft <= 3.0 and itl > 0.020)
    ss = [r for r in done if 20 <= r['issue'] <= 100]
    burst = [r for r in reqs[:C]]
    return dict(C=C, N=len(done), good=good, dur=last[0], score=good * 200 / last[0],
                ttft_fail=ttf, itl_fail=itf, burst_pass=sum(1 for r in burst if r['first'] is not None and r['first'] - r['issue'] <= 3.0),
                itl_mean=st.mean(itls) * 1e3, itl_p99=sorted(itls)[int(.99 * len(itls))] * 1e3,
                ss_ttft=st.median(r['first'] - r['issue'] for r in ss), ss_rate=sum(1 for r in done if 20 <= r['end'] < 110) / 90.0,
                dec_n=st.mean(x[0] for x in steps), dec_step=st.mean(x[1] for x in steps) * 1e3, tail=last[0] - window)


def avg(seeds=6, **kw):
    rs = [run(seed=s, **kw) for s in range(seeds)]
    out = {k: st.mean(r[k] for r in rs) for k in rs[0]}
    out['score_sd'] = st.pstdev(r['score'] for r in rs)
    return out


MEASURED_40883389891 = dict(score=9392.85, burst_pass=98, ttft_fail=127, itl_fail=2, itl_mean=15.30, itl_p99=18.00, ss_ttft=1.481,
                            ss_rate=49.28, dec_n=41.9, dec_step=37.43, tail=3.73)


P_MEASURED = (0.1208, 0.1202, 0.1239)    # 40883389891 Prefill log, per pipeline (8064 tokens / input throughput)
P_PERMUTED = (0.1208, 0.1196, 0.1225)    # last stage on the slowest GPU of prefill-1 (+0.5%) and prefill-2 (+1.1%)
LOOP_V0 = dict(lead_r=1.216, release_r=4.838, kvready=0.47, adm=0.376, adm_burst=0.10)  # depth 2, PyNccl in stream
KEYS = ("score", "score_sd", "burst_pass", "ttft_fail", "itl_fail", "itl_mean", "itl_p99", "ss_ttft", "ss_rate", "dec_n", "dec_step")


def scenarios():
    yield "calibration 40883389891", dict(C=225)
    yield "check 40825417205", dict(C=235, P=(0.1234, 0.1233, 0.1250), s_ref=37.4e-3 * 1.025, **LOOP_V0)
    for C in (205, 215, 225, 235, 240):
        yield "concurrency", dict(C=C)
        yield "gpu order", dict(C=C, P=P_PERMUTED)
    for rx in (-0.24, -0.12):
        for C in (215, 225):
            yield f"gpu order, release {rx:+.2f} s", dict(C=C, P=P_PERMUTED, release_extra=rx)
    for gain in (0.02, 0.03, 0.045):
        for C in (225, 235, 240):
            yield f"3-request microbatch, {gain:.1%} less work per token", dict(C=C, mb=3, P=tuple(p * 1.5 * (1 - gain) for p in P_MEASURED))
    # Other workers' targets as kernel-time cuts: MegaMoE is 57% of a PP4 layer, the dense/attention path 43%.
    for moe, dense, dstep in ((0.02, 0, 1), (0.02, 0, 0.95), (0.02, 0.03, 0.95), (0.05, 0, 0.95), (0.05, 0.03, 1), (0.05, 0.03, 0.95)):
        f = 1 - 0.57 * moe - 0.43 * dense
        for C in (215, 225, 235):
            yield f"gpu order, MegaMoE -{moe:.0%}, dense -{dense:.0%}, Decode x{dstep}", dict(C=C, P=tuple(p * f for p in P_PERMUTED), step_scale=dstep)
    for f in (0.935, 0.90):
        for cap in (44, 48):
            for b in (0.72, 0.30):
                for C in (225, 235, 240):
                    yield f"gpu order, Prefill x{f}, Decode x0.95, cap {cap}, slope {b}", dict(C=C, P=tuple(p * f for p in P_PERMUTED), step_scale=0.95, cap_per_rank=cap, b=b * 1e-3)


def evaluate(item):
    name, kw = item
    o = avg(seeds=6, **kw)
    return dict(name=name, C=kw["C"], **{k: round(o[k], 2) for k in KEYS})


if __name__ == "__main__":
    import json
    import sys
    from multiprocessing import Pool
    from pathlib import Path
    with Pool() as pool:
        rows = pool.map(evaluate, list(scenarios()))
    for r in rows:
        print(f"{r['name']:58s} C{r['C']} {r['score']:8.1f} burst {r['burst_pass']:5.1f} TTFT fail {r['ttft_fail']:5.1f} ITL fail {r['itl_fail']:5.1f} ITL {r['itl_mean']:5.2f}/{r['itl_p99']:5.2f} ms")
    Path(sys.argv[1]).write_text(json.dumps({"schema": "pp4-serving-model/v1", "measured_40883389891": MEASURED_40883389891, "rows": rows}, indent=1) + "\n")
