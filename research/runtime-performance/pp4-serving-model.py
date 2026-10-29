#!/usr/bin/env python3
"""Closed-loop model of the Firmus P3(PP4)/D1 serving path under the official AIPerf window.

Usage: python3 THIS.py OUTPUT.json   (CPU only, about 25 minutes on 8 cores)

Round 3 (research note 2026-10-30-pp4-system-r3.md): calibrated on run 40925558339 (9590.42, cap 48)
and checked on 40883389891 (9392.85, cap 44). Round 2 (2026-10-30-pp4-structure-model.md) calibrated on
40883389891 alone; its results are pp4-serving-model-20261030.json.

Client: C slots, all issued in the first 0.08 s, re-issued 4 ms after each completion until 120 s;
score = 200 * good / (last completion). Router: least outstanding Prefill pipeline, 8 ms serial
tokenization per pipeline, then `adm` until stage 0 can admit the request (bootstrap). Prefill pipeline w:
microbatches of up to `mb` requests; stage 0 starts one every P[w] (gaussian jitter; a short microbatch
takes its share of the period plus 6% of the rest); a microbatch is formed lead_r*P before its start;
the last stage finishes it pipe_r*P after the start; Decode can admit it `kvready` later; the Prefill
notice that releases the router's held first token comes release_r*P + release_extra after the last
stage. lead_r, pipe_r, release_r and kvready come from the PP4 loop timing model of the round-2 worker
(codex-scratch/pl/des/loopsim.py; depth 1, stage-0 admission, last stage result first, copy engine);
burst requests see start_cut less of the notice and Decode admission delays (empty system).
start_extra[w] is the one-off delay of the third microbatch after the profiling start (Prefill logs).
Decode: 4 DP ranks in lockstep, step = s_ref + b * (n - n_ref) with n the mean running per rank
(40925558339 Decode log: 38.0 ms at 43.2, slope 0.55 ms per request from the medians binned by n),
cap per rank, per-request acceptance alpha ~ N(0.61, 0.02), chain of K = 3 drafts.
Decode output stalls: every stall_period the Decode server delivers nothing for stall_len (0.33 s every
14-19 s in every archived run), while its scheduler keeps stepping.
"""
import heapq
import random
import statistics as st


def run(C=225, P=(0.1208, 0.1202, 0.1239), jit=0.02, mb=2, lead_r=1.015, pipe_r=3.907, release_r=4.041, release_extra=0.0,
        kvready=0.230, adm=0.005, adm_burst=0.03, start_cut=0.0, launch=0.027, s_ref=37.4e-3, n_ref=41.9, b=0.72e-3,
        alpha=0.61, alpha_sd=0.02, K=3, tok_ms=8e-3, cap_per_rank=44, window=120.0, seed=1, burst_spread=0.08,
        worker_start=0.06, idle_tick=0.005, step_scale=1.0, n_floor=30.0, stall_period=0.0, stall_len=0.33,
        start_extra=(0.0, 0.0, 0.0)):
    rng = random.Random(seed)
    W, DR = len(P), 4
    # Decode output stalls: every stall_period the Decode server delivers nothing for stall_len while its scheduler
    # keeps stepping; first tokens and completions that fall inside are delivered at the end (random phase per seed).
    phase = rng.uniform(0.0, stall_period) if stall_period else 0.0

    def deliver(t):
        if not stall_period or t < phase:
            return t
        k = int((t - phase) / stall_period)
        a = phase + k * stall_period
        return a + stall_len if t < a + stall_len else t
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
    free = [None] * W    # when stage 0 of pipeline w can start its next microbatch
    last_done = [0.0] * W
    formed = [0] * W
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
            # A microbatch short of `mb` requests takes its share of the period plus a fixed part
            # (one 4032-token request: 62 ms of a 117 ms period in 40925558339).
            f = len(batch) / mb + 0.06 * (1.0 - len(batch) / mb)
            Pk = f * Pw * ((1.0 + rng.gauss(0.0, jit)) if jit else 1.0)
            formed[w] += 1
            gs = t + launch if free[w] is None else max(t + launch, free[w])
            if formed[w] == 3:
                gs += start_extra[w]    # the one-off hiccup at the third microbatch after the profiling start
            free[w] = gs + Pk
            done = max(gs + pipe_r * Pw * f, last_done[w] + Pk)
            last_done[w] = done
            rel = release_r * Pw + release_extra
            for i in batch:
                rq = reqs[i]
                rq['te'] = done
                w_out[w] -= 1
                dec_wait.append((done + (max(0.03, kvready - start_cut) if rq['burst'] else kvready), i))
                rq['notice'] = done + rel - (start_cut if rq['burst'] else 0.0)
                push(rq['notice'], 4, i)
            form_pending[w] = True
            push(max(t + 0.001, free[w] - lead_r * Pw), 2, w)
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
                        rq['dout'] = deliver(tend)
                    al = rq['alpha']
                    k2 = 1
                    while k2 <= K and rng.random() < al:
                        k2 += 1
                    rq['tok'] += k2
                    if rq['first'] is None and rq['notice'] <= tend:
                        rq['first'] = max(rq['dout'], rq['notice'] + 0.01)
                    if rq['tok'] >= 200:
                        rq['dend'] = deliver(tend)
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


MEASURED = {
    "40925558339": dict(score=9590.42, burst_pass=104, ttft_fail=122, itl_fail=5, itl_mean=15.60, itl_p99=18.47, ss_ttft=1.326,
                        ss_rate=50.22, dec_n=43.17, dec_step=38.00, tail=3.58),
    "40883389891": dict(score=9392.85, burst_pass=98, ttft_fail=127, itl_fail=2, itl_mean=15.30, itl_p99=18.00, ss_ttft=1.481,
                        ss_rate=49.28, dec_n=41.68, dec_step=37.49, tail=3.73),
}
KEYS = ("score", "score_sd", "burst_pass", "ttft_fail", "itl_fail", "itl_mean", "itl_p99", "ss_ttft", "ss_rate", "dec_n", "dec_step", "tail")
P_R2 = (0.1172, 0.1171, 0.1203)    # 40925558339 Prefill log, median 8064-token period per pipeline
STALL = dict(stall_period=15.2, stall_len=0.33)
# 40925558339 as served: cap 48, round-2 kernels, Decode output stalls, start hiccup of 0.10/0.10/0.22 s.
CAL = dict(C=225, P=P_R2, cap_per_rank=48, s_ref=38.0e-3, n_ref=43.2, b=0.55e-3, adm=0.06, release_extra=0.06, start_cut=0.20,
           start_extra=(0.10, 0.10, 0.22), **STALL)
NOSTALL = dict(CAL, stall_period=0.0)


def scenarios():
    yield "calibration 40925558339", dict(CAL), {}
    yield "check 40883389891", dict(CAL, P=(0.1208, 0.1202, 0.1239), cap_per_rank=44, s_ref=37.49e-3, n_ref=41.68, start_extra=(0.09, 0.10, 0.33),
                                    stall_period=15.6), {}
    for C in (215, 225, 228, 232, 236, 240):
        yield "concurrency, Decode output stalls", dict(CAL, C=C), {}
        yield "concurrency, no output stall (GC frozen)", dict(NOSTALL, C=C), {}
    for cap in (44, 52):
        for C in (225, 240):
            yield f"cap {cap}, no output stall", dict(NOSTALL, C=C, cap_per_rank=cap), {}
    yield "no output stall, start hiccup removed", dict(NOSTALL, start_extra=(0.0, 0.0, 0.0)), {}
    yield "no output stall, burst first token without the hold", dict(NOSTALL, start_cut=0.40), {}
    for f, d in ((1.0, 1.0), (0.94, 0.95)):
        for C in (225, 232):
            yield f"Prefill x{f}, Decode x{d}", dict(NOSTALL, C=C, P=tuple(p * f for p in P_R2), step_scale=d), {}
            # Decode on the head node and two Prefill pipelines on the tail node (tail node ~2.6% slower at the cap).
            yield f"Prefill x{f}, Decode x{d}, Decode on g1", dict(NOSTALL, C=C, P=tuple(p * f for p in (0.1172, 0.1203, 0.1203)), step_scale=d * 0.986), {}
    for gain in (0.019, 0.03, 0.045):
        for C in (225, 240):
            yield f"3-request microbatch, {gain:.1%} less work per token", dict(NOSTALL, C=C, mb=3, P=tuple(p * 1.5 * (1 - gain) for p in P_R2)), {}
    for C in (225, 240):
        yield "async depth 2 (loop ratios of 40825417205)", dict(NOSTALL, C=C, lead_r=1.216, release_r=4.838, kvready=0.47), {}
    # Score surface: Prefill period and Decode step as factors, concurrency, Decode cap, with and without the stalls.
    for stall in (True, False):
        for f in (1.0, 0.98, 0.96, 0.94, 0.92):
            for d in (1.0, 0.98, 0.96, 0.94, 0.92):
                for C in (215, 225, 232, 240):
                    for cap in (44, 48, 52):
                        yield "surface", dict(CAL if stall else NOSTALL, C=C, P=tuple(p * f for p in P_R2), step_scale=d, cap_per_rank=cap), dict(
                            pf=f, ds=d, cap=cap, stall=stall)
    for f in (0.97, 0.94, 0.92):
        for d in (0.95, 0.92):
            for C in (232, 240):
                for cap in (48, 52):
                    for b in (0.35, 0.75):
                        yield "slope", dict(NOSTALL, C=C, P=tuple(p * f for p in P_R2), step_scale=d, cap_per_rank=cap, b=b * 1e-3), dict(
                            pf=f, ds=d, cap=cap, stall=False, b=b)
    for stall in (True, False):
        for C in (225, 232, 240):
            yield "Prefill x0.97, Decode x0.95", dict(CAL if stall else NOSTALL, C=C, P=tuple(p * 0.97 for p in P_R2), step_scale=0.95), dict(stall=stall)


def evaluate(item):
    name, kw, tags = item
    o = avg(seeds=6, **kw)
    return dict(name=name, C=kw["C"], **tags, **{k: round(o[k], 2) for k in KEYS})


if __name__ == "__main__":
    import json
    import sys
    from multiprocessing import Pool
    from pathlib import Path
    with Pool() as pool:
        rows = pool.map(evaluate, list(scenarios()))
    for r in rows:
        if r["name"] not in ("surface", "slope"):
            print(f"{r['name']:58s} C{r['C']} {r['score']:8.1f} burst {r['burst_pass']:5.1f} TTFT fail {r['ttft_fail']:5.1f} ITL fail {r['itl_fail']:5.1f} ITL {r['itl_mean']:5.2f}/{r['itl_p99']:5.2f} ms")
    Path(sys.argv[1]).write_text(json.dumps({"schema": "pp4-serving-model/v2", "measured": MEASURED, "rows": rows}, indent=1) + "\n")
