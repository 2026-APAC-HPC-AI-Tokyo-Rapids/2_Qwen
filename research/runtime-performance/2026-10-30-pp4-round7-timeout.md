# PP4 round 7: FA3 pipe adopted, job timed out before the score (2026-10-30)

- Branch: `perf/pp4-r7-20261030`.
- Serving commit: f2d1877. Build and serving ran in one run.
- Topology: P3-TP1-PP4 / D1-TP4-DP4-EP4.
- Concurrency: 232. Decode cap: 48 per DP rank.

| run | what | result |
|---|---|---|
| 41040348302 | build + PP4 serving of f2d1877 | image OK. Slurm 27605 **TIMEOUT** at 17:13, during the AIPerf main phase; no score |

## 1. Gates

| candidate | gate result | expected |
|---|---|---|
| Prefill FA3 `n192-direct-o-varlen-pipe` (fixed in d8ba09f, gated in 50ab866) | **adopted**. It saved at least 8.1 µs and on average 15.0 µs per layer call, on all 12 GPUs and the 3986 / 4027 / 4096 / varlen lengths (median 404.5 µs). The 1200-call endurance gate passed on every GPU. It then served about 1480 PP0 microbatches per worker without a stall | −0.6 ms per period |
| MegaMoE `four-stage-lean-split-rcp` (shared FC1 reciprocal) | **not adopted**. It was 0.3% faster in total (61111 against 61295 µs). Its error against the FP32 reference was the same as the control's (relative L2 1.86e-2 against 1.86e-2). But it failed the endurance gate on every GPU: the 600th output was not bit-equal to the variant's own first output, on all four routings | −0.17..−0.36 ms |
| Prefill MoE input `onepass-split` | below margin: 88.9 → 90.2 µs at 8064 rows | −0.3..−0.6 ms |
| Prefill Q/K norm `fp32-heads4` / `fp32-heads8` | below margin: 72.0 → 81.0 µs at 8064 rows (heads4); 36.9 → 37.9 µs at 4032 rows (heads8, which saved 4.3 µs at 8064) | −0.36..−0.53 ms |
| Decode MoE input, QKV BLOCK_M 192, PP4 copy engine | the same winners as in 41005872470 and 41020974704: `fused`, `QKV>128:192,80,5,1,0` and `copy` | — |

What the results mean:

- **The FA3 pipe hang of 40991744289 is fixed.** The fixed row passed the 1200-call gate and then served under load.
- **The shared reciprocal is nondeterministic.** The endurance gate fed the same input to the same variant, and the output bits changed. A race inside the kernel is the likely reading. It was not localised on a GPU. The candidate is dropped.
- **Both memory-bound rewrites lost.** The pinned one-pass and FP32 Q/K writers are already close to their memory floor at these shapes. Splitting or batching heads added more cost than it removed.

## 2. Prefill period

From each run's Prefill logs: the median over every PP0 microbatch of two 8064-token requests.

| run | PP0 period, prefill-0/1/2 (ms) |
|---|---|
| 41020974704 (round 6, 9828.95) | 115.0 / 115.5 / 118.6 |
| 41040348302 (round 7, FA3 pipe) | **114.6 / 115.0 / 118.3** |

- The period fell 0.3–0.5 ms (−0.3%). That is the FA3 pipe saving: 15 µs × 24 layers plus the varlen pair.
- The Decode running requests were 43.3 per rank on average, with a maximum of 48, the same as in round 6.
- At the score's sensitivity to the period, this is worth about +0.3%: about 9,860 against 9,832.74. That is an estimate from the period alone, not a measured score.

## 3. Why the job timed out

| time (UTC) | event |
|---|---|
| 11:22:25 | job start |
| 11:22:25–11:29:43 | Prefill-GPU selection chain (7.3 min): MegaMoE selection → FA3 screen → MegaMoE and FA3 endurance gates → elementwise selection |
| 11:34:56 | workers ready (freeze-gc) |
| 11:36:17 | AIPerf warmup start |
| 11:39:25 | 17-minute limit |

- The chain was estimated at 4.1–4.8 min. It took 7.3 min.
- In 41005872470 the job finished in 12:39.
- The three endurance gates and the elementwise step added about 3 minutes of serial time before worker start. That left no room for the 60 s warmup and the main phase.

## 4. Consequences

- **Pin every selection that has settled, and run no in-job selection in the next PP4 job.**
  - Prefill FA3: the pipe row.
  - MegaMoE: four-stage-lean-split.
  - Decode: MoE input `fused`, QKV `192 × 80`.
  - PP4 transport: `copy`.
- **Remove the losing candidates** from the image: the shared reciprocal, onepass-split and fp32-heads.
- A future candidate gets its selection and endurance gate only if the job timeline still leaves the main phase inside 17 minutes.
- 10,000 still needs about −1.4% of the period, about 1.6 ms. MegaMoE (64.5 ms per stage) and FA3 (20.8 ms) remain the only kernels large enough.
