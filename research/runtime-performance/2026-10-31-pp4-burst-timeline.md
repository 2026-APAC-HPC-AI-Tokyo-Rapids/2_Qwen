# PP4 start burst: from the AIPerf send to the first token (2026-10-31)

- Branch: `perf/pp4-burst-r10-20261031`, based on b9f5f45.
- Source and log analysis only. No job was run.
- Runs analysed:
  - round 9, 41135856718 (unofficial 9883.51)
  - round 8, 41073991312 (9883.36)
- Code: the image source with the 29 PP4 overlays of `configs/firmus-prefill-pp4.json` applied in order. File and function names below refer to that tree.

## 1. Summary

- **The `Prefill batch` lines are printed at a stage's result step, not at batch formation.** The PP0 line at +0.288 s is the result of the first microbatch: all four stages have finished its forward by then. The brief's question "why does the first microbatch start at +0.288 s" therefore rests on a misreading.
- **The client's first token is the Prefill completion, not the Decode's first token.**
  - The router (`patches/sglang-router/prefill-load-lifetime.patch`, `try_join!`) streams the Decode response only after it has read the whole Prefill response body.
  - The Prefill responds when its KV release consensus frees the request.
  - Measured offsets (count-matched over 4500 steady-state requests):
    - client first token minus Prefill completion: 0.000 s
    - Decode first output to client first token: 0.316 s
- **In steady state and in the burst, each microbatch follows the same timeline.** Times are relative to stage 0's result step:

  | event | time after stage 0's result step |
  |---|---:|
  | stage 1's result step (its KV send) | +0 ms |
  | the last stage's result step | +20 ms |
  | stage 2's result step | **+115 ms (one period late)** |
  | Decode's first output | +150 ms (34 ms after stage 2's KV) |
  | Prefill completion = client first token | **+466 ms = 4 iterations** |

  The 4 iterations are 2 of consensus lag, 1 for the poll, and 1 lost because every poll runs right after the KV send it should see.
- **Shortening only the release (or removing the router hold) trades TTFT for ITL and does not pay.** The end of each stream stays where it is, so ITL rises by d/199 for every request. Replayed on the actual exports, removing the whole hold costs −9 to −11 good requests.
- **What pays is moving the Decode start and the release together.** Implemented (overlay only, no rebuild):
  - every stage after the first takes its result before its launch
  - every stage polls its transfers after its launch
  
  Expected effect:
  - Decode's first output −0.095 s
  - release −1 iteration (−0.115 s)
  - burst intercept −0.115 s, about +6 good requests (+0.10%, ~9893)
  - steady-state hold 0.316 → ~0.30 s, so TTFT and ITL stay where they are

## 2. What each log line means

| line | where it is printed | when its fields are taken |
|---|---|---|
| `[.. PPk] Prefill batch, #new-seq ...` | stage k, `process_batch_result_disagg_prefill` → `metrics_reporter.report_prefill_stats` (disaggregation/prefill.py), inside `_qpp_result_step` | `#pending-token`: at formation (`_get_new_batch_prefill_raw` → `PrefillStats.from_adder` → `_get_num_pending_tokens`, after the batch left the waiting queue). `#queue-req`, `#inflight-req`, `token usage`: at print time. |
| Prefill `"POST /v1/chat/completions" 200` (uvicorn, second resolution) | `_handle_streaming_request` awaits the first chunk before it returns the response (serving_chat.py) | The Prefill produces its only chunk when the request finishes, at release. |
| Decode `"POST ..." 200` | same | The Decode's first output (first token). |

Stage 0's result step for microbatch i runs in iteration i + 4 (`pp_loop_size` 5). It comes after that iteration's formation and launch (`_qpp_step`), and it waits for the last stage's published result (`_qpp_batch_result`).

- So the first PP0 line shows `#queue-req: 69`: the 69 requests still waiting after microbatch 5 was formed.
- It also shows `#inflight-req: 1`: microbatch 1's request had just entered the inflight queue.

## 3. Timeline of the first burst microbatch (prefill-1, round 9)

t0 = profiling start, 07:04:19.025 (+08:00). The other workers are within 10 ms.

| t (s) | event | code | evidence | status |
|---:|---|---|---|---|
| +0.002 to +0.004 | AIPerf sends the first requests | — | `request_start_ns` | established |
| ~+0.05 to +0.08 | stage 0 forms and launches microbatch 1 (1 request) | router → `serving_chat` (template, gigatoken) → ZMQ → `_qpp_step` (recv, `_qpp_admit_bootstrapped`, formation, `_qpp_launch`) | back-solved from the next two rows | inferred |
| ~+0.05 to +0.27 | four 1-request stage forwards at near-idle clocks (t1 ≈ 50–60 ms) | — | telemetry: 1980 MHz idle; 1440–1965 MHz at +0.48 s | inferred |
| **+0.288** | stage 0's result step for microbatch 1; its KV is sent | `_qpp_result_step` (after the launch of microbatch 5) | PP0 line, `#pending-token: 0`, `#queue-req: 69` | established |
| +0.415 / +0.422 | result steps of stage 1 and the last stage | stage 1 after its launch, the last stage before it | PP1 / PP3 lines | established |
| **+0.529** | stage 2's result step (the last KV) | after its launch of microbatch 5 | PP2 line | established |
| ≤ +0.602 | Decode's first output for the first three requests | Decode transfer queue → first decode step | Decode 200 OK lines between ms-stamped lines | established (bound) |
| +0.504, +0.636, +0.754, +0.868 | stage 0's result steps for microbatches 2–5 (iterations 5–8) | — | PP0 lines | established |
| **+0.868** | release of microbatch 1 in iteration 8, i.e. result + 4 iterations | `_qpp_step`: transfer poll → consensus (lag 2) → `process_disagg_prefill_inflight_queue` | first Prefill 200 OK lines follow the +0.868 PP0 line | established |
| **+0.870** | client first token (+0.883 and +0.889 for the other workers) | router `try_join!` releases the Decode stream | AIPerf | established |

## 4. Answers to the four questions

### 4.1 Why the first PP0 microbatch "starts" at +0.288 s

It does not start then; that is its result step.

From the two result times, with t1 and t2 the 1- and 2-request stage times at startup clocks:
- result(mb1) = L + 4·t1 + ε = 0.288 s
- result(mb2) − result(mb1) = 4·t2 − 3·t1 = 0.216 s

With t1/t2 between 0.55 and 0.6, this gives L ≈ 0.05–0.08 s for the stage-0 launch, about 50–75 ms after the send. Inferred.

That span covers:
- the router
- HTTP parsing
- the chat template and gigatoken tokenization of a ~4000-token prompt
- ZMQ to the scheduler
- one scheduler iteration

The Decode side is not on this path:
- The router sends to Prefill and Decode concurrently.
- Since 8d26550, stage 0 admits a request while its bootstrap is still pending.
- Round 8, which still waited for the bootstrap (`#bootstrap-req: 9`), had the same result time (+0.294 s).

**What would settle L:** `--enable-request-time-stats-logging`, an existing server argument (observability/req_time_stats.py, `convert_to_duration`). On each Prefill stage it prints `entry_time`, `bootstrap_queue_duration`, `queue_duration` and `forward_duration`; launch ≈ entry + bootstrap + queue. It also prints `transfer_speed` and `transfer_total`, which give the transfer latency. On Decode it prints the prealloc and transfer durations.

### 4.2 Why the first microbatch has one request while 69 wait

- `#pending-token: 0` is computed at formation, after the request left the queue. So exactly one request was waiting when microbatch 1 formed.
- `#queue-req: 69` was read four iterations later.
- The waiting counts at the formation of microbatches 2–5 were 10, 25, 46 and 69. The tokenizer delivered the burst over those four iterations.
- There is no cap:
  - `PrefillAdder` is limited only by `pp_max_micro_batch_size` 2 and the 8192-token chunk.
  - The prefill delayer and the min-free-slots delayer are disabled.
- A 2-request first microbatch would gain only t1 − t2/2 (the fixed per-forward cost, a few ms). Every later microbatch already found ≥ 2 requests waiting, and stage 0's GPU never idled after microbatch 1.

### 4.3 Why the second result comes 216 ms after the first

- Microbatch 2 enters stage 0 right after the shorter microbatch 1, so it finishes at L + t1 + 4·t2. Microbatch 1 finishes at L + 4·t1.
- The difference, 4·t2 − 3·t1 = 216 ms, is the pipeline fill of the first full microbatch, with t2 ≈ 92–98 ms at startup clocks. It is not a scheduling gap.
- The next gaps (132, 118, 114 ms) approach the steady period as the clocks fall to the power cap.

### 4.4 Where the 0.58 s from the first PP0 line to the first token goes

| span | prefill-1, microbatch 1 | steady state | cause |
|---|---:|---:|---|
| stage 0's result → stage 2's result (the last KV send) | 0.241 s (longer than in steady state while the pipeline fills; not resolved) | 0.115 s | stage 2 takes its result after a launch that blocks for about a period |
| → Decode's first output | ≤ 0.07 s | 0.034 s | transfer, Decode transfer-queue poll (interval 1), first decode step |
| → Prefill completion (client first token) | ≥ 0.27 s | 0.316 s | release four iterations after the result |

How the iterations add up:
- KV of microbatch i is sent at result step i + 4.
- At iteration i + 5, the poll runs right after the send (stage 0 sends at the end of an iteration and polled at the start of the next), so it misses the completion.
- At iteration i + 6, every stage reports the transfer done.
- The consensus lag of 2 then releases it at iteration i + 8 = result + 4.
- The CPU harness on the old source reproduces this exactly under a "transfer completes during the next launch" model: r3 has its result at 5 and its release at 9.

The four stage forwards are before the PP0 line (§4.1). The Decode does not run an EAGLE3 draft prefill: the last Prefill stage runs it (`check-firmus-pp4-source.py`: "only final PP target stage executes EAGLE draft prefill"), and the Decode's first step after the KV is one verify step.

## 5. The router hold and why removing it does not pay

The end of a stream is set by the Decode and does not move. A first token d seconds earlier therefore raises the request's ITL by d/199. The table below replays the AIPerf export of each run with per-request shifts (burst = the first 232 requests by send time).

| scenario | round 9 good (Δ) | round 8 good (Δ) | TTFT fails | ITL fails |
|---|---:|---:|---:|---:|
| as served | 6116 | 6117 | 121 | 3 |
| router streams at the Decode's first output (−0.32 s for all) | 6107 (−9) | 6106 (−11) | 103 | 30 |
| release −1 iteration only (poll after launch alone) | 6121 (+5) | 6121 (+4) | 115 | 4–5 |
| Decode −0.095 s only (stage 2 result first alone) | 6118 (+2) | 6118 (+1) | 121 | 1–2 |
| **both (implemented): burst first token −0.115 s, steady hold −0.02 s** | **6122 (+6)** | **6123 (+6)** | 115 | 3 |

- Every ITL failure the hold removal would create is a steady-state request with ITL 18.4–20.8 ms.
- The 2026-10-30 structure model reached the same conclusion for lag 2 → 1 (§6, ±0).
- A hold that depends on load, deadline or position would be metric shaping, so it is not proposed.

## 6. Candidates

Scores are relative to 9883 at 6240 requests, 5 passes per 0.1 s.

| candidate | burst intercept | score | steady-state risk | files | rebuild |
|---|---:|---:|---|---|---|
| **A. later stages take the result before the launch, and the transfer poll moves after the launch (implemented)** | **−0.115 s** if a KV transfer finishes within the next launch (≥ 25 ms of eager launch); else 0 | **+6 (+0.10%)**; +2 if only the Decode part acts | GPU work unchanged. Stage 1 and 2 launches start after the published result, 1 and 2 periods before their GPUs need them. Hold 0.316 → ~0.30 s. | `patches/sglang/pp4-loop-latency.patch`, `scripts/check-pp4-direct-loop.py`, tests | no |
| B. router streams Decode without waiting for the Prefill body | −0.32 s | −9 to −11 | ITL +1.6 ms for every request | `patches/sglang-router/prefill-load-lifetime.patch` | yes (router binary) |
| C. consensus lag 2 → 1 | −0.115 s | about +4 | ITL +0.6 ms for all; stage 0 would wait on the last stage's consensus (the reason for lag 2, `_QWEN_PP_CONSENSUS_LAG` comment) | pp4-loop-latency.patch | no |
| D. first microbatch of 2 requests (wait for a second arrival) | a few ms | ~0 | none | — | no |
| E. shorten send → stage-0 launch (L ≈ 0.05–0.08 s) | ≤ 0.05 s, split unknown | ≤ +2 | depends | needs the time-stats measurement of §4.1 first | — |

Already rejected and not proposed again:
- 3-request microbatches
- concurrency changes
- stage rebalancing
- steady-state router policy changes
- the g4 swap

## 7. The implemented change

`_qpp_step` in `patches/sglang/pp4-loop-latency.patch`:

- **Result before the launch on every stage but the first.** This was `if is_last:` (1b5a2da). Stage 0 keeps its result after its launch, because its GPU runs the microbatch it launches next.
  - In the pipeline, stage k's GPU starts the microbatch of iteration i + 4 about k periods after the result of microbatch i is published, so the wait costs stage 1 and stage 2 no GPU time.
  - Stage 2's KV moves from +115 ms to about the time of stage 0's. The last KV is then the last stage's, at +20 ms. Decode's first output therefore moves from +150 ms to about +55 ms.
- **Transfer poll after the launch.**
  - The poll, `loop.tmbs[mb_id]` and the forward of the transfer list to the next stage move from before batch formation to after the launch.
  - The next stage receives the list after its own launch. That launch already waited for this stage's proxy metadata, which is sent inside this stage's launch, so there is no new wait.
  - The message order on the world gloo group is the same on both sides: requests, admissions, transfers, release.
  - Expected release: result + 3 iterations (+0.35 s) instead of + 4 (+0.466 s).
- **Unchanged:**
  - every iteration-indexed state transition
  - formation before the result step on every stage
  - the consensus lag
  - the launch, proxy and publisher paths
  - `_qpp_admit_bootstrapped`

CPU evidence:
- The 29 overlays apply in config order, and py_compile passes.
- `scripts/check-pp4-direct-loop.py`, extended:
  - asserts result-before-launch exactly for ranks > 0
  - asserts the transfer poll after the launch on every stage
  - asserts every release exactly `lag` after the first iteration in which all four stages polled the transfer done
  - for transfers that finish during the next launch, asserts a release at result + 3. On the round-9 source the same check fails with result 5 → release 9, which reproduces the Firmus timeline.
  - passes 96 iterations, in both the copy-engine and the corrupted-slot run
- `check-firmus-pp4-source.py`, `check-pp4-output-relay.py` and `check-pp4-scheduler-sync.py` pass.
- `test_pp4_loop_order.py` pins the order in the overlay text.

What only the serving run can show, and how to read it from logs it already writes (no new gate is needed: the GPU work is unchanged):
1. The PP2 lines no longer trail PP0 by a period. Expected PP2 − PP0 ≈ 0–20 ms; now 112–120 ms.
2. Prefill completion (Prefill 200 OK) − PP0 result ≈ 0.35 s instead of 0.466 s.
3. Decode first output − PP0 result ≈ 0.06 s instead of 0.150 s.
4. The PP0 period stays at 114.9 / 114.6 / 118.3 ms (round 9).

If (2) stays at 0.466 s, the transfers take longer than a launch. The change is then about +2 requests, and `--enable-request-time-stats-logging` shows the transfer latency.

## 8. Established vs inferred

**Established from code and logs:**
- §2 (the meaning of each line)
- the per-stage result offsets (rounds 8 and 9, all workers)
- client = Prefill completion
- Decode first output 34 ms after the last KV
- release = result + 4 iterations
- no microbatch cap
- the counterfactual table

**Inferred:**
- L ≈ 0.05–0.08 s and the startup stage times (from two equations, not measured)
- stage 2's lag is caused by its launch blocking, as documented for the last stage; the release receive is the alternative. The change removes both, since the result step precedes them.
- the lost fourth iteration is the send→poll race. It is reproduced by the harness model; the transfer latency itself was not measured.
- the expected gains of §6 A

## 9. Result in 41150387984 (round 10)

Official score 9896.09, against 9883.36 in round 8. Details are in `2026-10-31-pp4-round10.md`.

The four checks of §7:
1. **Passed.** PP2 now prints its result 7–15 ms after PP0 in steady state.
   - The first microbatch still shows 121 ms, while the pipeline fills.
2. **Passed: one iteration earlier.**
   - Request 0's first token moved from +0.868 s to +0.757 s (−0.111 s).
   - From the first PP0 result line to that token, the time fell from 0.58 s to 0.47 s.
   - The first PP0 result stayed at +0.287 s, so the whole shift is the release.
   - So the KV transfers finish within one launch. The −0.115 s case of §6 A applies, not the +2 case.
3. Not measured separately. Check 2 covers it.
4. **Passed.** The period is 114.9 / 114.6 / 118.1 ms.

The predicted gain was +6 burst passes; +6 were observed (TTFT failures fell from 121 to 115).
