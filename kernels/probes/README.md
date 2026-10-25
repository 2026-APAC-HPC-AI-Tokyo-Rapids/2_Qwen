# Genkai kernel probes

Run **Genkai kernel probe** from GitHub Actions, selecting the commit/branch to test. The `benchmark` input selects a preparation script under `kernels/probes/`; the common runner uses `gpu_count=1` for single-GPU probes or `gpu_count=4` for `prepare-megamoe-full.sh`, and executes the resulting `source/probe`. The script may compile native CUDA or prepare a container command. MegaMoE math is the default. Leave `candidate_patch` empty for a control/control check, or supply one repository-relative patch path such as `patches/deep-gemm-overlay/example.patch`. The workflow does not trigger the serving workflows or an image build.

The default `prepare-megamoe-math.sh` extracts the Prefill FC1 and FC2 math loops from the pinned serving header plus its configured overlays. Each phase continuously processes 32 experts with 1024 received tokens each (3072 FC1 tiles, 4096 FC2 tiles), using the serving TMA stage layout and register budgets. Candidate patches must change only those math bodies; changes to loaders, resource declarations or other paths are rejected before submission. The probe writes FP32 outputs for validation. Routing, NVLink, SwiGLU, combine and serving are outside this probe, so its time is not a prediction of full MegaMoE or serving throughput.

The login node extracts headers from the already deployed Genkai copy of `configs/firmus-framework.json`'s SGLang image and compiles with CUDA 12.9.1 / GCC 13. It checks the sidecar identity, image size and extracted base-header hash, rather than rescanning the entire image. Results appear in the Actions summary; no artifact archive is uploaded.

PJM receives `b-batch`, `gpu=1`, and a five-minute limit. There is no node or MPI allocation request. Three processes each check occupancy, changed-input graph replays, full bit equality, and a sampled independent arithmetic reference, then take interleaved timings per variant. Each process has 90 seconds including container startup and cold JIT compilation; a failed process stops the job. After terminal-state confirmation, the workflow removes its temporary run directory. No Firmus or Aspire job is submitted.

MegaMoE findings on 2026-10-24 (H100, math probe only):

| Candidate | Paired result | Decision |
| --- | --- | --- |
| Three-stage loop unroll | FC1 −1.65% to +0.36%; FC2 +2.25% to +5.11% latency | Reject; [run 40003839043](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40003839043) |
| Earlier consumed-stage release, both phases | FC1 −2.08% to −4.64%; FC2 −0.45% to +0.50% | Keep only FC1; [run 40004596361](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40004596361) |
| Earlier release, two-warpgroup FC1 only | 950.624 → 912.992, 927.744 → 904.352, 926.112 → 900.480 µs (−2.52% to −3.96%) | Candidate for full validation; [run 40005113228](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40005113228) |

The retained `patches/deep-gemm-bench/prefill-fc1-early-release/prefill-fc1-early-release.patch` waits for the current WGMMA and releases its TMA stage before waiting for the next stage's scales. It changes no arithmetic or resource allocation: all output bits match, sampled reference error is below 0.002, registers remain 168, local memory is zero, and dynamic shared memory remains 225472 bytes. The unchanged FC2 control in the final run varied by −0.02% to +2.74%, so treat the speedup magnitude as provisional. Six FC1 process trials across the two release runs improved. The patch was enabled in the historical integrated candidate; it is now isolated as a kernel-bench candidate, outside the serving overlay; the serving result below does not establish an isolated FC1 benefit.

## Integrated Firmus result, 2026-10-24

Final authorized [run 40023670050](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40023670050), commit `b39a1b3cecd81822d6fe31a964da99fb8812e5df`, reused image `40012459952` (SHA-256 `ba57eee78bc8c17b792c45f9018e8c5f73b392e3cdf0146ff85e5f64bff2d068`). Two nodes, three TP4/DP4/EP4 Prefill workers and one Decode worker, cap160, client concurrency240. Candidate includes FC1 early release, N224 FA3 with group128 FP8 output, input norm/quant, dense `compiled_dims="nk"`, and the previously tested cap160 components. The slower QKV epilogue fusion is excluded.

Before submission, [CPU CI 40022997208](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40022997208) passed. [Genkai 40023005079](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40023005079) passed all three trials of `prepare-prefill-serving.sh`: actual Qwen → RadixAttention → FA3 → O, raw/FP8-tuple QKV, Decode fallback, and graph-configuration fallback matched the original outputs bitwise for lengths [17,13], [3986], and [2048,2048]. This is a one-layer entry-point check, not full distributed serving or CUDA graph capture.

Normal official measurement completed: **8639.35 output tok/s**, eligible, 5518 requests, good fraction **97.6622%**, zero request errors and OSL mismatches. TTFT mean/p99 **2.075/4.433 s**; ITL mean/p99 **16.317/19.033 ms**. Score is 0.50% below the recorded cap128 main result (8682.63), and 0.40% below the prior cap160 result (8674.09). These are historical comparisons, not a paired repeated experiment. No demonstrated serving gain; main was not changed.

The separate profile client and capture both returned zero; validation passed for all four workers, 16 GPUs, and 16 CPU traces. Per-worker aggregate kernel means for Prefill were MegaMoE **1545.9–1555.3 µs**, fused N224 FA3 **437.7–442.9 µs**, QKV **245.8–247.8 µs**, O **214.6–215.9 µs**, QK-norm/RoPE **135.0–136.6 µs**, and the new norm/quant kernel **31.3–31.5 µs**. New kernels executed in serving, but this profile does not split MegaMoE's FC1/FC2/wait time or measure their FP8/HBM utilization. Kernel sums overlap across streams/ranks and are not batch critical-path latency. The normal score pass had tracing disabled but used the Nsight launcher. Raw profile remains in the run's one-day artifact; no home archive was created.

Earlier integration failures: run 40008291392 used the wrong compile-gate patch directory; run 40012459952 built successfully and passed the four-rank numeric gate, then failed on an invalid `get_server_args` import. Fixed the import, removed the obsolete `ForwardBatch.attn_backend` access, and kept the FP8 tuple keyword out of the graph ABI before the successful final run. Probe-only failures were exhaustive DeepGEMM warmup, missing TP1 group setup, and one initialization timeout; the completed three-trial test used real single-rank process groups.

## QKV retuning on Genkai H100, 2026-10-24

The retained probe candidate is **stock GEMM plus fast standalone postprocessing**; the improved GEMM fusion remains a comparison. The winning postprocessing was subsequently integrated into the serving branch as described below; main is unchanged. The same GitHub one-GPU route runs `kernels/probes/prepare-qkv-fusion.sh`.

The deployed dense tile is **BM256/BN128/stages3/cluster2, A multicast**. The first retuning attempt misread M/N and tried BM128/BN256, causing register spills. Correcting M improved the exact shared-memory fusion, but [run 40048324159](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40048324159) still measured 433–438 µs against stock 341–348 µs. A direct-register variant failed a bitwise check at 3986 rows and was removed.

The main successful change replaces CUDA 12.9's software `__NV_NOSAT` FP8 conversion, which goes through double, with hardware `__NV_SATFINITE` conversion plus exact overflow/NaN repair. Values above the 464 rounding midpoint become signed NaNs; input NaNs become canonical positive NaNs. BF16 rounding, RMSNorm reduction order, RoPE, contiguous Q/K/V and paged KV writes remain unchanged. Using this converter without GEMM fusion permits five resident postprocessing CTAs per SM, versus one for the fused kernel.

Paired results below are medians of six process medians from [40049422662](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40049422662) (`277b500`) and independent repeat [40050521422](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40050521422) (`90999c3`). Each process uses 25 interleaved forward/reverse-order samples, 12 graph calls per sample, and four rotating input sets. Same pinned image as above, one H100, no profiler. All times are µs and include all QKV postprocessing/cache writes except the last column.

| Rows | Stock GEMM + post | Old BM128 fusion | Improved BM256 fusion | Stock GEMM + fast post | Reduction vs stock | Fast post alone |
|---|---:|---:|---:|---:|---:|---:|
| 3986 | 356.29 | 466.13 | 326.55 | **293.41** | **17.65%** | 63.78 |
| 4027 | 357.89 | 467.92 | 327.48 | **294.52** | **17.71%** | 64.50 |
| 4096 | 361.85 | 463.02 | 328.87 | **297.37** | **17.82%** | 65.57 |

Every process passed exhaustive bit equality for all 65,536 BF16 input patterns in both packed FP8 positions, including subnormals, signed zeros, infinities and NaNs. Full Q/K/V and cache buffers match the reference bitwise at 129/3986/4027/4096 rows; changed-input/position/cache-location graph replays pass at 129/4027. Dense outputs also pass sampled independent FP32 references. Resource measurements: fast post 48 registers, zero local/shared memory, five resident CTAs; fused GEMM 168 registers, zero local memory, 216240 shared bytes, one resident CTA. A token/head CTA layout [40050012335](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40050012335) was slower (297–302 µs total), so the 32-token layout was restored.

CPU CI passed for the repeated candidate ([40050515096](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40050515096)); all GPU jobs are terminal and their temporary run directories were removed. The earlier 55-second setup failure was confirmed as PJM exit 124, so the common route now allows 90 seconds per process and five minutes total. No further Firmus submission was made. This QKV-local reduction is not a measured Prefill batch or serving-score improvement.

## QKV serving integration, 2026-10-24

User-authorized [Firmus run 40059609107](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40059609107), commit `bf59ba6c8d0e138c7992eaaf835b8a8f62d53edc`, adds only the fast QKV postprocessing overlay to the earlier serving configuration and pinned image. Before submission, [Genkai 40059262101](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40059262101) passed three production integration processes and CPU CI passed. Eight prologue cases cover 0–4096 rows, int32/int64 indices, packed/separate inputs and changed-input graph replay. Actual Qwen/cache/FA3/O outputs match bitwise. Production prologue median: **109.244 → 66.056 µs**. The preceding preflight found and fixed a missing forward context in the harness and an outdated overlay-order test; serving kernel code was unchanged by those fixes.

Normal score **8793.137 tok/s**, eligible, versus 8639.347 previously (**+1.78%**, historical single-run comparison). Raw output throughput **9009.747 tok/s**; 5613 requests, 5478 good (**97.5949%**), no errors or OSL mismatches. TTFT mean/p99 **1.927/4.350 s**; ITL **16.577/19.370 ms**. Per-request records for the normal measurement phase show **121 TTFT-only (>3 s)** and **14 ITL-only (>20 ms)** misses. All 121 TTFT misses belong to the initial 240 requests: 239 arrive within 100 ms and all 240 within one second. A Prefill rank's queue grows to 14 during that burst. These requests remain included in the official score.

The separate profile validated all **16 GPUs and 16 CPU traces**. Prefill aggregate kernel means: fast QKV post **65.16–65.73 µs** (previously135.0–136.6), MegaMoE **1549.98–1561.37 µs**, FA3 **443.37–447.01 µs**, QKV GEMM **247.97–249.41 µs**, O GEMM **216.20–217.55 µs**, norm/quant **31.43–31.52 µs**. QKV activation is confirmed; MegaMoE has no clear aggregate improvement. Roughly70 µs saved per layer is 6.6 ms over94 layers, about2.6% of the earlier255 ms Prefill estimate, not a measured critical-path speedup. Normal effective Decode concurrency median154/p99159 is close to cap160; ITL rose from16.317 to16.577 ms. These observations suggest limited Decode headroom, but do not isolate its hard capacity or MegaMoE's internal stalls.

Workflow and Slurm completed successfully, runtime cleanup passed, no new home archive was created. Only the existing one-day Actions artifact retains raw output; main remains unchanged. At the measured good fraction, reaching9000 requires about **2.35%** more throughput; the former +5–8% overall forecast was not demonstrated.

The merged runner retains the calibration protocol below, with the upstream five-minute one-GPU PJM limit (ten minutes for four GPUs).

## Decode calibration against Firmus

Select either benchmark below, with **`candidate_patch` empty**, on the commit
containing these files. Run them separately through the existing workflow;
neither preparation script builds an image or submits a job itself.

| benchmark | Scope |
| --- | --- |
| `kernels/probes/prepare-calib-dense.sh` | M128, QKV N9216/K4096 and O N4096/K8192 |
| `kernels/probes/prepare-calib-fa3.sh` | B32 TARGET_VERIFY, Q4, 64 Q heads / 4 KV heads / D128 |

The current image `39932105942` has the required DeepGEMM host override and
FA3 companion. Its host override requires M≤128 and 132 SMs; the companion
requires B32–36. These calibration probes use **M128 and B32 only**. M160,
B40 and p128split2/3 are excluded. The container has CUDA13.0 / Torch cu130;
loading the host's CUDA12.9.1 module does not recompile the container's AOT
FA3 binary. Interpret this as an H100/container calibration, and keep the
H200 power/clock and four-GPU serving differences in view.

Dense reuses `select-firmus-decode-dense.py`'s preparation, CUDA graph capture,
timing and alternating order: four prequantized weight/input sets, 20 calls
per graph, 25 samples per turn, two blocks with reversed candidate order.
Each candidate turn is paired with stock. QKV candidates are stock,
BM128/BN80/stages7, BM64/BN144/stages5, BM128/BN144/stages5,
BM128/BN80/stages7/multicast2-on-A, and Triton64/128/K128/stages3/warps4.
O uses its Firmus selector geometries: stock, BM128/BN32/stages10,
BM64/BN64/stages9, BM128/BN64/stages8,
BM64/BN64/stages13/multicast2-on-A, and the same Triton candidate.
All non-multicast configurations use multicast1/on-A0.
The BM128/BN144/stages5 QKV row is diagnostic: the supplied Firmus result
for that configuration was M160, so it has no supplied M128 reference.

FA3 reuses `select-firmus-decode-fa3.py`'s 94-layer graph and paired CUDA-event
timing: four KV sets, shared live sequence metadata, one scheduler preparation
per 94 layers, 20 alternating stock/candidate pairs per block, two blocks.
The original calibration candidates were stock, control96, p128s2, t64s2 and t64s3. Stock samples are
also retained separately for each candidate; its displayed aggregate pools
those paired samples. Times are **per layer**, including amortized scheduler
preparation. This follows B4 metadata sharing rather than 94 scheduler calls.

Both probes reuse the existing FP32/stock error tolerances and changed-input
graph checks. A candidate that fails its numeric check makes the calibration
fail; no failed candidate is silently omitted. Results include median, p10,
p90, candidate/stock ratio (below 1 is faster), and both block ratios. Raw
sample and numeric-check JSON goes to the PJM/Actions log. The summary prints
the candidate table; no new artifact archive is uploaded.

Calibration uses one process with a **165-second timeout**, retaining the
five-minute one-GPU PJM limit. Other one-GPU probes use three 90-second trials; four-GPU probes use three 180-second trials. The two measurement blocks provide the
within-process repetition; repeat the workflow manually to check run-to-run
stability before drawing a calibration conclusion.

Timing budget (estimate, **not GPU-validated**): dense's timed graph work is
about 0.3–0.7 seconds using the supplied Firmus latencies; FA3's timed work is
about 1.5 seconds at 50 µs/layer. Imports, Enroot setup, cold JIT, numerical
references and graph capture dominate. Budget roughly 30–120 seconds for
dense and 20–90 seconds for FA3, within the 165-second process limit; cold
JIT completion on Genkai remains to be checked by the parent workflow.

For comparison, Firmus run 39901974117 at M128 had QKV stock 17.0 µs,
BM128/BN80/stages7 15.2 µs (0.894×), BM64/BN144/stages5 16.5 µs (0.971×),
multicast candidates 15.5–20.2 µs and Triton 30.8–31.6 µs. O stock was
15.2 µs, other DeepGEMM candidates 15.1–19.6 µs and Triton 50.3 µs.
FA3 B32 stock was 49.5 µs, control96 50.6 (1.022×), p128s2 46.1 (0.931×),
t64s2 54.4 (1.099×) and t64s3 53.2 (1.075×). Compare rankings and ratios
with these same-shape references; do not treat H100 absolute times as
Firmus serving throughput predictions or infer a serving win from one run.

CPU protocol/aggregation fixtures: `python3 kernels/probes/test-calibration.py`.

## Single-server EAGLE3 acceptance on Genkai

Select `benchmark=kernels/probes/prepare-eagle-accept.sh`, `gpu_count=4`, and an
empty `candidate_patch`. Login preparation downloads the pinned Nebius BF16
draft and Instruct-2507 tokenizer, applies the exact Firmus `prepare_eagle.py`
config adaptation, and checks the existing image and FP8 target. Compute runs
offline: one TP4 SGLang server, EAGLE3 steps3/topk1/draft4, context8192, FP8 KV,
concurrency32, no PD/DP attention/MegaMoE/runtime overlays, and no CUDA graphs.
This measures acceptance only; its timings and throughput are not meaningful
performance comparisons or official scores. AIPerf uses the official synthetic
prompt settings, with no warmup, 20 seconds of measurement and 30 seconds grace.

The 570-second calibration process budget leaves 30 seconds of the four-GPU PJM
allocation for finalization. Historical TP4 loads took 360–368 seconds in runs
35105770792/35108498091; graph capture took another 147–150 seconds. Disabling
graphs and exhaustive DeepGEMM precompilation leaves an estimated 560 seconds
including client setup and cleanup; cold on-demand compilation remains.
Startup must finish by 450 seconds, and collection stops at 555 seconds to allow
process cleanup. Budget overruns fail without `PASS`. All server-info states
and `accept len` log samples are collected; `PASS` reports bonus-inclusive X,
accepted draft X−1, completed requests and startup seconds. Raw server-info,
AIPerf export and server/client logs are also printed into Actions/PJM logs
before the existing workflow removes its runtime directory. The calibration
summary has no acceptance table; read the `PASS`/`EAGLE_ACCEPTANCE` log lines.
CPU checks: `python3 kernels/probes/test-eagle-accept.py`.

## Calibration against Firmus, 2026-10-24

Runs [40043191081](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40043191081) (dense, M128) and [40043193928](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40043193928) (FA3 TARGET_VERIFY, 32 requests, 94-layer graph) measured cases whose Firmus H200 results were already known from the in-job selections of runs 39901974117 / 39970713547. Ratios are candidate / stock (median).

| case | candidate | Firmus H200 | Genkai H100 |
| --- | --- | ---: | ---: |
| QKV M128 | BM128/BN80/7 (Firmus winner) | 0.894 (15.2 / 17.0 µs) | 1.003 (23.7 / 23.6 µs) |
| QKV M128 | BM64/BN144/5 | 0.971 | 0.985 |
| QKV M128 | BM128/BN80/7, multicast A | 0.912 | 0.993 |
| QKV M128 | Triton 64x128 | 1.81 | 1.42 |
| O M128 | BM128/BN32/10 | 1.02 | 1.03 |
| O M128 | BM128/BN64/8 | 1.22 | 1.08 |
| O M128 | multicast A | 1.27 | 1.04 |
| O M128 | Triton 64x128 | 3.31 | 2.41 |
| FA3 B32 | p128s2 (Firmus winner) | 0.931 (46.1 / 49.5 µs) | 1.010 (71.1 / 70.0 µs) |
| FA3 B32 | control96 | 1.022 | 1.019 |
| FA3 B32 | t64s3 | 1.075 | 1.052 |
| FA3 B32 | t64s2 | 1.099 | 1.056 |

What the comparison shows:
- Genkai ranks the clear losers correctly (Triton, the N64 TMA FA3 tiles, BM128/BN64 for O): every candidate that is 5% or more slower on Firmus is also slower on Genkai.
- It misses both Firmus winners. They are memory-path improvements for Decode-sized (memory-bound) shapes. On H100, HBM is 3.35 TB/s against 4.8 TB/s on H200, and the stock kernels are already about 1.4x slower in absolute terms, so the same change does not help.

Use the one-GPU probe as a correctness gate (bit equality, changed-input graph replay, boundary shapes) and to drop candidates that are clearly slower. Do not use it to reject a candidate that is neutral or slightly slower on Genkai. Decide performance on Firmus: the in-job selection in the serving job, or the one-node kernel bench. Compute-bound Prefill kernels have not been calibrated here.

## Merged probe and bench configuration, 2026-10-24

Serving retains the cap160 configuration, FP8 integration and fast QKV overlay, but no FC1 early-release overlay. The Firmus one-node bench compares `prefill-fc1-early-release` (no env), `prefill-iwg-a` and `prefill-iwg-b` against that common control. Only the control has a timeline marker: eight Slurm steps in total, below the twelve-step limit. Rejected scale-share and unroll probe patches remain under `patches/deep-gemm-probes/`, outside candidate discovery and still accepted by the Genkai candidate-patch path check.

FA3 calibration now takes the current selector's four companion IDs: control96, p128s2, p128split2 and p128split3. The historical table above retains the removed t64 measurements; split candidates have no paired historical reference in that table. The calibration still runs B32 with shared metadata and the same numeric checks.

Both timeline instruments are retained as alternative overlays on the same control, not composed: the Firmus control uses the main v2 per-WG phase instrument; Genkai EP4 uses the upstream producer/transfer plus WG0/WG1 instrument, now ABI v3 to avoid its original v2 collision. All recording points remain in their respective patches. The shared decoder accepts v1/v2/v3 and the EP4 summary retains producer/transfer and both-WG records. A/B have no timeline marker.

## ASPIRE single-GPU FA3 probes

The **ASPIRE kernel probe** workflow (`aspire-kernel-probe.yml`) accepts
`benchmark=kernels/probes/prepare-fa3-prefill.sh` (M1) or
`benchmark=kernels/probes/prepare-fa3-prefill-profile.sh` (M2), with
`candidate_patch` empty. Select the ref containing this workflow and its
preparation scripts. GPU count is fixed at one; other benchmarks and nonempty
patch inputs are rejected before submission.

The login node provisions the ASPIRE copy of the image pinned in
`configs/firmus-framework.json`, then compiles with host CUDA 12.6.2 / GCC 13.3.
Genkai retains CUDA 12.9.1 / GCC 13 and its existing image and Enroot setup.
Preparation reads `scripts/build-env.sh` and `sglang.sqsh` from the runtime;
direct Genkai preparation still supplies the original defaults.
M2 extracts Nsight Compute 2025.3.1 from the pinned image and uses it for both
collection and report import. The existing common M1/M2 probe commands and
numerical gates are unchanged. Treat results from different toolchains and
clusters separately.

PBS receives project 59010014, queue `normal`,
`select=1:ngpus=1:ncpus=14:mem=180gb`, and a five-minute walltime, without
exclusive-node placement. Both protocols use one calibration trial with a
165-second process limit. At 128 SU per GPU hour, five minutes costs about
10.67 SU per run; the 165-second process alone is 5.87 SU. Actual scheduler
billing includes job startup and teardown. Summary records project balance
before submission and after observation using `myprojects -p 59010014`;
concurrent project activity can affect the balance difference.

Logs, provenance, sidecar identity, M1 JSON and M2 CSV / binary `.ncu-rep` are
recovered in the `aspire-kernel-probe-<run_id>-<attempt>` Actions artifact before
runtime deletion. Cleanup requires a terminal PBS state and successful artifact
upload. An active or unobservable job is preserved, and no job is cancelled.
The parent must validate GPU execution, NCU counter access and actual runtime
through workflow dispatch; login-only compilation and CPU fixtures do not
establish GPU correctness or performance.
