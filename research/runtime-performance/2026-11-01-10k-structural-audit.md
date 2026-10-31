# PP4 10k structural audit and split-PV prototype (2026-11-01)

This audit rereads the issues, chats, raw request records and newest available kernel trace. It does not treat an old EP4 profile as the current PP4 execution. V1 split-PV was numerically correct but about 25% slower; three V2 CPU assembly attempts failed the pre-allocation gate. V3 passed its full CPU proof, all seven GPU numerical cases and 1,200-call endurance on both real pairs, but was **10.9–11.7% slower than the served M192/N96 kernel**, with no positive timing pair. V3 is also rejected and serving remains unchanged. The independently completed round 12 reached **9958.4323 tok/s**; this audit does not claim credit for that result or claim 10k achieved.

## Required work reduction

| Normal run | Served behavior | Official output goodput | Good / total requests | TTFT / ITL failures |
|---|---|---:|---:|---:|
| [41150387984](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41150387984) | Round 10, earlier KV release | 9896.0941 | 6124 / 6243 | 115 / 4 |
| [41166990387](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41166990387) | Round 11, fused QKV rejected; same served kernels | 9881.6659 | 6119 / 6239 | 117 / 3 |
| [41201412459](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41201412459) | Round 12, M192/N96 pair path adopted | **9958.4323** | 6160 / 6276 | 113 / 3 |

The 14.43 tok/s spread is an observed repeat difference, not a confidence interval. Recovered AIPerf goodput denominators are 123.76600222 and 123.84551467 s. The request-span `benchmark_duration` is shorter by about 3 ms and must not replace the official denominator. Fixed-duration 10k would require 65 and 74 additional good requests. All TTFT failures are in the initial arrival burst (request starts in the first 0.04–0.10 s); removing burst work can improve eligibility without increasing the steady serving rate.

For the three Prefill pipelines, measured periods are 114.9/114.6/118.1 ms and 114.6/115.0/118.3 ms, each completing two requests. Solve

`G_new = G_old * sum(2000 / (T_i - delta)) / sum(2000 / T_i) = 10000`.

With fixed good fraction, Decode behavior and drain, the required **uniform reduction is 1.203–1.372 ms per pipeline period**, or 50.1–57.2 µs per layer on a 24-layer critical stage. This is a conditional capacity model, not a forecast. A practical kernel target is at least 60 µs per actual two-request call across the whole FA3 path. Decode is already around 43.8 live requests per rank against a cap of 48; its ITL and acceptance must be rechecked in normal serving.

**Update after round 12:** its official denominator is 123.714251097 s; at fixed duration 10k requires 26 more good requests. Recovering PP0 period from each two-request log record's token-count/throughput gives medians **114.4097 / 113.9316 / 117.4825 ms**. The same conditional equation now requires **0.478993 ms per pipeline**, or **19.958 µs per layer**. This is the latest target, superseding the earlier 1.2–1.4 ms requirement. A full-path candidate should clear that target with margin before serving; changing a selection threshold after a losing measurement is not evidence of a speedup. Decode ITL was 15.569 ms mean / 18.331 ms p99 with three failures. The logger's page-padded 8064 tokens are used only to invert its throughput field, not as GEMM rows.

Round 12's artifact confirms all twelve GPUs passed numerical checks and 1200-call endurance and saved at least 5 µs in both timing orders on all seven request pairs before enabling M192/N96. This is an implemented, verified change from the prior chat, not an untried idea. A single serving run is not a confidence interval or a guarantee of 10k.

## What the profile actually says

Latest normal runs contain AIPerf request profiles, not a new Nsight kernel timeline. The newest available kernel trace is round 3 [40964714831](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40964714831), before split MegaMoE and the FA3 tile pipeline. Archive: `archive-host:/srv/qwen-artifacts/40964714831/1/`; original `nsys-prefill-2.nsys-rep` SHA256 `a9dcb4cb032687977f660a32046585e263e6ecf3afee062a4583a4c435478705`. The exported SQLite passes integrity checking. Device ordinal 2 maps to physical GPU 3.

For 18 matched steady windows, median period is 117.9844 ms and median interval with no active kernel is 0.4202 ms. One actual 24-layer PP2 window (batch 37, trace-relative 4792062779–4910084868 ns):

| Work | Time (ms) |
|---|---:|
| MegaMoE | 65.4627 |
| FA3 | 21.5867 |
| QKV / O GEMMs | 12.8809 / 11.1956 |
| QK postprocessing | 2.0162 |
| Norm / quantization | 1.4912 |
| Fused MoE input | 2.9247 |
| Other kernels | 0.1576 |
| Subtract kernel overlap | −0.1130 |
| Add no-kernel intervals | +0.4195 |
| Actual complete period | **118.0221** |

There are no NCCL kernels in these windows; PP copy-engine transport is already active. CPU and communication gaps share the 0.42 ms exposed interval and cannot each be claimed as separate savings. Occasional large gaps are not necessarily removable. Do not subtract later isolated screen improvements from this old trace to manufacture a current timeline.

## Computation and communication accounting

Actual shape is Qwen3-235B-A22B FP8, H=4096, I=1536, 128 experts, top-8, 94 layers, 64 Q heads / 4 KV heads / d=128. Prefill is three TP1/PP4/EP1 workers; Decode is TP4/DP4/EP4. A model request has 3994 rows; two packed requests give M=7988. The logged page capacity 8064 is not the GEMM row count.

At a **conditional** 1.40 GHz, 132 SM × 8192 dense FP8 FLOP/cycle = 1.5138816 PFLOP/s. Do not use Decode's higher clock, nor the advertised sparse tensor rate. NVIDIA's H200 specification provides the 4.8 TB/s HBM and 900 GB/s bidirectional NVLink reference ([datasheet](https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23?xs=443995)).

| Per layer, M=7988 | Useful FLOP | Tensor-only lower bound |
|---|---:|---:|
| MegaMoE: `6*M*8*4096*1536` | 2.41230 TF | 1.59345 ms |
| Causal QK + PV, two L=3994 sequences | 0.52285 TF | 0.34537 ms |
| QKV: `2*M*4096*9216` | 0.60307 TF | 0.39836 ms |
| O: `2*M*8192*4096` | 0.53607 TF | 0.35410 ms |

These are useful-work floors, not removable slack; padding, softmax, FP8 promotion, tensor/scalar dependencies, data movement and power all remain. Cold MoE expert weights alone are 2.4159 GB/layer (0.5033 ms at 4.8 TB/s). This overlaps arithmetic and is not added to 1.59345 ms.

One PP BF16 hidden/residual/auxiliary tensor is `7988*4096*2 = 65,437,696 B`. EAGLE auxiliary taps [2,47,91] mean the three stage boundaries forward 3/4/4 tensors, or 719,814,656 B per pair in total. At 450 GB/s in one direction, boundary payload floors are **0.4363 / 0.5817 / 0.5817 ms**. Their sum is a path latency reference, not an addition to steady throughput; overlap and independent stage links matter. Trace memcpy records have inconsistent host/device labels and missing endpoints, so their 11–13 µs durations are not a measured NVLink bandwidth.

FP8 KV payload is `94*3994*4*128*2 = 384,446,464 B/request`. At an assumed 50 GB/s, its serialization reference is 7.689 ms/request. NIC line rates and effective bandwidth were not established; 50 GB/s is explicitly a scenario. At 51.8 req/s, aggregate payload is about 19.9 GB/s, with roughly two-thirds crossing nodes in this topology. The full transfer can overlap computation; this calculation alone does not prove a network bottleneck.

The user-sent [modeling article](https://superhotdogcat.github.io/ja/blog/2026/10/25/modelingestimation/) was read in full. Its applicable principles are per-resource `max(F/P, B/W)`, the critical dependency path, and Little's law `inflight bytes / latency`. Its Qwen3-30B BF16/GH200 example is a different model and platform. The article's host gather example does not justify treating all apparent bandwidth headroom here as removable communication time.

## Previously rejected work, corrected against the underlying evidence

- **V byte-image is not a new untested idea.** An earlier survey remembered only its incorrect layout. The corrected map (`e3e7744` / run [40201493911](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40201493911)) passed the numeric gate, but improved median by only 1.9 µs with a 3.1 µs worst regression, below its 5 µs gate. See `2026-10-24-broad-ceiling-review.md`, section 8. PP4 shapes differ, but this cannot be sold as an untried transformation.
- PV-live register identity/PRMT preserved late operand liveness but lost 10–17 µs. Polynomial exp passed numerics and lost about 5.2%; half2 compilation retained two MUFU instructions. The new experiment retains ordinary FP32 softmax and FP8 rounding.
- MegaMoE independent warpgroups lost 22–36%; FC2 bulk scatter lost 1.3–4.9%; direct scatter and precombine also lost. The large no-rescale diagnostic is numerically invalid. The split FC1 tile change is already adopted. Unroll16 and FC2 address hoisting have already failed to produce meaningful serving improvement.
- Dense tail work skipping passed correctness but was 6.3% slower. QKV epilogue fusion passed 300-call gates on all 12 H200s but was 32–34% slower, so round 11 served the control.
- M192/N96 was already under test when this audit began. Its previous pair-only savings were 17.858–27.255 µs/call, corresponding to 0.429–0.654 ms/24-layer stage, not twice that amount. Round 12 passed its stricter pair gate and served at 9958.4323, as recorded above. The later round-13 screen failed and its normal serving repeated this fallback at 9948.4637. Neither is a new idea from this audit. Earlier single-sequence gate rejection did not establish that the actual two-request serving path was slower.

Chats inspected: `01a18d65-545c-7e02-995b-405c88f180ff`, `01a186bf-9f7a-7ae2-915e-dcf6ab5cbf55`, `01a17bd5-993c-74f2-8418-44158a05e6d8`, `01a18e75-f048-78d0-86a3-334929f20e49`, `01a16e62-c295-77f0-abb2-9f940111d56c`, `01a16cfb-4b6f-7c42-b182-429fbde9c83d`, `01a14eb3-35d6-7d51-a5a3-98f1f322bba5`; repository issues and discussion comments #20–24 were also read.

## New structural experiment: give PV its own warpgroup

The v1 source at commit `11b9493` separates the dependency chains, while keeping M128/N192, the native K/V representation and transpose, FP32 online softmax, causal attention and FP8 output:

1. One 128-thread producer warpgroup loads Q/K/V and performs the native V transpose.
2. Two QK/softmax warpgroups write FP8 probabilities and FP32 rescale factors into a two-slot shared-memory mailbox.
3. A fourth warpgroup performs PV, accumulates both 64-row waves and writes output directly.

In the existing two-math-warpgroup architecture, QK and PV share a warpgroup and the registers holding P; the compiled PV wait is placed within the exp sequence. There is already partial overlap. The new roles remove PV completion from the QK/softmax register lifetime; they do not merely move an identity operation or adjust a tile knob. This follows the asynchronous resource accounting used by [FlashAttention-3](https://tridao.me/publications/flash3/flash3.pdf), but is a new experiment for this pinned implementation, not a measured application of that paper's speedup.

Resource budget: 512 threads, 217088 shared bytes, and `128*(24 + 2*152 + 176)=64512` role registers, below 65536. The P mailbox costs 49152 bytes of storage. For the 3994+3994 pair, 43680 KV-block visits add **2,146,959,360 logical shared-memory bytes** (P write + read) per layer. This is not HBM traffic; its bank/port pressure, barriers and the changed CTA scheduling can erase the overlap gain. The first prototype uses one CTA per M tile with reverse-M order rather than the serving persistent tile scheduler. Performance must cover that whole path.

CPU checks exhaustively reconstruct the native V producer's LDSM.T/PRMT/STSM.N cross-lane byte map, verify all 24576 P entries against V's key indices, verify 4-byte shared stores, and verify all 16384 output coordinates after the native output permutation. All errors are zero. An initial host oracle used aliased uint2/float access and failed under optimization; replacing host swaps with memcpy fixed the oracle and O3 now passes. This was a test bug, not evidence that a wrong GPU permutation should be accepted.

The first kernel parameter passing form created a 2416-byte stack copy. `__grid_constant__` removed it. Current CPU native compilation has 24-byte stack, 20-byte spill stores / 56-byte spill loads over the complete kernel; steady QK/softmax and PV loops require separate SASS inspection. No claim of a completely spill-free kernel is made. Full patch-chain build and numerical GPU results are recorded separately below.

## Decision gate and reproduction

The one-candidate probe checks ordinary, zero and strongly peaked distributions; partial M/N boundaries; unequal sequence lengths; the actual 3994+3994 pair; and the 3986+4027 screen pair. Every candidate output and scale must match the control exactly, with sampled FP64 attention as an independent check. Only then may 300-launch endurance and alternating paired CUDA-graph timing run. A faster kernel is still not a normal serving result. Adoption requires a material full-path saving and later official quality, SLA, ITL, acceptance and goodput checks.

`pp4-10k-structural-20261101.json` is the numeric audit result. To recreate it from the recovered archives and SQLite:

```sh
python3 research/runtime-performance/analyze-pp4-10k.py --artifacts /Users/yuta/Dev/APAC/outputs/qwen-10k-audit-20261101 --output /tmp/pp4-10k-audit.json
```

The bounded probe builds both control and candidate from the exact SHA-pinned sources and ordered patches in `kernels/fa3-verify/build.py`. It records commands, compiler, patched-header hashes, source hashes, binary hashes, layout proof and SASS. ASPIRE connection attempts timed out from both the workstation and the intermediate server; the selected fallback is Firmus, one visible H200, node-exclusive allocation (8 GPUs), 5-minute scheduler limit / 165-second process limit. Jobs run through Actions only.

## Pre-allocation CPU result

The exact 14-patch chain compiled and linked both DSOs with CUDA 13.0.88 on the CPU. Host ABI loading reports control 166912 B / candidate 217088 B and one variant each; neither check initializes a GPU. Candidate SASS has no LDL/STL anywhere in its QK/softmax role (0x54b0–0xc510); the PV steady loop has no local-memory instructions either (its two scalar reloads are before/after the loop). No C7511/C7513/C7519 serialization warning occurred. The revised shared P stores use STS instructions. The whole-kernel spill counts include the producer and setup/epilogue; they are not described as zero.

CPU tests: 24 Firmus protocol tests pass, including failure propagation for this new probe; 18 existing ASPIRE protocol tests passed before selecting the Firmus fallback. Ruff, shell syntax and actionlint pass. The numerical audit reproduces byte-for-byte from the recovered inputs. GPU execution remains required for tensor instructions, mbarrier visibility, numerical/endurance and performance.

The broad repository test suite was also attempted on macOS. Its first failure is the existing GNU `tar -I zstd --transform` invocation against the macOS tar executable; this does not exercise the changed kernel. The focused protocol tests pass. The broad Linux Actions validation remains the gate before GPU dispatch.

## GPU result: first split-PV architecture is correct but rejected

[Run 41222602275](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41222602275), commit `11b9493`, Slurm 27932, completed successfully; 38 seconds allocated on one node. Artifacts were recovered before runtime cleanup. The preceding Linux validation run 41221925722 passed all three CPU jobs. All six numerical cases had exact output/scale equality, the independent FP64 errors matched, and both 300-call endurance checks passed.

| Pair | Control median | Split-PV median | Candidate/control | Median paired saving |
|---|---:|---:|---:|---:|
| 3994+3994 | 677.71 µs | 846.48 µs | 1.2490 | −163.44 µs |
| 3986+4027 | 680.55 µs | 851.01 µs | 1.2505 | −164.70 µs |

Every one of the 20 paired comparisons on each shape regressed (135–207 µs slower). The job's `PASS` means numerical/endurance gates and timing completed, **not** that the candidate is faster. This version is rejected and has never been enabled in serving. Clock/power endpoints reached 1725–1920 MHz / about 691–696 W; these are isolated-kernel timings, not the Prefill worker's 1.4 GHz serving profile. Raw result: `fa3-split-pv-gpu-41222602275.json`. ZIP digest: `3b71433435dacdcedf4c08c9098ff7c8a40d04a3401f62b5f32726adeaa865db`.

## Follow-up CPU investigation: independent 64-row PV consumers

The first version makes one PV warpgroup accumulate 128 rows and wait for both QK warpgroups' P. Restore two 64-row PV consumers, each with its own P mailbox, while retaining separate QK/softmax roles. This tests ownership and synchronization granularity; it is not an assumption that the measured loss was entirely caused by this choice.

A direct 640-thread conversion with N192 QK instructions fails CPU assembly: C7602 requires an initial register target of at least 122, while 640 threads permit 96. Two N96 QK atoms compile, but C7507 ignores register reallocation and C7511 serializes WGMMA; spill stores/loads are 2164/2240 bytes. Both are rejected before any GPU use.

There was also a resource-model defect: the initial proposal's `128*(24 + 2*152 + 2*88)=64512` fits the SM's 65536 registers but exceeds this CTA's initial `640*96=61440` pool. [PTX setmaxnreg](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg) exchanges registers within a CTA pool, and allocation can block until registers are released; unused registers elsewhere in the SM cannot simply be counted. The final CPU attempt uses QK136/PV88/producer24, or 60416 registers, and splits PV into N64 instruction atoms. The N64 fragment needed a view in the original N128 output layout for full-row amax and output permutation; an initial host layout bounds failure identified that defect and it was fixed. All P/V/output entries then passed, including maximum P-store bank multiplicity **1**. Nevertheless native assembly reports **C7512, 1236 bytes spill stores / 1136 bytes spill loads, 560-byte stack**. This variant is also rejected, not sent to a GPU.

Splitting QK's N atom adds approximately 715,653,120 logical shared operand bytes per actual pair; splitting PV's N atom adds approximately 1,073,479,680 more, on top of the original 2.147 GB P mailbox traffic. These are logical operand bytes, not measured HBM or physical shared transactions. The additional work and assembly failures do not support a speedup claim.

Evidence is in `fa3-split-pv-v2-cpu-rejected-20261101.json` and the archived `fa3-split-pv-v2-rejected.patch` (against `11b9493`; apply with `git apply --unidiff-zero`). At the v2 rejection commit, active probe code was restored to the GPU-tested four-warpgroup version. The later v3 below replaces that research probe only. The pre-allocation compiler gate now also rejects C7507 and C7512, which the original narrower warning check did not catch. The P-store bank check remains in the CPU oracle. Any further split-PV design must satisfy the actual CTA pool, avoid serialized/spilled inner loops and account for additional operand traffic before another GPU allocation; repeating this implementation is not the next experiment.


## V3: remove repeated V transpose and shorten the live register ranges

The old V-byte-image experiment was already correct and too small a speedup on its own. V3 combines that representation with two independent PV consumers: it removes the repeated transpose from the split-PV producer and removes the 49,152-byte transpose staging buffer. This is separate from the concurrent round-13 native-V/four-chain-sum serving proposal (`becabbf4`); that run 41225578193 stopped at build-input preparation, before GPU allocation, and supplies no performance evidence.

V3 uses five full warpgroups (640 threads): QK128/QK128/PV96/PV96/producer32 registers. Native initial allocation is 96 registers per thread, so both initial and active totals are 61,440. This also fits each SM subpartition: `(128+128+96+96+32)*32=15,360`. The producer release funds both QK increases. After releasing the last P slot, a QK group returns to 24 registers before its PV group grows to 176 for output. Explicit warpgroup barriers precede the second register adjustment.

A 544-thread version was rejected before GPU use: its native allocation was 96 rather than the required 120, and its partial producer warp could not supply the required registers. Reducing PV to 88 caused C7507/C7511 and over 2 KB of spills, so that alternative was also rejected. The viable 128/96 version initially spilled a row coordinate during softmax. Computing shared P addresses as 32-bit offsets and reloading the lane identity only after P conversion removes that live range. The final native build has **zero stack, zero spill stores/loads and no LDL/STL anywhere in either candidate kernel**, with no register/WGMMA serialization warnings.

The producer issues direct Q/K tensor-map loads and bulk copies of the native V byte image. A separate kernel creates each V tile once with the same LDSM.T/PRMT/STSM.N map; padded tokens are zero. Its launch and global reads/writes are included in candidate timing. Probe-owned scratch is allocated before CUDA graph capture and freed after synchronized use. This is a single-stream research ABI, not a serving workspace implementation.

CPU proof covers all P/V/output coordinates, both K-stage offsets, Q/K 128-byte TMA swizzling, the recomputed row coordinates and conflict-free P stores. All errors are zero. An exhaustive logical queue model explores **56,117 reachable states** over 1–22 KV blocks with no nonterminal deadlock. This proves the model's backpressure behavior, not GPU barrier visibility. Evidence: `fa3-split-pv-v3-cpu-20261101.json`.

Against the served M192/N96 control, the actual 3994+3994 pair has:

| Work | Served M192/N96 | V3 M128/N192 |
|---|---:|---:|
| CTA tiles | 2,664 | 4,000 |
| KV-block visits | 56,784 | 43,680 |
| Tiled QK + PV FLOPs | 535,881,056,256 | 549,621,596,160 |
| Repeated native V transpose, logical shared bytes | 1,395,523,584 | 0 |
| New P mailbox write + read, logical shared bytes | 0 | 2,146,959,360 |
| Requested Q/K/V global-to-shared payload bytes | 1,460,994,048 | 2,212,495,360 |

Including the different Q/K/V operand reuse, V3 adds **894,566,400 logical shared operand bytes** versus current serving, before its small once-per-input V packing. It also performs **2.56% more tiled FLOPs**. Against the older N192 control, removing V transpose cancels the P mailbox traffic, but applying that cancellation to current M192 would be incorrect. These are logical byte counts, not physical transactions or measured bandwidth. The only proposed benefit is better overlap of independent QK/softmax and PV; a speedup is not assumed.

The operand subtotal excludes the producer's global-to-shared writes. Those requested Q/K/V payloads are `tiles*M*128 + 2*visits*N*128` bytes, adding **751,501,312 bytes** versus M192/N96. Once-per-input packing adds 16,515,072 logical shared bytes (four passes over its 4,128,768-byte padded image), reads 4,089,856 valid V bytes globally and writes that padded image. The counted shared-byte increase including payload writes and packing is therefore **1,662,582,784 bytes**; this excludes mailbox alpha/scale metadata, barriers and output epilogue traffic. Global-to-shared requests can hit L2, and masked tiles can be padded, so these payload counts must not be reported as physical HBM traffic.

The bounded V3 GPU protocol uses N192 as an exact output/scale oracle and sampled FP64 attention independently. Seven inputs include a changed-input replay, partial tile boundaries, unequal lengths, zero and peaked scores. Both realistic pairs then run 1,200 repeated calls and 20 alternating CUDA-graph timing pairs. The **speed control is served M192/N96**, and the candidate includes both packing and attention. The next-stage threshold is at least 30 µs saved in both timing orders on both pairs, with at least 18/20 positive pairs; it is a screening margin above the conditional 19.958 µs/layer estimate, not a goodput prediction. Normal serving remains unchanged until actual numerical and performance gates pass.

## Fixed-request SLA rescue is a different counterfactual

The round-12 raw records have 113 TTFT-only failures. Holding the request set and the official 123.714251097 s denominator fixed, reaching 10k requires 26 additional good requests. Their sorted SLA excess shows that a uniform **20 ms TTFT reduction rescues only 2 requests**; rescuing 26 requires **463.439531 ms**. An 8 ms reduction rescues none (the smallest gap is 8.238854 ms). These are derived directly from the 3,000 ms threshold and each request's measured TTFT, with ITL failures excluded.

This does not rule out communication as a throughput limiter. It only shows that saving a few milliseconds at request completion, while keeping admitted work unchanged, cannot close the score gap. Increasing sustained Prefill capacity is the separate model that requires a 0.478993 ms shorter pipeline period, or approximately 19.958 µs per layer across 24 layers. Do not confuse the 463 ms latency-only rescue with the 0.479 ms period reduction.


## V3 GPU result: correct, stable, and rejected

[Run 41230406211](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41230406211), pinned to `757eff9caa58a7f46326ce1470076fb8e22ecc46`, completed with Slurm 27946 exit `0:0` after 76 allocated seconds. Seven numerical cases, including the final changed-input random replay, had exact output and scale equality to the N192 oracle and identical sampled FP64 errors. Both real pairs passed 1,200 calls and post-timing graph replay equality. The actual compiled candidate also has zero stack/spills and initial 96 registers, matching the CPU proof.

| Real sequence lengths | Served M192/N96 median | V3, including V packing, median | Median paired saving | Positive pairs |
|---|---:|---:|---:|---:|
| 3994 + 3994 | 663.922 µs | 741.470 µs | **−79.896 µs** | 0 / 20 |
| 3986 + 4027 | 675.816 µs | 749.567 µs | **−74.389 µs** | 0 / 20 |

Forward/reverse paired savings were −78.081/−81.172 µs and −76.539/−70.415 µs respectively. The worst-to-best pair ranges were −120.514 to −59.275 µs and −112.744 to −61.866 µs. The performance gate is false; the workflow success means numerical/endurance completion and artifact recovery, not a speedup. No normal serving run or adoption follows this result. The added 1.663 GB of counted shared traffic, additional tiled FLOPs and CTA work remain costs of this design; timing alone does not assign the regression to a specific hardware stall.

Canonical evidence: `fa3-split-pv-gpu-41230406211.json`. Artifact 13055932903 ZIP SHA256 `bba8c47932ae701013999b13ee071549700c6e81af8ba2ea21e386ae2154935e` was recovered and verified. Actions confirmed terminal Slurm state and removed its runtime after upload. The preceding queue wait was `Priority` with both nodes allocated; [read-only diagnosis 41235009917](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41235009917) allocated no GPU, and its temporary workflow change was removed.


## Round-13 native V/sum result and lane follow-up

Run [41240194309](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41240194309) rescued the existing round-13 proposal from the failed source-staging path. The unchanged production source was first fully compiled and CPU-loaded with the actual CUDA 13.0.88 toolchain, then freshly built and deployed. All 32 FA3 translation units linked, the actual-image PP4 CPU gate passed, and Slurm 27959 completed normal serving in 740 seconds with exit 0:0. This proves the execution route, not a performance improvement.

The original native-V/four-chain-sum path was **43.257–54.423 us slower** on the first 3994+3994 pair across all twelve H200s and both timing orders (median loss 48.821 us). First-shape native bytes, paged KV/QK bytes and changed-input FP32 checks passed. Each rank stopped at the speed rejection, so no later-shape, single fallback or endurance success is claimed. Independent Slurm steps exchanged the two head-node workers; comparing their labels instead of the complete physical UUID set caused the misleading aggregate reason `wrong GPU`. The twelve UUIDs and allocation were actually identical. This bookkeeping defect did not cause the measured slowdown.

Normal serving therefore retained round 12 and scored **9948.463699 good output tok/s**, eligible with zero request errors: 6156/6273 good, 115 TTFT-only failures and 2 ITL-only failures. This score is not round-13 performance. Recovered period medians are 113.9231/114.4037/117.4183 ms. Under the same conditional capacity assumptions as above, the uniform period reduction needed for 10k is 0.593736 ms, or 24.739 us per 24-layer stage layer. The best prior round-12 observation implied 19.958 us; their 9.969 tok/s difference is not a confidence interval. At a fixed request set, 32 additional requests are needed and the corresponding TTFT-only reduction is 573.379396 ms; a 20 ms cut rescues only four.

The new implementation changes only native V thread ownership to 32 tokens by 16 adjacent dimensions per warp. Q/K, FA3, quantization and dispatch are unchanged. It keeps sixteen byte-store instructions but turns each warp's native store from eight partial sectors into one full sector, with one aligned 128-bit paged-V store. Including increased input and paged-cache sector requests, counted V I/O requests fall from 45,299,712 to 28,667,904 bytes for a 3994 pair (36.715%). These are issued-sector counts, not physical HBM bytes or a predicted speedup. No new shared memory, barrier or shuffle is added.

CUDA13.0 full production JIT loading, 48 registers/spill0, independent pinned CuTe byte mapping, all65536 BF16 patterns on every seven-pair/index-type case, and 141 focused CPU tests passed with no skips. Twelve other kernels have identical encoded SASS; two Decode instances change only their device-assert line number. CPU-only Actions [41245881739](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41245881739) passed all three jobs. Empty commit 744011e did not start a workflow because of path filtering. Execution commit 6318cf7 changes only the module description; its executable AST and kernel bytes are identical to CPU-validated 38c2d4a. Run [41246674412](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41246674412) completed successfully, but rejected the lane candidate as detailed below. Its all12GPU full-path gate was unchanged; isolated producer and FA3+O timing is diagnostic only and is not added together or used to relax the gate.

Round-13 evidence is `fa3-prefill-vsum-gpu-41240194309.json` in branch `perf/pp4-vsum-cuda130-20261101` at bace6f4; lane source/CPU proof is in `perf/pp4-vsum-native-lanes-20261101`. The original round-13 serving artifact digest is `7eef34d557e1a3c9740494a0b213ffa8008a35f441d35e10d10e2ecdf750056b`. Its temporary workflow/script changes were removed. After every dependent job became terminal, the private temporary source release/asset was deleted (verified404; no matching tag existed). Both lane-branch workflows were restored exactly to main4a1e86e and the temporary staging script was removed. The owned CPU scratch was removed after evidence recovery; shared toolchains and authoritative archives were retained.

## Latest lane result and the actual 10k budget

Run 41246674412 / commit `6318cf750d343fe0ebd79a0480278703776b4b69` passed the fresh CUDA13.0.88 build, all32 translation units and full Torch ABI load, actual-image PP4 CPU checks, normal serving and artifact archival. Slurm27962 finished `COMPLETED 0:0` in733 allocated seconds on two nodes. The runtime was removed by the workflow. The image digest is `f21e2a33e33142fa56a731089437176a9a9c587811b184b46f1ff7728dbfbe9e`.

All12 H200s rejected the candidate on the first3994+3994 pair:

| Timing scope | Median slowdown across GPU pooled results | GPU range | Faster paired samples |
|---|---:|---:|---:|
| QK post + FA3 + O, adoption path | 37.369 µs | 35.447–40.948 µs | 0/240 |
| Isolated producer | 7.631 µs | 6.887–8.268 µs | 0/240 |
| Isolated FA3 + O | 29.328 µs | 27.403–40.013 µs | 4/240 |

These component measurements are separate experiments and are **not additive**. FA3+O is not an isolated FA3 timing. Every GPU passed native bytes/padding, QK/paged KV equality, reference and two changed-input replays on the first shape. Speed rejection stopped the remaining six shapes, singles and endurance. The generic aggregate reason is `failed or late GPU`, but all actual rank failures say `below 5 us margin`, with roughly29-second rank runtimes; this was not a timeout. The physical-UUID aggregation repair worked.

The official score **9951.183629** is therefore **round12 fallback**, eligible with zero errors and zero OSL mismatches. It has6161 good requests of6275,113 TTFT-only failures and one ITL-only failure. All TTFT failures started in the initial0.05194–0.09474 seconds. The official denominator is123.824466107 seconds; the123.821603840-second request span is a different value and must not replace it.

The static V-sector comparison must use the right control: original native45,299,712 → lane native28,667,904 bytes is a36.715% reduction, but the deployed contiguous-V control requests only16,359,424 bytes. The lane writer still requests **75.238% more** than the deployed control. These are issued-sector counts, not physical HBM/L2 traffic. Neither this count nor the original-r13/lane cross-run timing difference establishes a causal latency gain.

The measured Prefill PP0 periods are114.319430 /113.933467 /117.525185 ms (1485/1495/1449 samples). Reusing the conditional saturated-capacity model, all three pipelines must shorten by **0.562437 ms**, or **23.434892 µs per layer over24 layers**, to reach10k. From the rejected candidate itself this would require a60.804223 µs/layer recovery. This is why more small edits to this native-V/sum design have low priority.

| Uniform period reduction | Reduction per24-layer-stage layer | Conditional goodput |
|---|---:|---:|
| 0.184 ms | 7.667 µs | 9967.10 |
| 0.480 ms | 20.000 µs | 9992.81 |
| 0.600 ms | 25.000 µs | 10003.28 |
| 0.800 ms | 33.333 µs | 10020.76 |
| 1.000 ms | 41.667 µs | 10038.31 |

These are scenarios, not predictions: they hold good fraction, drain and Decode behavior fixed, assume saturated Prefill and equal millisecond improvements across pipelines. A useful engineering target is0.8–1.0 ms rather than barely0.6 ms, given the approximately10 tok/s spread already observed in round12 fallback runs; that observed spread is not a confidence interval.

For the separate fixed-request latency-only counterfactual,31 more requests must become good. A uniform8/20/50/100/200/500 ms TTFT cut rescues1/2/2/4/10/26 requests; reaching31 requires571.878119 ms. A few milliseconds saved only at request completion will not close the gap. Communication that limits sustained throughput remains a different question. The prior compute/communication byte models above are still conditional resource floors, not quantities to add to the observed pipeline period.

No new Nsight trace exists in this run (`profile:false`). Run40964714831's MoE65.46 ms and FA321.59 ms breakdown predates the current split/pipe changes and cannot identify the latest subphase bottleneck. What the latest run establishes directly is the full-path regression and its producer/FA3+O component regressions.

Canonical evidence is `fa3-native-v-lane-gpu-41246674412.json` on the lane branch. The existing archival workflow had already deleted the GitHub ZIP, so155 archived serving files were recovered read-only from the home-server archive and checked against its SHA256SUMS. The repacked tar digest is `281e1ef0523e90319277dee305a032045af3d6afa3fb9755b5684a9a056d4dd3`; it is not the original ZIP digest `c20541c0bd6680b09b0775625f65005a329cde39f8dfeb6535724d00c362032a`.

## CPU follow-up: repair the previously rejected shared-reciprocal path

The round7 shared-RCP experiment (run41040348302, source911a2e8) passed numerical checks and reduced the sum of24 GPU/routing timings from61,295 to61,111 µs, but failed all12 GPUs' rotating four-routing600-call determinism checks. That is a mean7.667 µs/call gain before the repair, not a measured gain of the corrected code.

Source reconstruction found a specific instability: the overflow fallback combines two row bounds and uses a warp-wide vote to overwrite every row with a different FP32 formula. Atomic expert dispatch changes the neighboring token rows. A valid row's arithmetic can therefore change when a neighbor or invalid padding triggers fallback, even with no memory race. The two formulas agree mathematically but do not round identically.

The research repair gives each valid row its own bound and fallback decision; the warp vote only skips unused work. A1000-case host model, with FP32 rounding and explicit FMA, found4 FP8-byte changes under row permutation and one under padding changes in the old code, versus zero in the repair. Extracting and executing the actual source bodies with32 synchronized host threads agrees bit-for-bit with the independent model in48 cases. The host uses exp2f/division, **not a Hopper MUFU emulator**, so this is a source-level defect reproduction, not confirmation of the GPU endurance failure's sole cause.

CUDA13.0.88/sm90a compiled control-off, old-off, fixed-off, old-on and fixed-on for the relevant production template with zero stack/spills/local instructions and no C75xx warnings. All three off controls have identical encoded SASS. The actual compiler flags and700 dependency files are recorded. No production source/configuration/gate was enabled, and no repaired GPU execution occurred.

If the historical7.667 µs gain survived the repair unchanged,24 layers would save0.184 ms and the latest conditional score would be9967.10. That leaves about0.378 ms, or15.768 µs/layer, still required just to reach10k. This is a concrete correctness repair worth retaining, but it is not the complete performance solution. Any further MoE proposal must identify additional work removed from the active FC1/FC2 path; the old profile alone cannot supply that measurement.

The alternative native-V-only prototype (variant41184, remove Sum4) also completed all32 translation units, full link and CPU Torch ABI/resource queries with CUDA uninitialized. Its candidate has zero stack/spills and exactly the earlier prototype SASS. It changes neither tiled Tensor work nor EX2 work. At a hypothetical1.4 GHz,132-SM peak, the modeled pair floors are0.354 ms for tiled Tensor work and0.368 ms for score plus rescale EX2 work; these overlap and must not be summed. Fewer PRMT instructions do not establish a speedup, so this remains lower priority with no GPU request.

CPU proof and the minimal repair patch are recorded as `megamoe-shared-rcp-row-local-cpu-20261101.json` and `megamoe-shared-rcp-row-local-fallback-20261101.patch`; native-only ABI proof and the resource model are `native-v-only-cpu-20261101.json` and `fa3-sfu-resource-model-20261101.json`, all on the lane branch. The combined raw CPU archive digest is `f76298fef0a9f800f7ccdad679b84fe8cf6a9703206508f4948a99b7a89c60b7`. **No additional GPU submission or retry is authorized under the latest user instruction.**
