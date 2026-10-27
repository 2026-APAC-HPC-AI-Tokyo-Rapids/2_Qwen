# Prefill memory changes, 2026-10-27

## Baseline and scope

Start: `1adb18a4cc0b96a0622ee0ff8117417a077062df`, pinned image
`40197369083`, SHA256 `1f247668933c7252d3dfbe8f6c23897d354f095d56842788d4415a85d7dc5e55`.
Both controls include **all configured serving overlays**. An older image-only
SGLang control would not represent the current deployment.

The existing 40313004834 profile/score gives 9129.28 good tokens/s, 251.4 ms
Prefill cycle, MegaMoE about 142.5 ms, QK norm/RoPE 6.24 ms, add RMSNorm
3.19 ms, and norm/quant 2.86 ms. MegaMoE is fused; its current FC1/FC2
times cannot be extracted from that trace. Historical subphase measurements
are evidence of mechanisms, not an exact decomposition of this run.

9500 requires 4.06% more good throughput. Holding handoff, good fraction and
Decode service constant, that corresponds to about 9.81 ms less per Prefill
cycle. These two candidates do not establish that gain before measurement.

The supplied [roofline article](https://superhotdogcat.github.io/ja/blog/2026/10/25/modelingestimation/)
motivates counting actual bytes and examining in-flight work before adding
parallelism. Its GH200 C2C gather result is not evidence for H200 NVLink.
For 4032 rows, the QKV prologue reads 74,317,824 B of packed BF16 QKV and
writes 41,287,680 B of FP8 Q, contiguous KV and paged KV (minus masked slots).
That is about 115.6 MB plus metadata/weights/trig traffic, not just the 33 MB
Q output. At 66 us this lower-bound byte count is 1.75 TB/s, below the 4.8
TB/s nominal HBM bandwidth. The 24.1 us byte-only floor ignores arithmetic,
dependencies and cache traffic; it is not a predicted latency.

## Implemented candidates

1. **QKV head4.** Re-map each CTA from 32 tokens × 1 head to 8 tokens × 4
   adjacent heads. Preserve 256 threads, 32 groups of 8 lanes, reduction
   order, both BF16 rounding points and hardware FP8 conversion. This differs
   from rejected 40225743220's BM4 × 32 heads and 97 registers.
   Existing CPU compile for the exact generated header gives 57 registers
   with PDL, zero stack/spills, and identical arithmetic opcode counts.
2. **Prefill combine direct store.** Keep input TMA, ascending top-k FP32
   FMA order and BF16 cast. Only `kPrefill2WG` writes each output vector
   directly via coalesced `uint4` stores. Remove shared output staging and
   its TMA store/wait. Output payload and remote expert placement do not
   change. CUDA 13.0.88 SM90a compile of current source: Prefill128/140 both
   retain 168 registers, zero stack/spills, 3 barriers. Four other geometries
   have identical SASS encoding. Prefill has four fewer STS.128, four more
   STG.E.128 and one fewer DEPBAR; total WARPSYNC count is unchanged.

Current MegaMoE header SHA256:
`220f409fd723d931cdad4e70b8e149646ececb5d1ee8554048683043ae413aa2`.
Candidate header SHA256:
`76a486f256d7de1b9516b20c4ebbe4ab9705edb140b6be022b0fa7530d7801c1`.
CPU compile sources/logs/cubins/SASS were recovered under the task's
`outputs/prefill-memory-20261027/moe-cpu/`; remote scratch was removed.

No quartet combine experiment: the earlier paired combine already reached
about 301 MB / 86 us and did not shorten its measured interval. No repeat of
independent Prefill WG, bulk scatter, FP32 precombine, interleaved FC2, old
wide QKV, already-adopted FP8 conversion or the already-adopted address changes.
The unrelated Decode M1 campaign is retained under
`patches/deep-gemm-archive/decode-m1/` and excluded from this submission.

## One allocation and admission gates

One Firmus node, 4 GPUs, existing 17-minute cap and pinned container. Five
steps: current MegaMoE control, candidate numerical gate, candidate timing,
one-GPU QKV gate/timing, final MegaMoE control. No new image build or profiler.
Timing uses only Prefill balanced/local 4096 and the existing real43/62/7
replica4 generators. These are seeded synthetic routing distributions, not
actual routing replay; no Decode timing sweep or replica-count sweep.

QKV covers zero/edge/3986/4032/4096 rows, int32/int64 metadata, finite and
nonfinite BF16 inputs, five exact output buffers, changed-input CUDA graphs,
four rotating working sets, 12 alternating timing pairs and a final control.
It needs >3% speedup at all three large lengths and <5% control drift.

MegaMoE needs the numerical gate, exact hashes and deterministic replay,
<2% control drift, all three replica4 cases >0.5% faster than **both** controls,
median >1%, and no tested case >1% slower before a full serving run.
Source/patch hashes, numerical results, raw timings and final control are
retained in the Actions artifacts. A successful diagnostic is not a serving
throughput claim.

## Run 40459750245: no serving promotion

[Actions 40459750245](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40459750245),
commit `d80d858180f4617f93dfb90969dbc0e2ce7f9423`, completed successfully.
Slurm 26605 on g1: COMPLETED, exit 0, elapsed 3:25. All five steps passed
(34/62/34/38/35 seconds). Raw logs/results/source were archived at
`archive-host:/srv/qwen-artifacts/40459750245/1/`; 52 downloaded files
matched the archive SHA256 manifest. Runtime cleanup and GitHub artifact
deletion after archival succeeded.

| MegaMoE case | Initial control ms | Direct store ms | Final control ms |
| --- | ---: | ---: | ---: |
| balanced | 1.54237 | 1.54493 | 1.54675 |
| local | 1.38432 | 1.37408 | 1.37357 |
| real43 replica4 | 1.47123 | 1.47229 | 1.47293 |
| real62 replica4 | 1.43258 | 1.43190 | 1.43082 |
| real7 replica4 | 1.48246 | 1.47600 | 1.48144 |

All five output hashes match, including deterministic replay, and the
logical-128/physical-140 numerical gate passed. The local case's apparent
0.7% improvement is also present in the final control. Three replicated
cases do not satisfy the predeclared gate. **Do not promote or remeasure
this candidate.**

| QKV rows | Control us | Head4 us | Change |
| --- | ---: | ---: | ---: |
| 3986 | 52.49 | 51.75 | -1.41% |
| 4032 | 52.94 | 52.37 | -1.08% |
| 4096 | 53.73 | 53.06 | -1.25% |

All 24 shape/dtype cases passed finite/nonfinite five-buffer bit equality
and the applicable changed-input graph checks. The 0.57–0.74 us reduction
is below the predeclared 3% admission threshold. Even extrapolated over 94
layers, it is only about 0.05–0.07 ms, far below the 9.81 ms target. These
standalone timings must not be compared directly with the 66 us serving
profile. **No full serving run is justified by this result.**

## Next bounded candidate: exact BF16 paired rounding

The head4 result does not support another layout sweep. Current SASS still
has 32 scalar `F2F.BF16.F32` instructions per Prefill QKV lane. Pairing RN-even
conversion retains both existing BF16 rounding points and every scalar
multiply/FMA expression; it does not lower model precision.

The first CUDA-intrinsic version halved BF16 conversion instructions but
introduced 20 additional PRMTs: non-NOP instructions 763→768. It was
rejected before GPU submission. An explicit packed-u32 result with bit-shift
unpacking removes that overhead. CUDA 13's own `cuda_bf16.hpp` confirms the
PTX operand order. Final SM90a compile: slots 776→768, non-NOP 763→755 for
PDL/int64, registers 57→56, zero stack/spills. All four PDL/id-type variants
have fewer non-NOP instructions; FP32 arithmetic, reduction and FP8 repair
opcode counts are unchanged. The original 32-token/one-head CTA is retained.

One Firmus kernel-probe invocation runs only this candidate and its control,
using the same exact-output/changed-input/alternating-timing gate and the same
3% admission threshold. No MegaMoE rerun, no NCU permission preflight and no
new image. The existing partition route reserves one node for at most five
minutes; the probe exposes one GPU and has a 165-second trial timeout.
The source generator, control header, output patch, hashes and raw samples
are archived with the probe results. The completed result is recorded below.

## Run 40463471079: paired rounding rejected

[Actions 40463471079](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40463471079),
commit `d78916edbc29208f2d3c86002357a412ba673296`, completed the probe.
Slurm 26609: COMPLETED, exit 0, elapsed 39 seconds. Provenance identifies
the pinned image `40197369083`, CUDA 13.0, all configured SGLang overlays,
eight allocated GPUs and one exposed probe GPU; this is not a serving run.

All 24 shape/dtype cases passed finite/nonfinite five-buffer bit equality;
the eight cases with changed-input CUDA graph checks also passed. With four
rotating input sets, 12 alternating timing pairs and a final control, all
three large shapes were slower:

| QKV rows | Control us | Paired us | Paired/control | Change |
| --- | ---: | ---: | ---: | ---: |
| 3986 | 52.656 | 55.272 | 1.04968 | +4.97% |
| 4032 | 52.968 | 55.720 | 1.05196 | +5.20% |
| 4096 | 53.736 | 56.464 | 1.05077 | +5.08% |

The saved result reports `candidate_eligible: false`. Fewer compiled
instructions did not translate into a faster kernel. **Reject paired
rounding; do not promote it to serving or repeat this candidate.** No full
serving run was submitted for it.

Evidence: `outputs/prefill-memory-20261027/run-40463471079/firmus-kernel-probe-40463471079-1/`:
`logs/logs/slurm-accounting.txt`, `logs/logs/provenance.json`, and
`results/results/prefill-memory/results.json` retain terminal status,
commit/image/source hashes, all numerical cases and raw timing samples.

## FA3: reject half2 exponent before GPU, test bounded FMA offload

The user permits evaluating changed intermediate rounding, conditional on
numerical error and output quality. The original F1 estimate assumed one
`ex2.approx.f16x2` emitted one MUFU instruction. Native CUDA 13.0.88 / SM90a
compilation of the actual fixed N224 specialization instead emits two
`MUFU.EX2.F16` instructions, one per half. The source has 336 paired-path
scalar exponents: they remain 336 MUFU instructions, plus 168 F2FP packs.
Spill stores/loads increase 72/120 to 80/132 bytes. **Do not submit this
half2 implementation.** Its predicted 3.5–7 ms reduction is invalid.

The new `fa3-prefill-exp.patch` moves one in four exponents in **unmasked**
FP8 softmax iterations onto a cubic FP32 polynomial. Masked iterations,
row maxima, FP32 row sums, rescale arithmetic, E4M3 conversion, FP8 output,
and the analytic fixed Prefill scheduler remain unchanged. Range reduction
uses round-to-nearest FP32; coefficients interpolate `2**r` at the four
Chebyshev nodes on `[-0.5, 0.5]`. Host FP32/FMA evaluation at 327,685 points
has maximum relative error `9.973655784789592e-5`. This is an expression
check, not a GPU or model-quality result.

The compiled unmasked loop changes 114→86 MUFU.EX2, FFMA 116→200, FADD
112→196, FMUL 66→94. Registers remain 168; stack/spill stores/loads remain
40/72/120 bytes, with no local-memory instructions in that hot loop. The
masked loop is unchanged. Static instruction counts justify one bounded
comparison; they do not establish a speedup.

`prepare-fa3-exp.sh` reconstructs source from the pinned image's exact
FA3/CUTLASS pins and seven patches, checks the image manifest and binary,
and builds only control/poly on the login CPU. The one-GPU trial must
first reproduce the pinned native control's FP8 bytes and FP32 scales.
The candidate then checks 3584/3986/4027/4096 rows, normal/flat/sharp/zero-V
inputs, independent FP32 causal attention on boundary and stratified query
rows, changed-input CUDA graphs, output/tail/LSE canaries, and propagation
through the actual FP8 O projection.

Predeclared numeric admission: candidate/control relative L2 ≤1.5%, scaled
max error ≤8%; independent-reference L2 ≤5% and scaled max ≤25% for both.
Candidate reference error may grow at most `control*1.05 + 0.0005` in L2
and `control*1.10 + 0.005` in scaled max. These are kernel admission gates;
the official model-quality condition still requires separate evaluation.
There are no old-variant sweeps or profiler runs.

Performance admission requires all three large shapes to save ≥3% and
≥5 us, with control drift ≤5% and paired-ratio p90 <1, using four rotating
input sets and 24 alternating pairs. A failed gate must not be relaxed
after looking at measurements. One exclusive node is reserved, eight GPUs
allocated, one visible; the trial timeout is 165 seconds within five minutes.

The first dispatch, [40471657626](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40471657626),
commit `952c7ef1d01a8fbd7ecd89870b31cec72dc4aac4`, stopped during CPU
provision because the cluster's `unsquashfs` has no `-cat` option. No Slurm
reservation, numerical test or timing ran. The image SHA check passed;
source-manifest checks were not reached. Logs were recovered and runtime
cleanup succeeded. Replace `-cat` with the existing selective directory
extraction protocol, keeping the candidate and all gates unchanged.

## Run 40472152140: FA3 polynomial offload rejected

[Actions 40472152140](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40472152140),
commit `9deda82`, completed successfully. Slurm 26628: COMPLETED, exit 0,
elapsed 34 seconds. The source provenance reports `image_source_sha_verified:
true`; the saved image FA3 manifest SHA256 matches `manifest_sha256` in
provenance. The reconstructed control reproduced the pinned native FP8
bytes and scales in all 16 numerical cases. This run tested the cubic FP32
offload, **not** the half2 exponent implementation rejected on CPU above.

All 16 numerical cases passed, including changed-input graphs and canaries.
The largest candidate/control relative L2 was 0.0026858121 (4027 normal),
and the largest scaled max error was 0.0558035858 (4096 sharp). Independent
FP32 attention references covered 43 query rows per case; passing these
kernel gates does not establish official model output quality.

All three O-projection gates also passed. Each pair below is relative L2 /
scaled max error; reference columns use the independent FP32 reference.

| Rows | Candidate/control delta | Control/reference | Candidate/reference |
| --- | --- | --- | --- |
| 3986 | 0.00295644 / 0.00155492 | 0.02866838 / 0.02691524 | 0.02866817 / 0.02691524 |
| 4027 | 0.00297186 / 0.00135281 | 0.02761902 / 0.02085072 | 0.02761876 / 0.02085072 |
| 4096 | 0.00291543 / 0.00123355 | 0.02898429 / 0.02563107 | 0.02898381 / 0.02563107 |

| FA3 rows | Control us | Polynomial us | Paired-ratio median | Change |
| --- | ---: | ---: | ---: | ---: |
| 3986 | 358.396 | 378.090 | 1.052855 | +5.29% |
| 4027 | 369.220 | 388.320 | 1.051378 | +5.14% |
| 4096 | 378.568 | 398.476 | 1.052797 | +5.28% |

Control drift was 2.25%, 2.90% and 4.91%, within the predeclared 5% limit;
all three shapes were slower in the paired comparison. The saved result
reports `numerical_passed: true` and `candidate_eligible: false`.
**Reject this polynomial candidate; no remeasurement or serving promotion.**
No model-quality evaluation was performed because the performance gate
failed. Reduced MUFU count alone did not produce a faster kernel.

Evidence: `outputs/prefill-memory-20261027/run-40472152140/firmus-kernel-probe-40472152140-1/`,
especially `results/results/fa3-exp/results.json`, `logs/logs/provenance.json`,
`logs/logs/slurm-accounting.txt` and `results/results/source/image-fa3-manifest.json`.

The comparison above used the companion's N224 control, not the selection
active in the 9129 serving run. The recovered run 40313004834 selection
records show **Prefill N192**, which beat N224 on all twelve GPUs and all
three tested lengths (minimum saving 6.195 us). Do not interpret the N224
probe as a measurement of that serving configuration. Its rejected candidate
is not being retested or promoted.

## Next serving candidate: FP8 draft, prior target selections fixed

The separate current Firmus opportunity is the draft extend's BF16 GEMMs,
about 3.9 ms per Prefill iteration. Online FP8 is already implemented by
the pinned SGLang source; select it through
`--speculative-draft-model-quantization fp8` on all four workers. The
`ModelConfig` branch uses this field only for `is_draft_model`; target
quantization, checkpoint, KV dtype and verification remain unchanged.
The original BF16 draft checkpoint SHA/size check remains mandatory.
Server-info must confirm the requested draft precision before the client
starts, and the acceptance report records configured and observed precision.

The old ASPIRE test 36538766257 was neutral (2468.02 vs 2473.60) and lacked
acceptance evidence. It used another topology and does not resolve the
current Firmus case. No current Firmus draft-FP8 measurement was found in
the inspected evidence. The new run keeps the official 60/120/30 protocol,
240 concurrency, P3×TP4/DP4/EP4+D1×TP4/DP4/EP4, cap176 and pinned image.
It does not run a separate stock-TP4 acceptance-only server first.

Reuse run 40313004834's actual selections:

```
SGLANG_PREFILL_FA3_VERIFY=n192
SGLANG_DECODE_FA3_VERIFY=p128split2-s3-fp8
DG_SM90_1D2D_DECODE_CONFIG=QKV:128,144,5,1,0
SGLANG_DECODE_DENSE_GEMM_QKV=
SGLANG_DECODE_DENSE_GEMM_O=
```

`firmus-kernel-selection-40313004834.json` binds the original source commit,
image SHA, all 21 ordered overlay hashes, expert-map hash, normalized
serving config and evidence hashes. Only draft precision, the reuse-record
path and paths to identical images may differ. CPU validation must pass
before the runtime receives selection env files. Then the unchanged target
kernel sweeps and numerical gate are omitted, avoiding repeated tests of
known candidates and changes to the comparison's target kernels.

Reference scores are 9118.01 (40310899466) and 9129.28 (40313004834), with
BF16-draft acceptance about 2.211 including the bonus token. The latter
enabled request-time-stat logging; this candidate keeps the current normal
launcher without that logging or a profiler. A single small score difference
is not sufficient for adoption: require error-free official results, inspect
acceptance and telemetry, and confirm a promising improvement before merging.
This opportunity alone is not expected to supply the roughly 10 ms needed
for 9500 at otherwise unchanged conditions.
