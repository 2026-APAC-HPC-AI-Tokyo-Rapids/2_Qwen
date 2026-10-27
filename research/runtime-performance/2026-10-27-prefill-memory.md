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
