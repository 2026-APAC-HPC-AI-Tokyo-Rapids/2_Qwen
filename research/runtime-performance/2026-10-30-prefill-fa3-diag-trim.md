# Prefill FA3 round 8: where the 404 µs go, and trimming the diagonal block (2026-10-30)

Branch `perf/prefill-fa3-r8-20261030`, based on 249e24b (the PP4 round-8 pins).
Pinned row: `n192-direct-o-varlen-pipe` (PrefillVariant 224), about 404 µs per 4000-token
sequence on Firmus (median of the round-7 screen in 41040348302).
New row: `n192-direct-o-varlen-pipe-diag` (224 | 4096, `kernels/fa3-verify/prefill-diag.patch`).

All counts below are CPU-side: CUDA 13.4, `-gencode arch=compute_90a,code=sm_90a`, the image's
flags, `nvdisasm -gi` line info to attribute each instruction to its outermost source line in
`mma_pipe`. Instruction counts are per consumer thread (one warp issues one instruction for 32).

## 1. Work per sequence (PackGQA M128 x N192, 64 Q heads, 4 KV heads, D 128, FP8)

An M-tile is 8 tokens x 16 heads, so the M-tiles of one KV head never straddle a 192-key block
when seqlen_k == seqlen_q (the only case the fixed Prefill path serves). Each tile has exactly
one masked KV block, its diagonal (the masked loop of `mma_pipe` runs zero times).

| L | M-tiles | KV-block visits | blocks/tile | tile GFLOP | effective GFLOP | padding |
|---|---:|---:|---:|---:|---:|---:|
| 3986 | 1996 | 21,756 | 10.90 | 273.75 | 260.38 | 4.89% |
| 4001 | 2004 | 21,924 | 10.94 | 275.87 | 262.34 | 4.90% |
| 4027 | 2016 | 22,176 | 11.00 | 279.04 | 265.76 | 4.76% |
| 4096 | 2048 | 22,880 | 11.17 | 287.90 | 274.95 | 4.50% |

- The padding is all in the diagonal blocks: on average a warpgroup's 64 rows can see at most
  111 of the diagonal block's 192 keys (32-key granularity), 128 at 64-key granularity, and
  about 100 exactly.
- KV traffic into shared memory: 48 KiB (K and V) per visit, 1.08 GB per 4001-token sequence,
  against 4.1 MB of unique K and V. At 404 µs that is about 2.7 TB/s from L2; how much of it
  hits L2 is not measured.

## 2. Instructions of the pinned row

| region (per consumer thread) | instructions | MUFU.EX2 | FFMA | FADD | FMUL | FMNMX | FSEL/ISETP | F2FP | QGMMA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| steady KV step (`fwd_step`, no mask) | 587 | 98 | 100 | 96 | 66 | 100 | 0 / 2 | 48 | 10 |
| tile boundary (next diagonal QK, mask, softmax, finalize, direct-O store) | 1316 | 96 | 98 | 98 | 138 | 168 | 98 / 126 | 144 | 10 |
| first tile of a CTA (once per CTA) | 716 | 96 | 98 | 94 | 0 | 98 | 98 / 97 | 48 | 4 |
| last PV and store of a CTA (once) | 480 | 0 | 0 | 4 | 134 | 70 | 0 | 96 | 6 |

- Per tile (10.9 blocks): about 1,316 + 9.9 x 587 = 7,160 instructions and about 1,070 MUFU.EX2
  per thread.
- The steady step: 98 MUFU.EX2 = 96 elements plus 2 row rescales; FFMA = x * scale - max;
  FADD = the sequential row sum; FMUL = 64 O rescales plus 2; FMNMX = the sequential row max;
  48 F2FP pack P to E4M3; `permute_Cregs_fp8` costs one PRMT (register renaming does the rest).
  The 69 others are uniform-datapath descriptor arithmetic, barriers and branches.
- The boundary's 168 FMNMX include 70 for the group-128 output amax; its 144 F2FP are 48 for
  P, 64 BF16 packs (the output is rounded to BF16 before the FP8 group quant, as n192 does) and
  32 E4M3 for O. The diagonal mask is 98 FSEL plus about 97 ISETP.
- The XU pipe takes 8 cycles per warp MUFU, so one warp's softmax needs at least 784 XU cycles
  per step and the step of both warpgroups 1,568 per SMSP. The H100 stamps of 40981460928 put the
  steady step at 2,936 cycles and the boundary softmax at 2,640, so a warpgroup's softmax phase
  (about 1,470 cycles) is about half XU-bound issue and half latency (max chain, sum chain,
  quad shuffles) and is longer than its GEMMs (768 tensor cycles).
- Fill and drain (stamp builds, H100): per tile the pin spent 10,135 cycles outside its steady
  steps; the pipelined row still waits 2,300-3,100 cycles for Q and spends 1,330-1,970 in the
  store. Waits only return 0.36-0.6x under the power cap.

## 3. Candidate A: trim the diagonal block's softmax (implemented)

In the boundary step each warpgroup computes the largest key column any of its 64 rows may
see in the next tile's diagonal block n = n_block_max - 1:

    limit = min(t_last + 1 + seqlen_k - seqlen_q, seqlen_k) - n * 192,   t_last = (m_block * 128 + 64 * (wg + 1) - 1) / 16

which is the largest `col_limit_right` that `Mask` gives those rows (padding rows included).
With k = clamp(ceil(limit / 32), 1, 6), made warp-uniform by a lane-0 shuffle, one branch per
k runs the causal mask and the `Is_first` softmax on the leading 32k columns only, through a
view of the first 16k accumulators and a 32k-column `Mask` with its column origin at block n.
The P conversion is the pin's code over the whole block; then the P words of the skipped
k-blocks are set to 0x00. QK and PV run over the whole block as before.

Why every output byte equals the pin's:
- The skipped columns are -inf for every row of the warpgroup in the pin.
- FMNMX with -inf leaves the row max; FADD.FTZ with exp2(-inf) = +0 leaves the row sum (a sum
  of FTZ results is never denormal); the kept columns run the same instructions on the same
  values in the same order (`thread_reduce_` walks columns in increasing order).
- The pin converts the skipped P to 0x00, which is what the trimmed row writes, so the PV GEMM
  reads the same registers.
- Skipping masked entries is not "skipping computation" under the precision policy: the causal
  mask defines them as absent, as FA3 already skips fully masked blocks.
- `test_fa3_prefill_diag.py` checks the bound against the Mask limits for every M-block and
  warpgroup at 13 lengths and three prefix cases. The in-job screen checks the bytes against
  n192 per request.

Removed work (k uniform over 1..6 at 3986-4096, mean 3.49):

| | per thread and tile | share |
|---|---:|---:|
| diagonal mask + softmax | 600 -> 24 + 96k (mean 359) | |
| added: sub-block count, softmax dispatch, branch-free zero selects (20 SEL + predicates) | about +47 | |
| added: steady step 587 -> 588 (a re-materialized barrier id) | about +10 | |
| net instructions | about -185 of 7,160 | -2.6% |
| MUFU.EX2 | -40 of 1,070 | -3.7% |
| QGMMA, V transposes, KV traffic | unchanged | 0 |

Expected saving with power-cap factor 1 for the removed MUFU/ALU work: 2.3-2.9% of 404 µs,
**about 9-12 µs per sequence-call** (the critical-path view, the 2,640-cycle boundary softmax
shortened by about 1,100 cycles per warpgroup and tile of about 40,000, gives the same range).
Per two-request layer call about 18-24 µs, per 24-layer PP0 period about 0.4-0.6 ms of the
1.5 ms that 10k needs.

SASS: the pinned row's SASS is byte-identical with the patch applied; every other FA3
translation unit is unchanged (section 6). The new row: 168 registers at launch, spills
84/116 B like the pin, no wgmma serialization warning (CUDA 13.4 here; CUDA 12.9.1 on the
Genkai login node gave the same registers and spills and no warning).

The first commit (7f4a64e) zeroed only a quarter of the skipped P bytes: the FP8 view of the
tail had a quarter of its elements before the 32-bit recast. The byte gates (screen and
endurance) would have rejected it. The next commit fixes it and adds a static_assert
that the 32-bit view covers the whole tail; nvcc turns the six zeroing branches into 20
branch-free selects.

### Structures that did not compile well (CUDA 13.4)

| structure | result |
|---|---|
| QK GEMM with N = 32k inside the branch (removes the QK tensor work too) | C7511: every wgmma of the kernel serialized |
| the same with the sub-block count not provably uniform | C7519 |
| branch from QK issue to the last GEMM wait, or P converted into a second buffer | C7513 (the second buffer is coalesced with the in-flight PV operand) |
| P conversion inside the branches | about 50 register MOVs per branch at the merge |
| skipped scores zero-filled inside the branches | +5 instructions per steady step (re-materialized scale and barrier id) |
| 64-key sub-blocks | +5 per steady step, and 128 instead of 111 keys kept |

## 4. Other levers the accounting shows (not implemented)

1. **QK and PV of the diagonal block on the trimmed keys** (tensor work about 1.9% each of the
   kernel's QGMMA FLOPs). Needs wgmma without branches: a guarded (predicated) QGMMA if ptxas
   keeps the pipeline, or a separate N-specialized diagonal step. Trimming PV's k-blocks is
   byte-exact only if a k32 QGMMA whose products are all zero leaves the FP32 accumulator
   unchanged; Hopper's FP8 accumulation is documented as truncating, so this needs a GPU check
   (the in-job byte gate would catch it).
2. **V transposed once, not per visit.** The producer transposes every V tile (LDSM.T / PRMT /
   STSM) on each of 21,924 visits, 48 KiB of extra shared-memory traffic per visit, while
   unique V is 2 MB. A K-major V written by the KV store (the earlier N224 `VByteImage` idea,
   which failed its dump check) would remove it. Producer work is off the consumer critical path,
   so only its power share returns.
3. **Fewer KV visits per M row**: M256 tiles or a 2-CTA cluster that multicasts K/V would halve
   the 1.08 GB of L2-to-SM traffic; register budget and the persistent scheduler make this a
   large change.
4. **Q-wait at the tile boundary** (2,300-3,100 cycles on H100): already attacked in round 4
   (double-buffered Q, early claim, both slower); waits return only 0.36-0.6x.

Not repeated: polynomial exp (+5.2%), half-precision exp, N224, staged-Q rows, lazy rescale
(-0.4 µs per layer), tree max (+1.2-6%).

## 5. In-job gate

PP4 branch of `platforms/firmus/framework-comparison/sglang.sbatch`, before the workers start,
on the four GPUs of each Prefill worker, the three workers in parallel:

1. `select-firmus-prefill-kernels.sh --names n192-direct-o-varlen-pipe-diag --control <pin>`
   (175 s timeout): byte equality with n192 per request on three replays, FP32 reference,
   >= 5 µs per 94-layer graph layer at 3986, 4027, 4096 and 3986+4027 on all twelve GPUs.
2. After all three screens, `select-firmus-prefill-fa3.py --endurance` (60 s timeout): 1200
   serving-path calls over six shapes, each shape's last output equal to the pinned bytes.
3. The aggregate adopts the row only if every GPU passed both; otherwise the pin stays.

Added serial time before worker start: about 1.2-1.5 min (41040348302: 34 s inside the screen's
torchrun for one row, 7.2 s inside the gate's). The served pipe row is now in `UNGATED`.
The image needs a rebuild (one more FA3 translation unit).

## 6. Evidence

- SASS identity (CUDA 13.4, sm_90a, normalized SASS md5): all 27 other translation units of
  the FA3 companion (11 Decode verify rows, 12 Prefill rows including the pin, combine, the
  N224 prefill, prefill_fixed, prefill_image, decode_prepare) are identical with and without
  `prefill-diag.patch`; the pin is 9bdf52b13cca in both. The new row is c52497a1836d.
- Genkai probe: `kernels/probes/prepare-fa3-prefill-diag.sh` (pin vs diag, 3986+4027, H100,
  byte equality and FP64 reference, 3 trials of 20 alternating pairs). Run 41068992667 built
  on the login node and submitted PJM job 7218165, which `b-batch` scheduled for 11/04 00:00
  (five days). It also carried the quarter-tail bug. The job was deleted and the run ended
  cancelled with no GPU result; its runtime directory is gone. No second dispatch: the queue
  does not move, and the in-job screen and gate cover what the probe would measure.
