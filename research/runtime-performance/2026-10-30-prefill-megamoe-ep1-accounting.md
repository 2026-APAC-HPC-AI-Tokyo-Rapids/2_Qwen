# Prefill EP1 MegaMoE: where the 2.6 ms go, why round 2 lost, and the lean variant (2026-10-30)

Kernel: `sm90_fp8_mega_moe_impl`, PP4/EP1, 8064-token microbatch (64,512 routed rows),
four-stage 2WG tile (128 x 256 x 128, two m64n256 warpgroups, 168 registers), 132 CTAs.
Sources: in-job raw results of 40925558339 and 40911121619 (12 GPUs x 2 routings x 12
rounds per variant), the prefill-0 nsys sqlite of 40825417205 (GPU metrics inside the
MegaMoE windows), the real routing capture `kernels/probes/megamoe-local-ep1-routing.npz`,
SASS of the current chain under CUDA 13.4. No GPU was used for this note.

## 1. Measured anchors

| quantity | value |
|---|---|
| in-job four-stage median, balanced / hot routing | 2524-2655 / 2552-2723 us per GPU (40925558339) |
| serving MegaMoE, prefill-0 (three-stage, 40825417205) | median 2766-2825 us, p10 2584-2702, p90 2891-2922 |
| GPC clock inside MegaMoE (nsys) | 1343-1422 MHz |
| SM Issue / Tensor Active inside MegaMoE | 37-39 % / 54-58 % |
| DRAM read / write inside MegaMoE | 18-23 % / 6-9 % of 4.8 TB/s (about 2.9 GB read, 1.2 GB written per call: weights 2.4 GB, partials, pool, y) |

FP8 peak at the measured clock: 2 x 8064 x 8 x 4096 x 1536 x 3 = 2.435 TFLOP per layer.
132 SMs x 8192 FLOP/clk x 1.43 GHz = 1.546 PFLOP/s, so the useful work alone is
**1.575 ms** (1.64 ms at 1.37 GHz). The kernel runs at **59-60 % of FP8 peak**.

## 2. Tile accounting (real routing; the in-job hot routing is within 1 %)

Real routing: 567-580 pool blocks of 128 rows for 64,016 rows (pad128 13.4-16 %), 69-86
tails of at most 64 rows (run by one warpgroup, measured 0.64-0.70 of a full tile), 23-40
tails of 65-127 rows. Tile-equivalent blocks 546-553 against 500 ideal (+9-10 %).
Per SM: 52 FC1 tiles (32 K blocks each), 69.5 FC2 tiles (12 K blocks each), about 2400
tile-equivalent K blocks.

| part | per layer at 1.43 GHz | share | basis |
|---|---:|---:|---|
| FC1 mainloop, useful rows | ~1240 us | 47 % | 1597 eq. K blocks/SM x 1215 clk, minus padding |
| FC2 mainloop, useful rows | ~620 us | 24 % | 798 eq. K blocks/SM x 1215 clk, minus padding |
| padding and half-tile overhead (both GEMMs) | ~180 us | 7 % | 9 % of the mainloop |
| FC1 epilogue (SwiGLU, amax, FP8, stores) | 75-115 us | 3-4 % | 52 tiles x 2.0-3.1 k clk: 128 MUFU/thread, 2 warps/SMSP = 2048 clk floor; 3.1 k clk measured on EP4 |
| FC2 epilogue (BF16 pack, staging, row stores) | 35-60 us | 1-2 % | 69.5 tiles x 0.7-1.2 k clk; 223 instructions/thread; 512 full-line 128-B stores per tile |
| combine | ~150 us | 6 % | 528 MB partial reads + 66 MB y write at ~4 TB/s (TMA, 4 loads in flight per warp) |
| dispatch and first wave | ~15 us | 0.6 % | count + 2 grid syncs + SM 0 republication; the first FC1 wave needs 1408 pool rows = 5.3 dispatch rounds |
| pool copy (overlapped with FC1) | 0-25 us | 0-1 % | 8 KiB of SMEM per row, 3.9 MB/SM = 1.8 % of FC1's SMEM traffic |
| barriers and tails | ~10-15 us | 0.5 % | before-combine barrier (2 grid syncs + signal), last FC2 wave, combine token imbalance (7.64 tokens per warp) |
| residual | ~150-250 us | 6-9 % | stage waits beyond the 1215-clk K block (that figure excludes waits) and the clock uncertainty |

Against the peak: useful 1.575 ms + padding 0.155 ms + mainloop efficiency (1215 against
1024 clk per K block, 84 %) about 0.32 ms + outside the mainloop about 0.58 ms.

**Correction (round 4).** A standalone model of this K loop on Genkai
([2026-10-30-prefill-megamoe-split-accumulator.md](2026-10-30-prefill-megamoe-split-accumulator.md))
refutes the shared-memory-port explanation below: removing all A traffic saves 2%, B multicast
nothing, and removing the carried-scale rescale saves 17% (1406 -> 1090 clk). The rescale
between WGMMA groups binds the K block; the split-accumulator candidate hides it.

The mainloop per K block moves 128 KiB through shared memory (TMA A 16 + B 32 KiB, WGMMA
operand reads A 2 x 8 + B 2 x 32 KiB) in 1024 tensor clocks: exactly the 128 B/clk
shared-memory port. The data (4WG 144 KiB = 1330 clk, 2WG 128 KiB = 1215 clk) fit both
an SMEM-bound model and a tensor-plus-rescale-convoy model (both warpgroups wait on the
same stage, so their 2 x 128 FMUL rescales leave the tensor idle about 190 clk); the
forced alternation of da53188 removed the MMA wait and still lengthened the loop by 5 %,
so the shared-memory port is the binding term. No tile that fits 168 registers moves fewer
bytes per MAC (B is read once per m64 slice by WGMMA). The mainloop has no bit-exact work
left to remove apart from padding.

## 3. Why round 2 lost

**Direct FC2 (+1.6 %, +43 us per layer).** It removed 64 KiB STS + 64 KiB LDS per FC2
tile and about 97 instructions per thread, but the FC2 epilogue is not bound by shared
memory. Both paths issue 16 STG.128 per thread per tile. The staged path's warp store
writes 4 rows x 128 B (whole lines); the permuted direct store writes 8 rows x 64 B, so
the SM sent 1024 instead of 512 store requests per tile, half-line each, and the store
path became the epilogue floor (about +512 clk x 69.5 tiles = 25 us). The combine's
natural-order restore added 4 x STS.32 per uint4 with 4-way bank conflicts (about 8 us),
and the accumulators stayed live until the last STG issued, so the next tile's first
WGMMA waited on the store queue. Predicted +33-45 us against the measured +43 us.

**A gather (+44 %, about +1150 us per layer).** The copy it removed (dispatch warps,
TMA in and out, 3.9 MB of SMEM per SM) ran beside FC1 and cost at most 1.8 % of FC1's
shared-memory traffic. The replacement put the gather on the FC1 critical path: one loader
warp issued 32 cp.async per lane per K block with 57 instructions per four copies (about
460 warp instructions per K block, against one TMA instruction), each stage completing only
when the last lane's 16-byte copies landed from L2, plus a proxy fence per K block on the
math side. FC1's K block went from about 1215 to about 2000 clk (FC1 is about two thirds of
the kernel). On sm90 there is no TMA gather; a per-row gather into the 128B swizzle costs
one generic instruction per 16 bytes. Both designs counted the bytes they removed but not
what those bytes overlapped with.

## 4. Lean EP1 variant (`DG_SM90_PREFILL_LEAN=1`, bit-exact)

Host `patches/deep-gemm-host/sm90-prefill-lean.patch` (EP1, four-stage) appends `,true`
(kPrefillLean). Header
`patches/deep-gemm-bench/prefill-lean/sm90-prefill-lean.patch`:

- Every CTA adds its per-expert counts straight into the receive sums that the scheduler
  polls (ready at kNumSMs arrivals, the same condition as SM 0's single sum). SM 0's
  republication with system-scope atomics, its self-signal and the second dispatch grid
  sync are gone. The slot offsets are the same atomic returns.
- The pull loop takes rank 0 and the slot itself instead of the rank round robin (2 REDUX,
  BALLOT, POPC, FNS per row; matters for the first wave).
- Before the combine, the grid sync after SM 0's self-signal is gone (a single grid sync in
  place of the whole barrier lengthened the FC1 K loop by 9 instructions through register
  allocation and was not used).
- The FC2 staging writes four n8 chunks per `stmatrix.x4` (16 per thread per tile instead
  of 64 STS.32), same bytes at the same slice offsets (the mapping is checked in the tests).

CPU evidence (`megamoe-prefill-lean-cpu-validation-20261030.json`): 168 registers, 0
spill; FC2 epilogue span 223 -> 178 instructions per thread; FC1 K loop identical length,
FC2 K loop +2 allocation-only instructions; all 21 other instantiations SASS byte-identical;
184 host configurations PASS.

Expected 7-13 us per layer effective (sync removal at kappa 0.36-0.6, instruction removal
at kappa ~1), 0.17-0.31 ms per 24-layer stage period. This is at or below the selection's
0.5 % summed margin; the selection keeps four-stage unless the measured saving clears it.

## 5. Paired tails (`DG_SM90_PREFILL_TAIL_PAIR=1`, on the lean path, bit-exact)

A pool block with at most 64 valid rows used to run each N block on one warpgroup
(the other consumed the stages idle), 0.64-0.70 of a full tile. Now its N blocks run in
pairs: the item of N block 2j computes 2j and 2j+1 on the two warpgroups (rows 0..63,
warpgroup g on N block 2j+g) and uses two stages per K block (A, SFA and warpgroup 0's B
in the first, warpgroup 1's B in the second); the item of N block 2j+1 is skipped. Per
two tail tiles: TMA 80 instead of 96 KiB, both warpgroups' WGMMA busy. Each warpgroup
computes its N block with the same rows, B bytes, K order, scales and epilogue as before:
same output bytes. The tail split (m64n128 halves, below margin twice) is removed.

Deadlock argument. All roles (A loader, B loader, math warps) walk one item stream per CTA
through `sm90_fp8_mega_moe_for_each_block_dynamic_l2`, with their own scheduler copies:
FC1 is the static round robin over the expert counts every role polls until final, FC2 the
claimed words the A loader publishes through the ring that the B loader and the math warps
consume (skipped words are consumed and released like any other). The skip (`n` odd and
valid rows <= 64) and the pairing (valid rows <= 64) are one predicate,
`sm90_fp8_mega_moe_tail_pair`, of values each role holds identically, evaluated in the
driver (skip) and at the top of each role's item (pair). Per processed item every role
advances the ring by K blocks (unpaired) or 2 x K blocks (paired); skipped items advance
nothing. So all roles reach every item at the same (stage, phase), whatever the item
count parity, and an expert with 0 rows has no items for any role; an expert whose rows
fit one half tile has exactly one pool block, which is paired. Per stage use the full
barrier (count 2) gets the A loader's arrival (with A + SFA bytes, or byte-less on the
second stage of a pair) and the B loader's (with B bytes); the empty barrier (count 8)
gets one arrival from each math warp, which also waits on both full barriers of the pair.
Producers wait on the empty barrier of every stage they use, as before. The FC2 A loader's
FC1 arrival count is unchanged (12 N blocks x ceil(valid / 64) x 4: a tail block gets 4
per warpgroup per N block either way). `test_prefill_megamoe_tail_pair.py` replays the
per-role streams for the 36 archived 128-expert distributions, the selection routings and
edge distributions and asserts all of this; the in-job selection adds an untimed "edges"
routing (experts with 0, 1-64, 65, 127-129, 191-193, 255-256 rows) that every candidate
must reproduce bit for bit.

CPU evidence (`megamoe-prefill-tail-pair-cpu-validation-20261030.json`): 168 registers, 0
spill; unpaired K loops 284 / 225 instructions (lean 274 / 243, allocation only), paired
K loops 311 / 284 per K block; the 18 instantiations without the new flags and the lean
instantiation are SASS byte-identical; 184 host configurations PASS.

Expected: 78 tails per layer at 3.7 us per tail-tile unit: 0.06-0.19 of a tile saved per
tail tile, 17-55 us before the power-cap factor, 10-55 us after (kappa 0.6-1), 0.4-2 %.
Risk: a paired K block uses two of the four stages, so only one paired K block is
prefetched ahead.

## 6. Larger levers not implemented here

1. **FC1 epilogue MUFU (inexact, about 0.5-0.7 %).** One `rcp` for two sigmoids
   (1/a = b/(ab), clamped exponent) and the per-row scale folded out of the SwiGLU product
   cut MUFU 128 -> 96 per thread per tile; FP-order class change, needs a tolerance gate.
2. **Tail A box of 64 rows (bit-exact, 0-1 %).** A paired tail still loads the 128-row A
   box; a second descriptor would halve it (8 KiB per paired K block).
3. **Combine (6 %).** Bytes are fixed (BF16 partials); only overlap with the FC2 tail
   (kappa 0.36-0.6) or a downstream fusion with the next add-RMSNorm (saves the y write and
   read, about 25 us) remain.
