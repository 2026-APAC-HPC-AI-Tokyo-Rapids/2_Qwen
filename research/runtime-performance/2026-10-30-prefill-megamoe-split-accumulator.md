# Prefill EP1 MegaMoE: the K block is rescale-bound, and a split accumulator hides it (2026-10-30, round 4)

Kernel: `sm90_fp8_mega_moe_impl`, PP4/EP1, four-stage lean 2WG tile (128 x 256 x 128, two
m64n256 warpgroups). The round-3 accounting
([2026-10-30-prefill-megamoe-ep1-accounting.md](2026-10-30-prefill-megamoe-ep1-accounting.md))
put the 1215-clk K block (1024-clk tensor floor) on the 128 B/clk shared-memory port and
proposed removing shared-memory bytes per MAC (WGMMA RS form with A from global, B
multicast). This note measures those models before building anything; the shared-memory
model does not hold.

## 1. Mainloop model

`kernels/probes/megamoe-mainloop.cu` (`prepare-megamoe-mainloop.sh`, shared one-GPU Genkai
runner) is a standalone persistent kernel with the serving FC1 K loop: four 48 KiB stages,
an A+SFA TMA warp and a B TMA warp, two math warpgroups with one m64n256k32 FP8 accumulator
each, and the carried-scale rescale (ratio of the previous to the next `s_a * s_w`, 128 FMUL
per thread) between WGMMA groups. 132 CTAs, 44 experts x 4 pool blocks x 12 N blocks, K = 4096
(16 tiles of 32 K blocks per SM) in the scheduler's FC1 order; the `-k12` variants use 12 K
blocks per tile (the FC2 length). 3 trials x 40 rounds of all variants in rotating order;
ratios are medians of per-round ratios. The Genkai H100 ran at 1.33-1.54 GHz under this load
(power-limited, like the Firmus H200s at 1.40-1.44 GHz), so the time ratios include the
clock response.

| variant (what changes) | time / control | clk per K block | run |
|---|---:|---:|---|
| ss (serving loop) | 1 | 1405-1407 | all |
| ss-ares: no TMA of A (16 KiB fewer shared-memory writes and L2 reads) | 0.975-0.980 | 1394 | 40978170082, 40979704662, 40980472965 |
| rs-const: RS form, constant A (all 32 KiB of A traffic gone) | 0.981-0.987 | 1415 | same |
| ss-mcb: 2-CTA clusters, B multicast (L2 -> SM 48 -> 32 KiB) | 1.001-1.009 | 1418 | same |
| rs-ldg: RS form, A from global in fragment order, 2 K blocks ahead | 1.339-1.367 | 1961-2017 | same |
| rs-nat: RS form, A natural layout (`ld.global.cg`), B K-permuted | 1.610-1.634 | 2421-2466 | same |
| ss-norescale: no rescale (wrong numbers, bound) | 0.828-0.844 | 1088-1095 | same |

- Shared-memory bytes and the L2 fill are not binding: removing all A traffic saves 2%,
  multicast saves nothing.
- The WGMMA RS form with A from global loses 34-61%. The tensor core waits on the register
  operand loads (L2 latency per K block, and with the natural layout four 16-byte loads per
  thread whose rows straddle sectors). The RS form itself is legal for FP8 e4m3 on sm_90a
  (`QGMMA.64x256x32.F32.E4M3.E4M3 R24, R180, gdesc`, A fragment `ALayout_64x32`: thread t holds
  rows 16w + (t%32)/4 and +8, bytes 4(t%4)..+3 and +16 per k32 step; rs-ldg was bit-equal to
  ss on every tile). It costs 32 registers for a 2-stage prefetch.
- The carried-scale rescale is the binding term: without it the K block drops from 1406 to
  1088-1095 clk. In the serving loop each warpgroup does wait (group k) -> 128 FMUL -> issue
  (group k+1) on one m64n256 accumulator, and both warpgroups wait on the same stage, so the
  tensor core idles while they rescale.

## 2. Split accumulator

Each warpgroup issues its m64n256 accumulator as two m64n128 halves (B rows 0..127 and
128..255 of the same stage, the same A). Per K block, for h = 0, 1: `wgmma.wait_group 1`
(the oldest pending group is half h of the previous K block), release the previous stage after
both halves, rescale half h (64 FMUL), issue half h. The other half and the other warpgroup
stay queued on the tensor core. Each output element gets the same products, K order, ratios
and final scale: the same bits (checked on every tile of the model).

ptxas only tracks WGMMA groups in straight-line code. With a group pending across a loop edge,
or a branch around a rescale, it injects a full wait (C7517) or serializes every WGMMA (C7514).
So K blocks run in unrolled chunks of U, drained (`wait_group 0`) at the chunk end. Each chunk
start waits with `wait_group 1` before half 1 (a no-op that tells ptxas the half is not in
flight). The next K block's scales are prefetched while the current group runs, as in the
serving loop; without that, each drain cost about 340 clk (run 40979704662).

| variant | time / control | clk per K block |
|---|---:|---:|
| split2-u8 (FC1 tile, 32 K blocks) | 0.944-0.948 | 1277-1279 |
| split2-u16 | 0.937-0.940 | 1272-1274 |
| split2-u32 (whole loop unrolled, ~10.6 k instructions) | 2.36-2.38 | 3230-3277 |
| split2-k12-u4 (FC2 tile, 12 K blocks) | 0.946-0.947 | 1327-1329 |
| split2-k12-u6 | 0.954-0.955 | 1338 |
| split2-k12-u12 | 0.924-0.926 | 1293-1295 |

The u32 collapse is an instruction-cache cliff between about 6.7 k (u16 kernel) and 10.6 k
instructions. Four parts (m64n64) lose (split4-u4 1.105): each part re-reads A and the groups
are short. The remaining gap to the no-rescale bound (1278 against 1090 clk) is the exposed
rescale of half 0 at chunk starts and the extra A reads of two halves.

## 3. Kernel candidate `DG_SM90_PREFILL_SPLIT_ACCUM`

- Host `patches/deep-gemm-host/sm90-prefill-split-accum.patch`: lean EP1 tile without tail
  pairing; trailing template text `,true,false,true` (kPrefillLean, kPrefillTailPair=false,
  kPrefillSplitAccum).
- Header `patches/deep-gemm-bench/prefill-split-accum/sm90-prefill-split-accum.patch` (last
  MegaMoE overlay in `configs/firmus-prefill-pp4.json`).
- The unpaired FC1 and FC2 K loops use the split loop with U = 8 (FC1, 32 K blocks) and U = 12
  (FC2, 12 K blocks). U = 8 instead of 16 keeps the FC1 loop at about 1.5 k instructions,
  away from the cliff, for 0.7% of FC1. The paired-tail and padding-only paths are unchanged.
- The math warpgroups take 224 registers by setmaxnreg (64 x 48 + 64 x 40 + 256 x 224 =
  62,976). ptxas used none of this budget while a trap was reachable in the math role: the
  build spilled 304/344 bytes at 168. The split instantiation therefore inlines the lean
  pre-combine barrier (grid sync, SM 0's self-signal through the same workspace counters)
  without its 300 s timeout trap. It also compiles out the runtime combine-buffer assert,
  which a static assert already covers. Result: 0 spill, registers up to R206, no C7514/C7517.
- SASS (CUDA 13.4, sm_90a): FC1 188 and FC2 184 instructions per K block (lean: 274 / 243),
  8 QGMMA and 128 FMUL per K block. The other 22 instantiations (three/four stage, lean, pair,
  EP4, every Decode geometry) are byte-identical.
- `check-sm90-prefill-local-host.py`: 230 native host configurations PASS, 10 selected kernels
  at 0 spill.
- Details: [megamoe-prefill-split-accum-cpu-validation-20261030.json](megamoe-prefill-split-accum-cpu-validation-20261030.json).
- In-job selection: `scripts/select-firmus-prefill-megamoe.py` gains `four-stage-lean-split`
  (`DG_SM90_PREFILL_4STAGE=1 DG_SM90_PREFILL_LEAN=1 DG_SM90_PREFILL_SPLIT_ACCUM=1`). Like the
  others it must be bit-exact on the balanced, hot and edge routings and save the summed margin
  on all 12 Prefill GPUs, or the job keeps the current winner.
- Added in-job time: one more JIT compile (8.2 s against 4.9 s for lean under CUDA 13.4) plus
  about 30 calls per GPU (under 0.2 s). That is about 10 s per worker, in parallel across the
  three workers and beside the Decode selections, inside the step's 240 s timeout.
- Image rebuild required (host series).

## 4. Expected saving

From the round-3 budget (2.63 ms per layer at 1.43 GHz): FC1 mainloop about 1.36 ms and FC2
about 0.68 ms including their padding shares. With the model's ratios (FC1 u8 -5.2..-5.6%,
FC2 u12 -7.4..-7.6%, both measured at a power-limited clock, so the clock response is
included), that is 70-77 + 50-52 = about 125 us per layer (4.7%). The serving loop already
gets some de-synchronisation between warpgroups (epilogues, dispatch) that the model lacks,
so take 0.6-1.0 of that: **75-125 us per layer (2.8-4.7% of MegaMoE), 1.8-3.0 ms per
24-layer Prefill period (1.5-2.6% of 117 ms)**. That is short of the 4.7 ms/period target on
its own. GPU-only: speed and exactness on the real kernel (H200, power cap). The in-job
selection gate covers both.

## 5. Levers not done (ranked)

1. **Fewer drains.** U = 16 for FC1 (-0.7% of FC1 in the model) if the real kernel's hot code
   stays under the instruction-cache cliff; a loop form that ptxas can track across the edge
   would remove the chunk-start exposure entirely (model: about 1278 -> ~1200 clk).
2. **Padding** (7%): a 64-row A box for tails (bit-exact, 0-1%).
3. **Combine fusion** with the next add-RMSNorm (y write and read, about 25 us per layer).
4. **FC1 epilogue MUFU** (inexact, FP-order class, 0.5-0.7%).
5. Rejected by measurement here: RS form with A from global (+34..61%), B multicast (+0.1..0.8%),
   four-part split (+10%).
