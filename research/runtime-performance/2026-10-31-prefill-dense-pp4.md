# Prefill QKV and O GEMMs at the PP4 shape: config accounting (2026-10-31)

- Branch: `perf/prefill-dense-r9-20261031`, based on 18fec80.
- CPU analysis only: no GPU job, no Genkai dispatch.
- Result: **no DeepGEMM config is worth a Prefill override.** No host patch, gate or image change was made.
- The one work-removing lever at this shape is outside the config space: skip the empty 64-row waves of the M tail block (section 5).

## 1. The serving shape is M = 7988, not 8064

- The round-3 PP4 profile (40964714831, `nsys-prefill-0.nsys-rep`, all four PP stages of prefill-0) feeds every QKV GEMM from a `norm_quant` launch.
  - Its grid is 7988 rows in 3420 of the 4320 layer calls and 3994 rows in the rest.
  - No 8064-row call exists.
- The client sends 3986–3990 tokens per request (AIPerf export, identical in 40964714831 and 41073991312). With the chat template the server sees 3994.
- The worker logs print `#new-token: 8064` (11524 times in round 8). That is the page-aligned count: ⌈3994 / 64⌉ × 64 = 4032 per request. It is not the GEMM M.
- gigatoken (round 8) changes only the tokenizer backend. The client ISL histogram is unchanged, so M stays 7988.

Per-call medians at M = 7988 (40964714831, prefill-0, by PP stage):

| stage | QKV (µs) | O (µs) |
|---|---:|---:|
| 0 | 512.6 | 446.1 |
| 1 | 508.3 | 439.5 |
| 2 | 503.7 | 436.4 |
| 3 | 509.3 | 443.6 |

24 layers × (509 + 441) µs ≈ 22.8 ms per microbatch, in line with the 12.3 + 10.7 ms of the brief.

## 2. Stock config (reproduced and confirmed)

`get_best_config<SM90ArchSpec>` of DeepGEMM v0.1.5 (1c8138a, `csrc/jit_kernels/heuristics/sm90.hpp`) was reproduced in Python.

**Candidate set**
- BM {64, 128, 256}: 256 is allowed because the output is BF16.
- BN 16..192 in steps of 16.
- The 1D2D unroll rule removes BN 176.
- BM > 128 with BN > 128 is excluded.
- Clusters (1,1), (1,2), (2,1).
- At least 3 stages, or 4 when BM × BN < 128 × 192.

**Model**
- L1/L2 cycles, with L2 capped at 6153 B/clk.
- Divided by wave efficiency.

The heuristic picks the same config at 7988, 8064, 3994 and 4032 rows. The kernel names in the profile carry exactly these template arguments:

| | QKV (N 9216, K 4096) | O (N 4096, K 8192) |
|---|---|---|
| tile | 256 × 128 × 128 | same |
| swizzle A / B / D | 128 / 128 / 128 B | same |
| stages | 3 | 3 |
| shared memory | 216,448 B | 216,576 B |
| threads | 128 TMA + 256 math (2 warpgroups) | same |
| multicast | cluster 2, on A (`cluster_n = 2`) | same |
| scheduler | persistent, 132 SMs, static `iter × 132 + blockIdx` | same |
| compiled dims | `nk` (`prefill_fp8.linear`) | same |
| CUDA 13.4 sm_90a | 168 launch registers (232 math after setmaxnreg), 0 spill, 2376 instructions, 64 QGMMA, 1024 FFMA | 168, 0 spill, 2336, 64, 1024 |

Waves and padding at M = 7988:

| | tiles | waves | last wave | slot use | executed rows | M padding |
|---|---:|---:|---:|---:|---:|---:|
| QKV | 32 × 72 = 2304 | 17.45 → 18 | 60 / 132 | 96.97% | 8192 | 204 rows = 2.49% of executed MMA and promotion |
| O | 32 × 32 = 1024 | 7.76 → 8 | 100 / 132 | 96.97% | 8192 | same |

- No N padding (9216 = 72 × 128, 4096 = 32 × 128).
- The tail M block (rows 7936–8191) has 52 valid rows. Its four 64-row waves (2 warpgroups × 2 `local_idx`) all run: `is_computation_valid` is always true for `GemmType::Normal`.

**Per-tile time model.** Fitting t = a + b × k-blocks to both GEMMs (same tile, same 96.97% slot use) gives:
- a ≈ 1.2–1.5 µs per tile: the epilogue and the next tile's ramp, not overlapped with MMA.
- b ≈ 0.84–0.85 µs per 128-deep k-block: about 1180–1220 cycles at 1.40–1.44 GHz, against the 1024-cycle tensor floor (84–87%).

The model reproduces the 3994-row calls: 9 × 28.3 = 255 µs (measured 256–259) and 4 × 55.4 = 222 µs (measured 221–226).

**Round 2's "+12% schedule change" (7eb80b0, dropped in 7174620)** was not a config change.
- WG1 waited on named barrier 2 for WG0's WGMMA batch every tile, to stagger the promotions.
- It also skipped the 64-row waves past `shape_m`.
- It was keyed by `compiled_dims="n"`.
- It was bit-exact and lost on every GPU twice: 1066–1090 against 949–963 µs per layer.
- It is not repeated here. The tail skip alone has never been measured (section 5).

## 3. Legal alternatives and the work they do

Method:
- All legal layouts of the heuristic plus BLOCK_M 192 with three math warpgroups (the Decode overlay `sm90-1d2d-decode-bm192.patch`, 152 math registers) were enumerated at M = 7988.
- The scheduler's multicast pairing was simulated, including the odd-group fix.
- Byte counts per call:
  - **L2→SMEM** = TMA reads from L2: a multicast operand counts once per CTA pair, and rows past M are not fetched.
  - **fill** = bytes written into shared memory.
  - **WGMMA reads** = shared-memory operand reads: every 64-row wave reads its A rows and the whole B tile.
- Registers and spills: CUDA 13.4, sm_90a, compiled from the image's header (stock 1c8138a plus the BM192 overlay).

| proj | tile, multicast | stages | tiles / waves / slot use | executed-MMA change | L2→SMEM | fill | WGMMA reads | spill (in the MMA loop) |
|---|---|---:|---|---:|---:|---:|---:|---|
| QKV | **256×128 A (stock)** | 3 | 2304 / 18 / 0.970 | 0 | 2.36 GB | 3.62 GB | 7.25 GB | 0 |
| QKV | 192×128 A (3 WG) | 4 | 3024 / 23 / 0.996 | −1.56% | +16.8% | +9.4% | −1.6% | 1160 B (552 STL/LDL) |
| QKV | 128×192 B | 4 | 3024 / 23 / 0.996 | −1.56% | +17.9% | +9.4% | −12.5% | 2336 B (1165) |
| QKV | 192×96 A (3 WG) | 5 | 4032 / 31 / 0.985 | −1.56% | +33.3% | +31.2% | +9.4% | 0 |
| QKV | 128×144 B | 5 | 4032 / 31 / 0.985 | −1.56% | +40.2% | +24.0% | −5.2% | 0 |
| QKV | 128×160 B | 5 | 3654 / 28 / 0.989 | −0.86% | +31.7% | +18.9% | −7.5% | 0 |
| QKV | 128×128 A | 5 | 4536 / 35 / 0.982 | −1.56% | +50.5% | +31.2% | −1.6% | 0 |
| QKV | 256×128 B | 3 | 2304 / 18 / 0.970 | 0 | +25.4% | 0 | 0 | 0 |
| QKV | 256×112 A | 3 | 2656 / 21 / 0.958 | +0.87% | +25.6% | +19.8% | +4.0% | 0 |
| QKV | 64×192 B | 6 | 6000 / 46 / 0.988 | −2.34% | +67.6% | +73.6% | −13.2% | 0 |
| O | **256×128 A (stock)** | 3 | 1024 / 8 / 0.970 | 0 | 2.09 GB | 3.22 GB | 6.44 GB | 0 |
| O | 192×128 A (3 WG) | 4 | 1344 / 11 / 0.926 | −1.56% | +16.8% | +9.4% | −1.6% | 1200 B (spills) |
| O | 192×96 A (3 WG) | 5 | 1806 / 14 / 0.977 | −0.8% | +36.0% | +32.3% | +10.2% | 0 |
| O | 128×128 A | 5 | 2016 / 16 / 0.955 | −1.56% | +50.5% | +31.3% | −1.6% | 0 |
| O | 128×160 B | 5 | 1638 / 13 / 0.955 | 0 | +33.3% | +20.0% | −6.7% | 0 |
| O | 128×192 B | 4 | 1386 / 11 / 0.955 | +1.5% (N pad) | +21.6% | +12.8% | −9.8% | 2336 B (in the loop) |

The full enumeration (every legal tile, multicast side and BM 64) came from a CPU script that is not committed; the rows above are its non-dominated entries plus the stock config.

- **Stock is the traffic minimum of the legal space.** Two constraints set this:
  - The 1D2D kernel keeps the promoted accumulators in registers, which caps BM × BN at 256 × 128 (BM 192 × BN 128 and BM 128 × BN 192 already spill inside the MMA loop).
  - It supports only cluster 2.

  So 256 × 128 with A multicast gives the fewest L2→SMEM bytes (one 128-row A half and one 128-row B tile per k-block and CTA). It also gives the fewest shared-memory fills. Every alternative adds at least 16.8% L2 reads and at least 9.4% fills.
- **Every zero-padding alternative removes at most 1.56% of the executed MMA work.** It pays for that with:
  - at least +0.40 GB (QKV) or +0.35 GB (O) of L2 reads per call
  - 31–75% more tiles, each with the ~1.2–1.5 µs non-overlapped epilogue

## 4. Rule-8 ranking

Assumed power-cap factors:
- Removed MMA and promotion work on padded rows: 0.4–0.9. The A operand of those rows is TMA zero-fill, so the tensor pipe switches less than on real data. The promotion FFMAs and shared-memory reads are full work.
- Idle SM slots (wave quantization, epilogue bubbles): 0.36–0.6.
- Added L2→SMEM bytes and fills: about 1 (they are work).

Best case for each candidate (the added traffic charged at zero):

| candidate | removed padded MMA | fewer idle slots | best case per layer | verdict |
|---|---|---|---|---|
| QKV 192×128 (3 WG) | 1.56% × 509 µs × 0.4–0.9 = 3.2–7.1 µs | 23 × 0.75 vs 18 tile-times: −4.2% critical path, −2.6% beyond the padding → 4.8–7.9 µs | 8–15 µs | 552 in-loop spill accesses per thread and tile, +0.40 GB L2. Rejected. |
| QKV 128×192 | same as above | same as above | 8–15 µs, plus up to 12.5% fewer WGMMA reads | 2.3 KB spilled in the loop. Rejected. |
| QKV 128×144 (no spill) | 3.2–7.1 µs | −1.6% → 2.9–4.9 µs | 6–12 µs | +0.95 GB L2 (+40%), +0.87 GB fill, 1.75× the tiles and epilogues. Rejected. |
| O, any tile | ≤ 1.56% → 2.8–6.2 µs | negative: slot use falls from 0.970 to 0.926–0.955 | — | Net negative. Rejected. |

QKV 128×144 detail:
- At even 5% of the call's energy in L2→SMEM traffic (about 11 µs per GB), the added traffic alone costs about 10 µs.
- The extra 1728 epilogues cost about 1728 × 1.2–1.5 µs / 132 × 0.36–0.6 ≈ 6–12 µs.

**Conclusion.** None of these reaches 0.2 ms per 24-layer period (8.3 µs per layer) net, and every one adds L2→SMEM bytes. The stock config stays.
- No Prefill override (`DG_SM90_1D2D_PREFILL_CONFIG` or similar) was written.
- No in-job gate was wired.
- No image rebuild is needed.

## 5. The lever outside the config space: skip the empty tail waves

- The M tail block executes 4 × 64-row waves for 52 valid rows.
- A header-only change in `sm90_fp8_gemm_1d2d.cuh` skips the work for a math warpgroup whose wave lies wholly past `shape_m`: no WGMMA and no promotion. It still waits on the full barrier and arrives on the empty barrier at the last wave.
- This removes 3 of the 4 waves of the 72 (QKV) and 32 (O) tail tiles: **−2.34% of the executed MMA, promotion and WGMMA-read work** at 7988 rows, and −1.56% at 3994.
- It changes neither the L2/TMA traffic nor the tile order. Stored bytes are bit-identical: the skipped rows are never stored, because the TMA store clips them.
- The critical path stays at 18 / 8 tiles: only 8 of the 60 QKV and 16 of the 100 O last-wave tiles are tail tiles. The saving is energy only.

Expected saving:
- 2.34% × 950 µs × 0.4–0.9 = **9–20 µs per layer, 0.21–0.48 ms per 24-layer period**.
- At factor 1 it would be 22 µs per layer and 0.53 ms.

CPU prototype (a header diff applied after the BM192 overlay; not committed):
- The skip is inside `if constexpr (BLOCK_M == 256)`.
- CUDA 13.4 sm_90a:
  - 168 registers, 0 spill, no wgmma serialization warning.
  - QGMMA 64/64, FFMA 1024/1024, WARPGROUP.ARRIVE/DEPBAR 16/16 unchanged.
  - +144 static instructions: 32 more branches, one predicate per wave.
- The Decode instantiations 128×144 and 192×80 have identical SASS (md5 41f7c27cc1bc for 128×144).

Not done here:
- The brief limits this round to configs. The skip needs its own selection key, for example a distinct template instantiation such as round 2's `compiled_dims` trick, but not `"n"`, whose stock build spills.
- It needs an in-job timing gate: bit equality against stock at 7988/3994 rows, interleaved timing on the 12 Prefill GPUs, and a 200-call loop.
- It needs no image rebuild: it is a JIT header overlay.

## 6. Further levers, ranked

1. **Tail-wave skip** (section 5): 0.21–0.48 ms per period. JIT overlay with an in-job gate.
2. **QKV epilogue that writes the FP32 Q/K norm + RoPE + FP8 Q/KV directly.**
   - Removes the 147 MB BF16 QKV output write and its re-read (about 81 µs per layer of `qknorm_rope_prefill_fp32_kernel` plus part of the QKV epilogue).
   - The round-2 fusion attempt serialized the exact arithmetic inside the 1-CTA/SM GEMM. A fused epilogue now has to beat 81 µs plus the BF16 store, not 129 µs.
3. **Epilogue overlap.**
   - The non-overlapped per-tile epilogue and ramp is ~1.2–1.5 µs of a 28 µs QKV tile (4–5%).
   - Overlapping it needs a second accumulator set, which the 1D2D register budget does not allow at 256 × 128.
   - A 2-CTA/SM or epilogue-warpgroup design is a kernel rewrite. As idle removal it is worth 0.36–0.6 of its 4–5%.
4. **Ordering the tail tiles into the last QKV wave.** With the skip, 72 short tiles could shorten the 18-tile critical path by up to 0.75 tile (−4%) at the idle factor. It would break the L2 grouping of the swizzled order, so its L2 cost is unknown.
5. **The steady-state k-block runs at 84–87% of the tensor floor.** The promotion stagger that should hide this lost 12% in round 2. No config changes it.
