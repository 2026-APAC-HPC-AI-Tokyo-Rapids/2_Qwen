# Prefill EP1 MegaMoE: one reciprocal for two FC1 sigmoids (2026-10-30, round 7)

Kernel: `sm90_fp8_mega_moe_impl`, PP4/EP1, four-stage lean split tile (the serving
variant `four-stage-lean-split`). Item 1 of the levers in
[2026-10-30-prefill-megamoe-ep1-accounting.md](2026-10-30-prefill-megamoe-ep1-accounting.md):
the FC1 epilogue issues 128 MUFU per thread and tile, and with two math warps per SM
sub-partition its floor is 2 x 128 x 8 = 2048 clocks per tile (MUFU is 16 lanes per clock
per SM, 8 clocks per warp instruction). No GPU was used for this note.

## 1. Change (`DG_SM90_PREFILL_SHARED_RCP`, variant `four-stage-lean-split-rcp`)

- Host `patches/deep-gemm-host/sm90-prefill-shared-rcp.patch` (series after the split
  accumulator): on the split path, with fast math and no activation clamp, the template
  text becomes `,true,false,true,true` (kPrefillLean, kPrefillTailPair=false,
  kPrefillSplitAccum, kPrefillSharedRcp).
- Header `patches/deep-gemm-bench/prefill-shared-rcp/sm90-prefill-shared-rcp.patch` (after
  the split accumulator in `configs/firmus-prefill-pp4.json`).
- The split FC1 K loop no longer applies the final scales (`postscale_l1_final`, 128 FMUL per
  thread). It hands the row scales `s_gate`, `s_up` to the epilogue.
- Per row and column pair (c0, c1) of a thread:
  - `e_i = ex2(acc_g,i * (s_gate * -log2 e))` (the same `ex2.approx.ftz`),
  - `d_0 = 1 + e_0`, `d = fma(d_0, e_1, d_0)` (= d_0 d_1),
  - `R = rcp(d) * s_gate` (the same `rcp.approx.ftz`),
  - `v_0 = (acc_g,0 * fma(e_1, R, R)) * acc_u,0`, `v_1 = (acc_g,1 * (d_0 R)) * acc_u,1`.
- The values are the control's SwiGLU in units of `s_up`. The row amax is multiplied by `s_up`
  (exact: rounding is monotonic), and the quantizer multiplier is `s_up / sf`.
- `s_gate` sits in the reciprocal, not in the accumulator product. So no intermediate
  exceeds the control's value / `s_up`. A gate-up product of raw accumulators would overflow
  when a token's last K group has a tiny activation scale.
- Fallback. A chunk of 4 pairs per row is recomputed with one reciprocal per element,
  `(acc_g * (rcp(1 + e) * s_gate)) * acc_u`, if any lane of the warp has
  `d >= 2^126 min(1, s_gate)`.
  - Below that bound, R and `R s_gate` are normal FP32 numbers.
  - An earlier draft used 2^126 alone. Its `R s_gate` went denormal, and the CPU model then
    showed 1.8e-3 relative scale errors.
  - The bound is reached for a gate below about -40, or two below about -20, at `s_gate` near
    1e-5.
  - The `__any_sync` vote makes the branch warp-uniform. ptxas therefore branches around the
    64 fallback RCPs instead of predicating them, and reuses the EX2 results.
  - A per-pair `if` compiled to 64 predicated MUFU.RCP. A whole-tile fallback kept all
    accumulators live and spilled 72 bytes.
- A NaN gate spreads only to the other column of its row. That FC2 output row is NaN in the
  control too.
- Precision class: the same EX2 and RCP instructions, FMA and FP-order changes only. FP8 bytes
  may move by one step, so the variant is tolerance-gated (section 4).

## 2. SASS (CUDA 13.4, sm_90a)

FC1 epilogue span per thread, from the first postscale FMUL after the K loop to the last
`STG.E.128` of the row copy. For the candidate, the fast path excludes the fallback blocks
that the `@!P BRA` after each `VOTE.ANY` skips.

| | split (serving) | shared rcp, fast path |
|---|---:|---:|
| instructions | 865 | 855 |
| MUFU.EX2 / MUFU.RCP | 64 / 66 | 64 / 34 |
| FMUL / FFMA / FADD / FMNMX | 390 / 4 / 66 / 66 | 333 / 68 / 34 / 100 |
| registers / spill | 168 (setmaxnreg 224) / 0 | 168 (224) / 0 |

- Fallback blocks: 288 instructions, 64 MUFU.RCP, no EX2.
- The FC1 K-loop chunk has 1502 vs 1499 instructions (register allocation only) and 1020 FMUL
  both. The FC2 loop is 2212 in both.
- The other 21 instantiations (three- and four-stage, lean, pair, split, EP4 and every Decode
  geometry) are SASS byte-identical.
- Removing split-ldsm and combine-norm from the chain also left all 21 byte-identical,
  including the serving split kernel.
- `check-sm90-prefill-local-host.py`: 276 native host configurations PASS. All 12 selected
  kernels compile with 0 spill.
- Details: [megamoe-prefill-shared-rcp-cpu-validation-20261030.json](megamoe-prefill-shared-rcp-cpu-validation-20261030.json).

## 3. Expected saving

Per FC1 tile and SM sub-partition (two math warps):

| | split | shared rcp |
|---|---:|---:|
| issue slots | 1730 | 1710 |
| MUFU pipe (clocks) | 2080 | 1568 |
| floor (clocks) | 2080 | 1710 |

- The floor drops by 370 clocks per tile, and the bound moves from MUFU to issue.
- Per layer: 52 FC1 tiles per SM, so 19.2 k clocks, which is 13.5-14.8 µs at 1.30-1.43 GHz.
- The epilogue measured 2.0-3.1 k clocks per tile, so the floors explain only part of it.
  Taking 0.5-1.0 of the floor change (removed pipe work, power-cap factor ~1): **7-15 µs per
  layer (0.25-0.55% of MegaMoE), 0.17-0.36 ms per 24-layer Prefill period**.
- This is below the noise of a single GPU. The selection adopts it only if the summed
  medians beat the split.

## 4. Numerics and gate

CPU model of the epilogue and FC2:
- Every FP32 operation rounds to FP32. EX2 and RCP are modelled as correctly rounded
  `exp2` and reciprocal.
- e4m3 RNE SATFINITE. FC2 over the dequantized rows, then BF16.

| case (rows x FC2 columns) | fallback pairs | FP8 bytes changed | y relative L2 | y max abs diff / max abs y |
|---|---:|---:|---:|---:|
| selection-like, gate/up std 0.48 (2048 x 1024) | 0 | 6.4e-7 | 3.9e-5 | 2.5e-3 |
| serving-like, t3 tails (2048 x 1024) | 240 | 9.5e-7 | 2.0e-5 | 4.1e-4 |
| in-job wide, std 19 (2048 x 1024) | 3549 | 9.5e-7 | 2.3e-5 | 2.3e-3 |
| extreme, std 96 (2048 x 1024) | 655795 | 6.4e-7 | 9.8e-5 | 3.2e-3 |
| selection-like (8192 x 4096) | 0 | | 9.1e-5 | 4.7e-3 |
| in-job wide (8192 x 4096) | 14097 | | 8.2e-5 | 4.1e-3 |

- The scale `sf` moves by at most 4.1e-7 relative.
- Predeclared gate in `scripts/select-firmus-prefill-megamoe.py`:
  - The output must be finite.
  - Relative L2 against the control must be at most **1e-3** (10x the worst case).
  - Max abs diff / max abs control must be at most **2e-2** (4x the worst case). One BF16
    step of a top-binade element alone can reach 2^-7.
  - The FP32 reference check must pass on the balanced, hot, edge and new wide routings.
- The wide routing is the balanced routing with hidden states 40 times larger (gate std about
  19). Part of the chunks then take the fallback, so both paths run on the GPU. It is untimed,
  and the bit-exact variants must match the control on it too.
- Endurance gate: an unserved winner runs 600 back-to-back calls over the four routings. A
  tolerance-gated winner must reproduce its own first output bit for bit (determinism). The
  selection has already compared that output with the control.

In-job time:
- The variant replaces split-ldsm in the selection, so the JIT compile count is unchanged
  (9.4 s for this kernel against 9.2 s for the split under CUDA 13.4).
- The wide routing adds 6 calls, one 32-token FP32 reference and 5 compares per GPU (under
  0.5 s).
- The endurance gate runs only if this variant wins (about 20-40 s, as for split-ldsm in round
  6).
- All of this is in parallel across the three workers and beside the Decode selections.

## 5. Removed in this round

- split-ldsm (lost by +1.6% in 41020974704): host patch, header overlay, selection variant
  and tests.
- combine-norm (slower in 22 of 24 cases): SGLang overlay, host patch, header overlay,
  `--combine-norm` selection code, tests and its host-check hunks.

Their notes stay.
