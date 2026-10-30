# Prefill EP1 MegaMoE round 5: what binds the split-accumulator K block (2026-10-30)

Kernel: `sm90_fp8_mega_moe_impl`, PP4/EP1, four-stage lean split tile
(`DG_SM90_PREFILL_SPLIT_ACCUM`, [round 4](2026-10-30-prefill-megamoe-split-accumulator.md)).
The split loop runs 1272-1278 clk per K block in the mainloop model against 1090 clk for the
serving loop without the rescale. This note tests which term is left.

## 1. Two candidate terms

1. **Chunk hand-off.** ptxas tracks WGMMA groups only in straight-line code, so K blocks run in
   unrolled chunks drained at the chunk end. At each chunk start the warpgroup rescales half 0
   with nothing in flight, and both warpgroups tend to drain together (they wait on the same
   stages).
2. **Shared memory.** Each m64n128 half re-reads the warpgroup's 64 A rows, so per K block the
   WGMMAs read A 32 KiB instead of 16 KiB. With TMA writes (48 KiB) and B reads (64 KiB) that is
   144 KiB per K block, 1152 clk at 128 B/clk, against 128 KiB (1024 clk) for the serving loop.
   The four-part split of round 4 (176 KiB, 1408-clk floor) measured +10%, consistent with this.

## 2. Mainloop model variants (`kernels/probes/megamoe-mainloop.cu`)

| variant | change |
|---|---|
| split-norescale-u8, -k12-u12 | split loop without the rescale (timing only): its floor |
| split-pre-u8/u16 | at a chunk end, wait for half 0 of the last K block, rescale it for the next K block while half 1 runs, then drain; the next chunk issues half 0 at once (ratio 1 at the tile end: exact) |
| split-stag-pre-u8 | as pre, warpgroup 1's chunks are 4, 8, 8, 8, 4 K blocks (boundaries between warpgroup 0's) |
| ldsm-u8, ldsm-pre-u8/u16, ldsm-stag-pre-u8, ldsm-k12-u12, ldsm-pre-k12-u6 | A read once per K block by four `ldmatrix.x4` per warp from the TMA 128B-swizzled stage into 16 registers (two K blocks in flight, 32 registers); both halves issued in the RS form (`QGMMA.64x128x32 ... R, R, gdesc`). WGMMA A reads 32 -> 16 KiB per K block |
| ldsm-norescale-u8 | ldsm without the rescale (timing only) |

The 2026-10-20 header experiment (ldmatrix plus two RS m64n128, see
[2026-10-20-firmus-prefill-overlay-results.md](2026-10-20-firmus-prefill-overlay-results.md))
did not shrink the K loop, but it waited for both halves before every rescale (rescale-bound);
here the halves stay pending across K blocks as in the served split loop.

Local CUDA 13.4: no spill and no C75xx diagnostic in any variant. A first stagger form with two
separate half-chunk instances spilled 172-216 bytes and serialized the WGMMAs (C7512); one loop
with one half-chunk instance and one chunk instance does not.

Results: Genkai run 40993048750 (queued at the time of writing; see §5).

## 3. Kernel candidate `DG_SM90_PREFILL_SPLIT_LDSM`

- Host `patches/deep-gemm-host/sm90-prefill-split-ldsm.patch` (series after the split): on
  the split path only, the trailing template text becomes `,true,false,true,true`
  (kPrefillSplitLdsm).
- Header `patches/deep-gemm-bench/prefill-split-ldsm/sm90-prefill-split-ldsm.patch` (after the
  split overlay): `split_k_loop` loads A of the first K block after the caller's stage wait and
  A of the next K block after the next stage's wait (beside the scale prefetch), into the
  register buffer of K block u % 2; both halves are issued with
  `MMA_64x128x32_F32E4M3E4M3_RS_TN`. Same products, K order, ratios and final scale: the same
  bytes. Chunks stay even (FC1 8, FC2 12).
- CUDA 13.4, sm_90a: 0 spill, registers up to R201 (split R206) inside the unchanged 224-register
  budget, no C75xx; 160 QGMMA all with register A, 84 LDSM; FC1 194 and FC2 180 instructions per
  K block (split 202 / 191). The other 21 instantiations (three/four stage, lean, pair, split,
  EP4, every Decode geometry) are SASS byte-identical to the round-4 chain.
- `check-sm90-prefill-local-host.py`: 276 native host configurations PASS (ldsm only appends
  its flag on EP1), 12 selected kernels at 0 spill.
- `test_prefill_megamoe_selection.py` replays the ldmatrix lane addresses against the TMA 128B
  swizzle and checks that every thread receives exactly the m64k32 FP8 A fragment.
- Evidence: [megamoe-prefill-split-ldsm-cpu-validation-20261030.json](megamoe-prefill-split-ldsm-cpu-validation-20261030.json).
- In-job selection: `four-stage-lean-split-ldsm` in `scripts/select-firmus-prefill-megamoe.py`,
  adopted only if bit-exact on the balanced, hot and edge routings and faster in total by the
  summed margin on all 12 Prefill GPUs; otherwise the job keeps the current winner. Added in-job
  time: one more JIT compile (8.7 s against 8.4 s for the split under CUDA 13.4) plus about 30
  calls per GPU, about 10 s per worker in parallel across the workers.
- Image rebuild required (host series).

## 4. Combine fused with the next layer's add-RMSNorm (not implemented)

After the MegaMoE of layer L (onepass input path) the next Prefill op is layer L+1's
`add_rmsnorm_group_quant_fp8(y, residual)`: residual += y in place and the FP8 (q, scale) for
the QKV GEMM. In the PP4 profile (40825417205, prefill-0) that kernel is 59.8 us per layer for
8064 rows: it reads y and residual (66 + 66 MB) and writes residual and q + scales (66 + 34 MB),
232 MB at 3.9 TB/s. Each combine warp already owns whole token rows (7.64 tokens per warp), so
it could add the residual, reduce the row's sum of squares in the warp, normalize and quantize
per 128-column group. Bytes: the combine reads residual (+66 MB) and writes residual instead of
y (0) plus q + scales (+34 MB); the norm kernel goes away (-232 MB). Net -132 MB and one
kernel boundary per layer, about 35-40 us (factor 1, bytes removed), 0.8-1.0 ms per 24-layer
period. Costs: the MegaMoE call takes layer L+1's norm weight and the residual and q/scale
pointers (host and SGLang changes, image rebuild); the last layer of each stage keeps the
current path; the warp-level sum of squares changes the reduction order (allowed class, but a
tolerance gate is needed rather than the bit-exact MegaMoE gate).

## 5. Status and levers

- Model run 40993048750 was queued on Genkai b-batch for hours (estimated start up to 22:00
  JST). The ldsm kernel is wired behind the in-job gate; without the model number it is an
  unmeasured candidate.
- Ranked further levers: the chunk hand-off (pre, stagger) if the model shows it; the combine
  fusion (§4, about 1.3% of MegaMoE plus the norm kernel); tail 64-row A box (0-1%); FC1
  epilogue MUFU (inexact class).
