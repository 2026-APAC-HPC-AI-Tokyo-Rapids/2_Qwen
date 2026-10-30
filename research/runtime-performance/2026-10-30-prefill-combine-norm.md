# Prefill MegaMoE combine fused with the next layer's input add-RMSNorm (round 6, 2026-10-30)

Base: main cd262ed (round 4, 9832.74, run 41005872470). PP4/EP1 Prefill, 8064-row
microbatches, layer partition 24/24/24/22. This is lever 2 of
[2026-10-30-prefill-megamoe-split-ldsm.md](2026-10-30-prefill-megamoe-split-ldsm.md) §4.

## 1. What follows the MegaMoE call today

The analysis uses the full SGLang overlay chain of `configs/firmus-prefill-pp4.json` with the
pinned PP4 settings: onepass MoE input, FP32 Q/K norm and the input norm/quant kernel.

Layer L of a Prefill stage (`Qwen3MoeDecoderLayer.forward`), EXTEND with at least 2048 rows:

| step | kernel | reads | writes |
|---|---|---|---|
| input norm | `norm_quant` (Triton, `add_rmsnorm_group_quant_fp8`, rows >= 2048) | y(L-1) BF16 66 MB, residual 66 MB, weight | residual BF16 66 MB in place, q FP8 33 MB, scale FP32 1 MB (column-major, rows padded to 4) |
| QKV GEMM | DeepGEMM 1d2d | (q, scale) | QKV BF16 |
| Q/K norm, RoPE, FA3, O GEMM | ... | | o_proj output BF16 |
| MoE input | `onepass_moe_input` | o_proj output, residual | residual += o_proj output in place; FP8 x, scales, top-k into the symmetric buffer |
| MegaMoE | `sm90_fp8_mega_moe_impl` (lean EP1, split K loop) | buffer | **y(L) BF16 66 MB**, written by the combine |
| postprocess | trivial at TP1 | | |

Therefore:

- **Where y and the residual live.** y(L) is written by the MegaMoE combine and read once, by
  layer L+1's input norm. The residual is read and written by the input norm and by the
  onepass MoE input.
- **What the next input norm consumes.** BF16 y and BF16 residual. It computes
  h = float(y) + float(residual) in FP32, stores BF16 h, and normalizes the **unrounded** h:
  rstd = rsqrt(sum(h²)/4096 + eps), then BF16((h·rstd)·w). The output is per-128-column FP8
  q, with amax clamped at 1e-10, gain = 448/amax (IEEE division), saturating RN conversion,
  and scale = amax·(1/448). These go to the QKV GEMM as a (q, scale) tuple.
- **Stage boundaries.**
  - The last layer of a stage sends `hidden_states` (y) and `residual` through `PPProxyTensors`.
  - The next stage's first layer runs the input norm on the received pair.
  - The model's last layer (stage 3) feeds `model.norm` (fused add-RMSNorm, BF16 output) and then
    lm_head.
- **EAGLE3 capture layers.** These layers (`_is_layer_to_capture`) take
  `prepare_attn_and_capture_last_layer_outputs`, not the norm/quant path.

Measured in the PP4 profile 40825417205 (prefill-0, GPUs 0–3, 900–1035 calls each):

- `norm_quant`: **60.8 µs** median. That is 232 MB at 3.8 TB/s.
- Gap from the MegaMoE end to the `norm_quant` start: 2.9 µs.
- The QKV GEMM overlaps the norm's tail (gap −0.1 µs).

## 2. Design

`fp8_mega_moe_norm` makes the combine of layer L also do layer L+1's input norm/quant.

The kernel flag is `kPrefillCombineNorm`, the last trailing template flag on the lean EP1 tile;
it works with any lean K loop.

Per token (one combine warp owns a whole row):

1. **Combine.** Each of the four 2 KiB chunks is reduced exactly as now (ascending slot order,
   FP32 FMA) and rounded to BF16 y. The lane's residual elements of the chunk are loaded before
   the pair waits.
2. **Residual add.** h = float(y) + float(residual) (`__fadd_rn`).
   - BF16 h goes through the existing staging buffer and TMA store to the y pointer. The caller
     passes the residual as y.
   - The unrounded h goes to a per-warp 16 KiB FP32 row in shared memory: float4 pairs, lane-major,
     conflict-free.
   - Σh² is accumulated per lane (FMA).
3. **Norm and quant.** The sum goes through a 32-lane XOR tree, then
   rstd = rsqrt.approx(sum·2⁻¹² + eps). In a second pass over the lane's own h:
   - BF16((h·rstd)·w).
   - A 16-lane XOR max: each 128-column group is 16 lanes of one uint4 index.
   - max(·, 1e-10), gain = 448/amax with `__fdiv_rn`, min(·, 448), saturating RN e4m3.
   - 8-byte q stores, and the scale store by lanes 0 and 16.

**Arithmetic.**

- **Unchanged:** the combine, the BF16 rounding of y, h, the BF16 residual, and every
  normalization and quant operation.
- **Changed:** only the order of the row's sum of squares. It is lane-major then a warp tree;
  Triton uses `tl.sum`.

Allowed class: FP summation order, under the precision policy.

**Shared memory.** The paired combine buffers use 80 KiB and the rows 8 × 16 KiB. Starting at
1 KiB, that ends at 214,016 B, inside `SMEM_BEFORE_BARRIER_SIZE`. This is a static assert. All
of it is free once the GEMMs are done.

**Arguments.**

- A `SM90PrefillCombineNorm` struct (weight, q, scale, scale stride, eps) is appended as the
  last kernel parameter of every instantiation.
- The other instantiations ignore it, and their SASS is byte-identical (§4).
- The host export is `deep_gemm._C.fp8_mega_moe_norm(...)`: the `fp8_mega_moe` arguments, then
  norm_weight, eps, q and scale. The Python wrapper of the image is not changed, because only
  `_C.so` is rebuilt.
- The host asserts the lean path, hidden 4096 and top-k 8.

**SGLang** (`patches/sglang/prefill-combine-norm.patch`, last in the PP4 chain, env
`SGLANG_PREFILL_COMBINE_NORM=1`):

- `Qwen3MoeModel` links each layer to the next layer of the same stage. It uses a plain
  attribute, so the parameter names stay the same.
- Layer L fuses only when all of these hold:
  - the MoE input is `onepass`;
  - the summable-pair communicator is trivial;
  - there is no all-reduce fusion or reduce-scatter;
  - layer L+1 exists in the stage and is not a capture layer;
  - layer L+1's own `_use_input_norm_quant` would hold for this batch.
- Layer L then returns (q, scale) with the residual already updated. Layer L+1 sees the tuple
  and goes straight to attention.
- The last layer of each stage, the PP hand-off and the final norm/lm_head are unchanged.
- Per stage, 23/23/23/21 layers can fuse.

## 3. Bytes, kernels and expected time

| per layer, 8064 rows | today | fused |
|---|---|---|
| combine partial reads | 528 MB | 528 MB |
| y write + read | 66 + 66 MB | 0 |
| residual read + write | 66 + 66 MB (norm) | 66 + 66 MB (combine) |
| q + scale write | 34 MB (norm) | 34 MB (combine) |
| kernels | MegaMoE + norm_quant | MegaMoE |

132 MB and one kernel go per fused layer.

- **Removed:** the 60.8 µs norm kernel.
- **Added to the combine** (~150 µs, 594 MB at ~3.9 TB/s):
  - 100 MB at the same bandwidth: +26 µs.
  - The second pass: about 1.3 k instructions per lane per token, about 8 tokens per warp,
    2 warps per SMSP. About 15 µs of issue if nothing overlaps; mostly hidden behind the other
    warps' loads. Taken as 0–10 µs.
- The MegaMoE → QKV GEMM gap replaces the MegaMoE → norm gap, so no boundary saving is counted.
- **Net: 25–35 µs per fused layer, above the 15 µs bar.** Factor ~1: bytes and a kernel are
  removed, and the added combine work is already subtracted.
- **Per period:** 23 fused layers on a 24-layer stage, so **0.57–0.80 ms per Prefill period**
  (0.5–0.7% of 115 ms).

## 4. CPU evidence (CUDA 13.4, sm_90a)

| instantiation | registers | spill | instructions | C75xx |
|---|---|---|---|---|
| lean (served earlier) | 168 | 0 | 6856 | none |
| lean + combine norm | 168 | 0 | 8008 | none |
| split (served in 41005872470) | 168 (setmaxnreg 224) | 0 | 9856 | none |
| split + combine norm | 168 | 0 | 11000 | none |
| split-ldsm | 168 | 0 | 9784 | none |
| split-ldsm + combine norm | 168 | 0 | 10944 | none |

The combine-norm code adds, per instantiation:

- 16 F2FP.SATFINITE.E4M3;
- 4 STG.E.64 for q and 4 STG.E for scales;
- 4 LDG.E.128 for the residual and 4 LDG.E.128.CONSTANT for the weight;
- 14 STS.128 / 8 LDS.128 for the h row;
- 42 SHFL.BFLY;
- 2 MUFU.RSQ;
- 4 IEEE divisions (FCHK + MUFU.RCP with a slow-path call).

Other checks:

- **SASS identity.** The 22 instantiations without the flag are SASS byte-identical (instruction
  text and both encoding words) to the chain before this patch. These cover:
  - three-stage and four-stage EP1 and EP4 (128/140 experts, 256/512 math threads);
  - lean, pair, split and split-ldsm;
  - every Decode geometry.
- **Local host check.** `check-sm90-prefill-local-host.py` passes 368 native host
  configurations (276 before).
  - The combine norm appends `,true,{pair},{split},{ldsm},true` only on the lean path.
  - It is rejected on EP4 and on the three-stage EP1 tile.
  - The 16 selected kernels compile at 0 spill.
- **Host extension.** `csrc/tvm_ffi_api.cpp` with the full host series compiles cleanly
  (`g++ -fsyntax-only`, torch 2.11 headers, tvm-ffi 0.1.15, CUDA 13.4 and CCCL). The control
  series compiles in the same setup.
- **Evidence file:** [megamoe-prefill-combine-norm-cpu-validation-20261030.json](megamoe-prefill-combine-norm-cpu-validation-20261030.json).
- **CPU model of the arithmetic** (`test_prefill_combine_norm.py`, 2048 random rows):
  - The lane-order sum changes rstd in 89% of rows.
  - Yet 9.5e-7 of the q bytes differ, no scale differs, and the dequantized relative L2 is 8.2e-5.
  - The in-job bounds are 1e-3 (q bytes), 1e-2 (scale) and 1e-3 (relative L2).

## 5. In-job gate (`select-firmus-prefill-megamoe.py --combine-norm`)

It runs after the MegaMoE aggregate: one torchrun per Prefill worker, the three in parallel,
with a 90 s timeout. It reuses the selection's JIT caches.

If the adopted variant is on the lean path, each GPU runs:

1. **Numeric check.** The control (winner MegaMoE + `add_rmsnorm_group_quant_fp8`, the serving
   function) against `run_sm90_mega_combine_norm` (the serving function), on the balanced, hot
   and edge routings. The residual must be bit-equal; q, scale and the dequantized output must
   stay within the bounds above.
2. **Timing.** Both are timed over 12 rotating rounds on the two timed routings.
3. **Endurance.** The fused call runs 600 times back to back over the three routings and must
   reproduce its first outputs bit for bit.

`--aggregate-combine-norm` appends `SGLANG_PREFILL_COMBINE_NORM=1` only if all of the following
hold; otherwise every layer keeps the separate norm:

- all twelve GPUs gated the adopted variant;
- they all passed;
- they all finished the 600 calls;
- the summed median saving is at least Σ max(5 µs, 0.5% of control).

**In-job time.**

- About 30–45 s per worker, in parallel: container and torchrun start, weights, one DeepGEMM
  JIT (~10 s) and one Triton JIT, then about 2 s of calls.
- The Prefill-GPU chain becomes:
  - MegaMoE selection, about 1.7–2.2 min with the sixth variant;
  - endurance gate, 0.2–0.7 min;
  - combine-norm gate, 0.5–0.75 min.
- That is about 2.5–3.6 min, against 4–4.5 min for the Decode-GPU chain, so it adds no job time.
- Worst case with every timeout: 240 + 90 + 90 s.

## 6. What only the GPU shows

| question | covered by |
|---|---|
| combine time with +100 MB and the second pass (the 25–35 µs net) | the gate's paired timing; adopted only above the margin |
| fused (q, scale) against Triton on real data | the gate's numeric bounds; residual bit-exact |
| hang or drift over hundreds of calls | the gate's 600-call endurance; a hang kills the step and the norm stays separate |
| the period effect at the power cap | the serving run (PP0 period and clocks) |

Image rebuild required: the host series and header overlay change.
