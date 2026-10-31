# Round 11: direct-FP32 QKV epilogue (2026-10-31)

## Design, before implementation

Base d6134fd; branch perf/prefill-qkv-fused-r11-20261031. Control image
41150387984, SHA256 e7b09add5c72e0758df2639cd3200bfa1e1ababbeea250ec8fda8caa23053f62.
Official control score 9896.09 is serving evidence, not evidence for this candidate.
This note was written before the candidate CUDA code. Status and measurements
will be appended; the estimates below are conditional.

### Source and execution contract

The prepared image tree (image branch b08ebde) was cloned privately and all 29
PP4 SGLang overlays applied in config order, checking each before application.
DeepGEMM 1c8138a was cloned privately, then the complete host series, image
header series and 13 PP4 header overlays applied with the same checks.
Original trees and source mirrors are read-only. CUDA compile target is 13.4
sm_90a. No SSH, Firmus/ASPIRE job or cluster-tagged commit is permitted.

Read the round-10 and complete Prefill dense notes (including the rejected
round-9 gate), the kernel theory tracker/model and profile provenance, the
FP32 writer validation, QKV probe history, common worker conversation brief,
and system/theoretical-gap analyses. The older profile describes EP4, not the
current PP4 timing. No clock or timeline from it is combined with round 10.
The retained 2026-10-24 shared-memory fusion lost (326.55 versus 293.41 us at
3986 rows); round 9's empty-wave skip also lost its actual gate. Neither is
included here. The former run 40048324159 completed in about 4m22s including
preparation/cleanup; that is a probe budget precedent, not a new measurement.

In patched qwen3_moe.py, forward_prepare_native calls decode_dense_linear;
Prefill EXTEND routes to prefill_fp8.linear. A prequantized tuple is used as-is;
otherwise sglang_per_token_group_quant_fp8 makes E4M3 A with FP32 group-128
column-major TMA-aligned scales. Weight is contiguous E4M3 [9216,4096] with
row-major FP32 block scales [72,32]. Bias is absent. DeepGEMM fp8_gemm_nt with
compiled_dims="nk" selects BM256/BN128/BK128, 3 stages, 128 producer and 256
math threads, cluster 2 with A multicast, persistent 132-SM scheduler. Normal
GEMM computes every 64-row wave, even the empty M-tail waves. Preserve this
mainloop, stage/phase and tile order. No stagger, tail-work skip or host patch.

qkv_proj packs Q64, K4, V4 heads (split q_size=8192, kv_size=512).
Each N tile is exactly one 128-column head. The FP32 writer runs only with
contiguous Prefill and no byte V image. Current pinned varlen FA3 uses this
row-major path. The producer sets fixed_prefill_output, prefill_v_image=None,
_stored_fp8_kv_last_call=True, _used_fused_qk_norm_rope_last_call=False.
The FA3 consumer slices real rows using extend_seq_lens_cpu and cu_seqlens_q;
paged KV is still required for transfer. These attributes must be retained.

### Exhaustive output list (no output-scale tensors)

| N tile | output | dtype / layout | operation |
|---|---|---|---|
| 0..63 | Q | E4M3 [M,64,128], contiguous; returned flat [M,8192] | per-head RMSNorm, BF16 norm weights, FP32 NeoX RoPE, NOSAT FP8 |
| 64..67 | contiguous K | E4M3 [M,4,128], contiguous | same math with k_norm weights |
| 68..71 | contiguous V | E4M3 [M,4,128], contiguous | direct NOSAT conversion of accumulator |
| 64..67 | paged K | E4M3 [cache_slots,4,128], contiguous | identical codes to contiguous K at cache_loc[row] |
| 68..71 | paged V | E4M3 [cache_slots,4,128], contiguous | identical codes to contiguous V at cache_loc[row] |

**Q, K and V all have implicit scale 1.** qknorm_rope_prefill_fp32 writes no
scale tensor, no amax, no BF16 Q/K output, and no BF16 QKV output mutation.
Do not add group scaling to this path. FA3 q_descale/k_descale/v_descale and
layer.k_scale/v_scale are None. Its FP8 attention-output group-128 scales
belong to the downstream FA3/O path and are not Q scales. The removed BF16
QKV buffer is a dead GEMM intermediate. No V byte image is written by fp32.
Slot 0 is never modified, but its token still produces contiguous K/V.
Other cache slots retain their previous bytes. Valid cache_loc is in range.

### Register ownership and math

For math thread t=0..255, warp=t/32, quad_lane=t%4:
row = tile_m*256 + warp*16 + (t%32)/4 + wave*128 + r*8,
where wave=0,1 and r=0,1. For chunk j=0..15 and pair element e=0,1,
column = 8*j + 2*quad_lane + e, accumulator index = wave*64 + 4*j + 2*r + e.
Thus each m64n128 fragment gives a thread two rows of 32 values; BM256 keeps
two such fragments (four rows, 128 final FP32 values). c and c+64 are local
(indices separated by 32), and each row's full head spans one four-lane quad.

For Q/K: sum 32 squares per thread with FP32 FMA, quad XOR-2 then XOR-1
reductions (two shuffles), rho=rsqrtf(sum/128 + eps). This uses the same
rsqrtf as the current writer. For c<64, u=acc[c]*weight[c],
v=acc[c+64]*weight[c+64], then (u*cos-v*sin)*rho and (u*sin+v*cos)*rho.
No BF16 round after GEMM, norm or RoPE. FMA/order changes are allowed. V skips
norm/RoPE. All outputs use cvt.rn.satfinite.e4m3x2 plus the current NOSAT
repair: magnitude >464 becomes signed NaN; input NaN becomes canonical +NaN.
No clamp, changed quantization granularity or approximate norm is permitted.

Per valid Q/K row/thread, a first instruction budget is 32 square FMA,
2 SHFL + 2 FADD, 1 scale FMA + 1 rsqrt, 8 pairs of four BF16 weight conversions,
32 weight/RoPE/scaling arithmetic operations per four columns (about 112
arithmetic instructions for the row), 16 packed FP8 conversions plus overflow
checks/rare repair. Load 32 BF16 weights and 32 FP32 cos/sin values, plus one
position (positions is int32 or int64, matched to cache_loc). Work estimate is
about 300-400 executed instructions per Q/K row/thread including addresses,
packing and ordinary-path guards, versus roughly twice the current writer's
per-lane work because its row uses eight lanes. SASS is the required check;
this is not a measured count. V needs conversion and stores only. No amax.

Mainloop budget: final_accum 128 + accum 64 = 192 FP32 registers under the
stock 232-register math allocation. Epilogue must reuse dead accum registers.
Process one row and one RoPE pair at a time in place; avoid copying the 128
accumulators or loading all weights/trig at once. Budget approximately 32
row values already held, 8 temporary math/load registers, 8 packed output
registers, 4 transpose registers and row/address state. Persistent scheduler
and live pointers remain the risk: a local compile must prove no mainloop
spill before any probe. Stop on a mainloop spill, TMA issue, unreproducible
output or unavoidable image rebuild, as requested.

### Row validity and stores

TMA A bounds and scale descriptors keep stock behavior. M7988 has 32 tiles,
last tile starts at 7936 and only 52 rows are valid; M3994 has 16 tiles and
last tile has 154 rows. M129 has one tile and 129 valid rows. Every wave still
participates in all full/empty barriers. Mask metadata reads and every output
store with row<M; shuffles execute with full warp masks (validity is uniform
inside each quad). No early CTA return. Invalid rows may use position 0;
never read positions/cache_loc beyond M. No padded Q/K/V/cache stores.

positions[row] indexes the existing float32 cos_sin_cache [positions,128],
first 64 cos, second 64 sin. The model config is NeoX, not interleaved RoPE.
Q/K loads only the needed c,c+64 pair. V does not load positions/trig/weights.
K/V read cache_loc[row], assert 0<=slot<cache_slots, and suppress slot-0 writes.

Each output head-row is 128 bytes. A thread naturally owns sixteen scattered
2-byte pairs (8-byte quad segments). To reach 16-byte stores without SMEM,
pack adjacent chunks, transpose a 4x4 uint32 register matrix inside the quad
(two butterfly stages, four SHFL plus selects), then byte-permute into one
uint4 per half-row per thread. Each lane writes its contiguous 16-byte piece
at offset lane*16 and 64+lane*16. Each Q row uses two STG.128 per thread; K/V
also duplicate these into the paged cache. No BF16 shared staging/TMA store,
no new inter-warp barrier, no shared output scratch. Keep stock shared offsets
and allocation initially (216448 bytes) to keep the mainloop unchanged.

### Runtime overlay, host and routing

New standalone SGLang JIT source copies the pinned 1D2D body and includes the
image's DeepGEMM device headers from Path(deep_gemm.__file__).parent/include,
as decode_swapab_fp8 already does. No DeepGEMM header or native host change.
TMA A [4096,M], B [4096,9216] are UINT8 swizzle128 with boxes [128,256] and
[128,128]; A scales [M,32] use column-major byte stride sa.stride(1)*4 and
box [256,1], FLOAT32/no swizzle. Build descriptors with
cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",12000), exactly as
the existing JIT decode_swapab_fp8 host launcher, not direct libcuda linkage.
Use its LaunchKernel device/stream helper with cluster(2,1,1), PDL enabled,
132 CTAs/384 threads and dynamic SMEM opt-in. No D descriptor is needed.

Append one overlay after all 29 current SGLang overlays. Only the exact env
value SGLANG_PREFILL_QKNORM=fused routes an eligible Prefill call before
qkv_proj to this launcher. Preserve every pre-existing path for other values.
Use the same quantizer when the activation is not already an FP8 tuple.
Require existing precision/cache/contiguous guards and no V byte image.
No gate until the H100 probe passes every tensor's FP64-relative max/mean
absolute error and code mismatch criterion, tail/cache guards, endurance,
and >=3 percent paired improvement at M7988. If adopted later, all 12 Prefill
GPUs must pass; no Firmus/ASPIRE execution or serving commit in this task.

### Conditional savings and prior differences

At M7988 QKV is approximately 0.55*(950..981)=523..540 us; fp32 post ~81 us.
The eliminated BF16 output is M*9216*2=147234816 bytes written and re-read
(294469632 logical bytes); these are not a measured HBM count. No NVLink
traffic changes. GEMM useful work is 2*M*9216*4096=603073806336 FLOP;
all this MMA and promotion work remains, including padded waves. Expected
net saving target 45..60 us/layer, 1.08..1.44 ms for 24 layers, is conditional
on the register epilogue costing less than the removed post+BF16 store.
Removed instructions/bytes/kernels use power-cap factor ~1; any exposed wait
or overlap benefit must separately use 0.36..0.6. Do not count all 81 us plus
the BF16 read again. H100 ratios cannot be converted directly to H200 times
or a serving score. Reject at the specified H100 threshold rather than claim.

Unlike 4b68df3, this supports the actual BM256 two-wave ownership (the removed
candidate asserted BM128), performs direct FP32 math, drops the two old post
BF16 roundings and the GEMM BF16 rounding, uses the current hardware FP8
conversion with repair, and uses 16-byte register-transposed stores. Unlike
946d2e0/277b500, no output shared-memory staging/reload remains. Bit equality
with the current writer is not a criterion; FP64 error is. All output dtypes,
layouts, scale=1, cache semantics and downstream interfaces are retained.

## Local implementation and compile evidence

The appended overlay adds qkv_fused_prefill.cuh, its TVM-FFI wrapper and a
narrow eligible Prefill producer. The existing writer remains the fallback;
the settings pin stays fp32. DeepGEMM native host/device files are unmodified.
The A-scale tensor map uses the padded inner extent sa.stride(1), exactly as
DeepGEMM make_tma_sf_desc's TMA-aligned size. The logical output shape is M.

CUDA 13.4.3 sm_90a with register-usage-level=10: both int32/int64 full JIT host
exports compile, 168 launch registers, math SETMAXREG 232, stack/spill stores/
loads all zero, 216448 bytes requested dynamic shared memory. Actual buffer
layout occupies 216240 bytes; the stock allocation rounds up. No WGMMA
serialization warning. A normal assert initially triggered ptxas C7510;
replacing that cache-bound check with the existing kernel style's device trap
removed the function call and warning. No mainloop change was needed.

Stock QKV compiled alone and beside the full candidate has identical SASS:
2376 instructions and 64 QGMMA. Candidate totals 5888 static instructions and
64 QGMMA, with no LDL/STL. The source-attributed epilogue count and opcode
histogram are in prefill-qkv-fused-cpu-validation-20261031.json. Counts include
all four row fragments, both Q/K/V branches and cold NOSAT repairs; they are
not executed instruction counts. There are 64 packed FP8 conversions, four
MUFU.RSQ, 260 epilogue FFMA and sixteen STG.E.128. All epilogue data stays in
registers; no new shared-memory intermediate. This does not claim the mainloop
has identical register allocation/scheduling to stock, only the same source
body and arithmetic. Stock itself is instruction-for-instruction unchanged.

## Probe attempt and stop condition

Commit 7fde480 was pushed only to this branch; it starts with skip-ci and has
no cluster tags. One Genkai workflow dispatch was made (one of at most four):
[41159506648](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41159506648),
2026-10-31 02:20:00–02:21:30 UTC, terminal **failure**.

The first fatal command was preparation's image size check. At 02:20:29 UTC:

```
stat: cannot statx '/home/ACCOUNT/share/sglang-containers/41150387984.sqsh': No such file or directory
```

This is the normal Genkai shared-image directory used by existing prepare
scripts, with the exact PP4 pin's basename. The image was not available there.
No PJM job was submitted, no GPU kernel ran, and the workflow cleanup step
succeeded. The later missing provenance message in summary is secondary to
this error. CUDA 13.4 availability on Genkai was not reached or verified.
Do not substitute an older image or infer kernel failure/success from this.
Under the requested stop rule, no further dispatch or gate work was attempted.

| rows | paired fused / stock+fp32 | Q/K/V and paged KV FP64 errors | 300-call endurance |
|---:|---|---|---|
| 129 | not measured | not measured | not run |
| 3994 | not measured | not measured | not run |
| 7988 | not measured | not measured | not run |

Step 3 is incomplete. The >=3% criterion is **unknown**, not failed or passed.
Step 4 is not implemented. The settings stay fp32; no selection env, sbatch,
provisioning or native-host build change was made. Image rebuild is not
required by this implementation. A copy of the exact existing image must be
available to the allowed Genkai probe before GPU work can proceed; this task
made no SSH connection or Firmus/ASPIRE job. Added serving-job time is **0 s**
because the candidate is off and no gate is wired. Future selector JIT/time
cost is unmeasured and must not be borrowed from the tail-skip's 29–30 s.

## CPU verification and limits

The final full Python 3.12 suite, eight workers: **1240 passed, 69 skipped,
25 subtests passed**, 145.78 s. The prepared-image chain test ran explicitly,
with all 30 SGLang overlays checked and applied in config order; all native
DeepGEMM host/image and 13 PP4 header overlays also checked/applied separately.
Ruff on qwen-experiment-metrics/scripts/containers and the new probe/reference,
Python compilation, bash syntax for the preparation and unchanged sbatch,
and check-firmus-pp4-source.py pass. Existing overlay-position tests were
updated for the appended default-off patch. The final patch applies despite
unprefixed blank context lines, which avoid trailing whitespace in the patch.

CPU random-tile model (4096 heads, 128 values; BF16 random norm weights,
FP32 cos/sin) versus FP64: unquantized max abs 1.4973748268e-6, mean abs
7.9216418121e-8; FP8-code mismatch 1/524288 = 1.9073486328e-6. The BF16-input
stand-in's mismatch is 0.0292739868. This stand-in is not the actual GPU writer
and no GEMM was executed; these are CPU epilogue-model checks only. They do
not establish any GPU tensor error bound, paired speed or serving improvement.

Further work, ranked: (1) make the exact pinned software available to the
allowed Genkai probe and resolve the numeric/speed uncertainty; (2) only after
that evidence, inspect normal-path conversion/address overhead in the fused
epilogue; (3) prepare FA3 varlen metadata once per microbatch rather than per
layer; (4) investigate dense epilogue overlap, which needs a different register
budget and uses the lower 0.36–0.6 wait-removal factor. No rejected mainloop
stagger, empty-wave runtime branch, coarser quantization or new approximation
was reintroduced.
