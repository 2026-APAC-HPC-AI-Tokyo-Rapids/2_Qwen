# FC2 dependency audit and two-node measurement candidate

The candidate retains the original BF16 layout, reduction order and communication
protocol. It combines five small output/scatter patches: mapped destination
hoisting, BF16 row-guard hoisting, shared row-address factoring, early metadata
loads, and two-row scatter lookahead. The final three are investigated here.
No GPU job was submitted during codegen exploration. The user then explicitly
authorized a two-node serving measurement and accepted extra instructions when
they can improve latency; the combined candidate is now enabled in
`configs/firmus-framework.json` for that one measurement.

## Implemented changes

1. Factor the shared-memory row pointer outside the BF16 pair loop and express
   the even-column XOR swizzle in uint32-pair units. The source/destination
   mapping is unchanged. This removes repeated bit/address instructions.
2. Issue the three metadata loads before BF16 packing, but leave their address
   consumers after packing. Moving the entire address calculation early was
   rejected: SASS consumed the load results before packing and simply moved the
   dependency earlier. The load-only variant preserves useful independent work
   between load and first consumer.
3. Unroll scatter in pairs and load both rows before storing the first row.
   The second load and store have the same valid-row predicate. An initial
   zero-initialized temporary generated 16 needless CS2R instructions; removing
   that initialization is safe because no path reads it without its guarded load.
   The resulting pair change adds 8 slots but removes 7 BRA instructions and
   issues the second LDS before the first remote store. It is retained as a
   performance candidate, not a measured win.

Apply these patches in order, after the original six serving overlays:

```text
patches/deep-gemm-overlay/fc2-scatter-address.patch
patches/deep-gemm-overlay/fc2-bf16-row-guards.patch
patches/deep-gemm-overlay/fc2-bf16-row-address.patch
patches/deep-gemm-overlay/fc2-early-metadata.patch
patches/deep-gemm-overlay/fc2-scatter-pair-load.patch
```

## Fresh full-kernel code generation

Genkai login CPU compilation used the pinned image's CUDA 13.0.88, GCC 13.3.1,
C++20, O3, lineinfo and compute_90a/sm_90a, with the same six full-kernel
instantiations and include tree. The image SHA is
`112ff6f0831d21e703b686df96bf950501d51a75c67950ea6a707b80197638bf`.
The base of this audit already includes the first two patches.

| Variant | Prefill128 slots | Prefill140 slots | Masked nonswap slots |
| --- | ---: | ---: | ---: |
| Original serving baseline | 7008 | 7168 | 4520 |
| Previous address + row-guard changes | 6752 | 6928 | 4480 |
| + shared row-address factoring | 6696 | 6864 | 4472 |
| + load-only metadata hoist | 6696 | 6864 | 4472 |
| + two-row scatter (final candidate) | **6704** | **6872** | **4472** |

Slots include NOP padding. Final non-NOP totals are 6539/6708 versus 6584/6763
for this audit's baseline. Final Prefill BRA counts are 439/449 versus 446/456.
BSSY and BSYNC remain 213/218 each; there is no further reconvergence reduction
beyond the earlier row-guard change. All three SwapAB functions retain identical
128-bit instruction encodings, including control bits.

All six functions retain 168 registers and identical resource reports. Prefill
has zero stack/spill; masked nonswap, shared Decode and partial wave retain
8-byte stack and no spills. Independent Decode retains its existing 16-byte
stack and 8/8-byte spill stores/loads. There are **no added spills**.

### Load/consumer ordering in the final candidate

| Prefill128 metadata | Previous load → first consumer PC | Final load → first consumer PC |
| --- | --- | --- |
| rank | 9020 → 9090 | 7af0 → 8d00 |
| token | 9040 → 90c0 | 7b00 → 8d60 |
| slot | 9050 → 90b0 | 7b10 → 8d70 |

Linear code distance increases from 7/8/6 slots to 289/294/294 slots. These are
**static PC distances, not executed instructions or cycles**: partial-row guards
skip packing work, and scoreboard stalls are not measured. Prefill140 has the
same distances. In final Prefill128, the first two LDS.128 are at 8f40/8f50;
the first two ST.E.128 are at 8fd0/8fe0. Thus the second row's load is visibly
issued before the first row's store. The second load and store remain predicated.

## Memory bottleneck boundary

Existing full-kernel timelines identify substantial FC2 epilogue and combine
work, but have no per-PC stall counters. Exposed metadata-to-address and
LDS-to-store dependencies are plausible targets; SASS alone cannot prove their
runtime contribution. No reduction in HBM or NVLink payload traffic is claimed.
All global/shared payload load/store opcode counts, BF16 conversions, warp/CTA
barriers and memory fences are unchanged. The original 8×4096×2-byte routed
BF16 payload per token is preserved.

Combine output-TMA waits already follow useful reduction work for another chunk;
they were left alone because their latency may already be hidden. Existing TMA
input pipelining was also left unchanged. Duplicate metadata lane accesses do
not imply duplicate memory sectors, so reducing their lane count is not claimed
as a bandwidth optimization. The previous fragment-layout alternatives lost in
EP4 measurements and are not included.

## Validation and evidence

- Eight source variants were CPU-compiled/disassembled, including rejected
  early-address and zero-initialized-pair forms and both final combinations.
- Three new patches apply sequentially and exactly reproduce the compiled
  headers. The output-probe extractor verifies unchanged code outside the FC2
  output/combine region.
- Source-extracted packing bodies pass 7,409,760 symbolic BF16-cell checks,
  all valid_m=0..128, widths 64/128/256, both warpgroups, with sentinel padding.
- Source-extracted metadata/scatter bodies pass 16,908,288 destination/payload
  checks per variant: baseline, row-address, row+metadata, and final pair bundle.
  These cover every valid_m=0..128 (including odd tails), 16 hidden tiles and four
  rank mappings, checking invalid metadata/shared reads and lane ownership.
  CPU simulations do not validate GPU races, floating-point execution or NVLink.
- All recovered cubin/PTX/header hashes match remote checksums. Raw evidence is
  `outputs/qwen-fc2-redesign-20261024/cpu-fc2-dependency-check/`, including
  prepare/compile scripts, full and compact SASS, PTX, cubins, resource logs,
  host harnesses and SHA256SUMS.

```text
Previous baseline: 36861ed7812629223283b4e5a5b290d8df6140d06b9e8c3a06bb19e378f8843e
Row address only: bb7eacbc278ee86c0fd80eaf7ccd12be7865df0b5ce4c92e0a7323764e5c9815
+ early metadata: caca68715ec9e380bee0df0800aa90f0c4f39510d92e9e1288afdcded5af4452
+ paired scatter: 7997786dc2d5b1f12e54d934db6a379103d8dc36fc6cd48b93eed408c638daf0
```

## Authorized serving measurement

Use the existing `[cluster:firmus] [nodes:2]` push route exactly once, without
`[kernel-bench]` or `[profile]`. The image/host ABI/dependencies are unchanged;
the modified DeepGEMM headers are freshly JIT-compiled in the new runtime's
cache. CPU full-kernel compilation has already checked the modified source.
The existing EP4 numeric gate runs before model startup, followed by official
60/120/30 AIPerf at concurrency 240 and the standard eligibility parser.

The 17-minute allocation cannot hold two complete model startups: baseline run
40071892077 reported READY after 478 seconds and took 16m37s including its
separate profile phase. Use its uninstrumented 8854.80 output tok/s score as a
historical reference, checking resolved kernel selections and serving settings.
A single new run provides a provisional comparison, not a noise-controlled
causal speedup claim. The 9500+ target is not established by static code size.
