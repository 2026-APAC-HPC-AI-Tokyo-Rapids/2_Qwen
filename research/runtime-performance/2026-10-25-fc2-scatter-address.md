# FC2 scatter address hoisting: CPU code-generation evidence

The Prefill scatter address calculation is now hoisted out of its 16-row loop.
The independent patch is
`patches/deep-gemm-overlay/fc2-scatter-address.patch`.
It applies to the serving baseline after its six overlays, **not** on top of the
rejected fragment/combine or producer-transpose candidates. It is not activated
in a campaign. No push, Actions dispatch, scheduler submission or GPU execution
was performed for this change, as explicitly requested by the user.

## Implementation and correctness boundary

Each lane loads metadata for its own row, as before, and computes that row's
mapped destination once. A 64-bit shuffle broadcasts the selected row base;
the receiving lane then adds its own 16-byte vector offset. The tile's `n_idx`
offset is included before shuffling; the lane offset must be added afterwards.

The pinned `SymBuffer::map` only adds `offsets[rank_idx]`; it has no synchronization
or publication side effects. Invalid metadata remains `{0, 0, 0}`, so padding
lanes only calculate an in-bounds rank-0 table lookup. Their calculated addresses
are never selected by a live row. An all-padding warp still performs this address
setup before its loop exits; this is a potential cost for sparse tiles.

For row j and receiving lane l, both paths address
`combine_base + rank_offset[j] + topk[j]*4224*8192 + token[j]*8192 + n_idx*2 + l*16`.
Every live row j has live source lane j in its warp; all lanes execute the same
row loop and full-mask shuffles. Shared-memory swizzle, 512-byte warp row layout,
BF16 payload, vector stores, combine arithmetic and barrier protocol are unchanged.

An initial guarded-address version introduced one extra `BSSY/BSYNC` pair in the
full kernel. The final version retains the original predicated metadata loads
and performs safe address arithmetic outside that lane-dependent condition.
The extra reconvergence pair is absent in the final SASS.

CPU validation executed the extracted candidate prelude and row loop with a
warp-shuffle simulator and captured destination addresses. All 16,908,288 vector
address checks passed: every `valid_m` from 0 through 128, both warpgroups, all
warps/lanes, all 16 hidden-dimension tiles, four rank patterns, negative/positive
peer offsets and addresses crossing 32-bit boundaries. No padding metadata was
read. This checks addressing; it is **not** a GPU numerical or NVLink test.
The existing output-probe extractor also confirmed that the candidate changes
no code outside the extracted FC2 output/combine bodies.

## Compiler and provenance

CPU-only compilation ran on Genkai's login node using **CUDA 13.0.88 extracted
from the pinned Firmus image**, not the earlier CUDA 12.9 installation. GCC was
13.3.1. Image `40064516380.sqsh` had the configured size 12,273,000,448 bytes and
sidecar SHA-256 `112ff6f0831d21e703b686df96bf950501d51a75c67950ea6a707b80197638bf`.
The extracted header matched the image build manifest, and applying the six
configured overlays reproduced the saved baseline byte-for-byte.

| Header | SHA-256 |
| --- | --- |
| Baseline | `c40d4c141aea45bb17a864ae8d7088a97363f12d0188b2584e558203aacb11ad` |
| Final address hoist | `2a2227f91e692d23b1f9b6f8f2ec454d378da533122e3fcc50191a485bb28b8e` |

Both builds use `kernels/probes/fc2_output_instantiate.cu`: the complete MegaMoE
entry point instantiated for six geometries, including EP4 Prefill128/140. They
are not just an isolated output probe. Identical flags are C++20, `-O3`,
`--expt-relaxed-constexpr`, `--expt-extended-lambda`, `-lineinfo`,
`--ptxas-options=-v,-warn-spills`, and `-gencode arch=compute_90a,code=sm_90a`.
This is the full-kernel compile harness, not a recovered serving JIT cubin.

## Generated code

Counts below are static instructions, not execution-time measurements. PTX
contains two 32-bit shuffles for the new 64-bit address. SASS confirms that the
rank-offset load and wide slot/token arithmetic have moved out of the row loop.

| Quantity | Baseline | Address hoist |
| --- | ---: | ---: |
| Scatter SHFL instructions for 16 full rows | 48 | 32 |
| Scatter indexed rank-offset `LDC.64` instructions | 16 | 1 |
| Scatter 128-bit vector stores | 16 | 16 |
| Prefill128 whole-kernel instruction slots, including NOP padding | 7008 | 6936 |
| Prefill140 whole-kernel instruction slots, including NOP padding | 7168 | 7096 |
| Prefill128 non-NOP instructions | 6845 | 6771 |
| Prefill140 non-NOP instructions | 7003 | 6928 |
| Prefill128 whole-kernel `IMAD.WIDE.U32` | 81 | 50 |
| Prefill140 whole-kernel `IMAD.WIDE.U32` | 92 | 61 |

Each Prefill function is 1,152 bytes smaller in instruction storage. All six
geometries retain 168 registers and their original stack/spill resources.
Prefill128/140 have zero stack and zero spills. Independent Decode retains its
pre-existing 8-byte spill stores/loads and 16-byte stack; the other three
non-Prefill shapes retain 8-byte stacks without spills. All four non-Prefill
functions have **identical 128-bit instruction encodings**, including controls.

Representative second-row dependency chain in Prefill128:

```text
baseline: SHFL(rank/token/topk) -> rank-table LDC.64 + two IMAD.WIDE
          -> mapped-address additions -> ST.E.128
candidate: SHFL(address high/low) -> lane-offset add/carry -> ST.E.128
```

The baseline second-row `LDC.64` is at SASS PC `0x9e90`; the final candidate's
one-time lookup is at `0x9c70`, before the scatter loop. Its second-row shuffles
are at `0x9e00/0x9e50`, followed by additions and the store at `0x9ea0`.

This removes 15 repeated load-to-store dependencies per full 16-row scatter.
It does **not** establish that measured stall cycles fall. The new lookup has
lane-varying rank indices, while each old lookup broadcasts a single rank;
one instruction therefore does not imply one constant-memory transaction.
Startup latency, sparse-row overhead and scheduling need GPU measurement.
Explicit synchronization/wait instruction counts do not fall: for example,
Prefill128 retains 243 `BSYNC`, 22 `BAR.SYNC.DEFER_BLOCKING`, two `DEPBAR.LE`,
and two `WARPGROUP.DEPBAR.LE`. TMA phase checks, memory fences and WGMMA/FMA
instruction counts are also unchanged. No runtime or tok/s improvement is claimed.

The working performance estimate discussed with the user is approximately 0–1%
of a 1.4 ms MegaMoE invocation (0–14 us), with roughly 2% an optimistic scenario.
This is an unmeasured engineering estimate, not a bound; regression remains
possible. It must not be converted directly into a serving-throughput gain.

## Other small changes worth investigating

1. **Hoist row-validity guards out of the BF16 packing loop.** The current
   non-SwapAB loop repeats `if (valid_r0)` / `if (valid_r1)` for each chunk pair.
   Its Prefill SASS still contains 32 `BSYNC` instructions in this packing
   region. Process all r0 chunk pairs under one guard and all r1 pairs under
   another. This could remove repeated branches/reconvergence and expose
   invariant swizzle/address calculations. Preserve rounding and store mapping;
   inspect accumulator live ranges/spills before considering a GPU comparison.
   The subsequent CPU-only implementation and code-generation comparison are in
   [the BF16 row-guard report](2026-10-25-fc2-bf16-row-guards.md): packing-region
   reconvergence falls from 32 to 2 with no additional spills. GPU speed remains
   unmeasured.
2. **Reduce scatter unrolling.** CPU-only scratch ablations changed just the
   final candidate's pragma. All six shapes compiled; Prefill remains at 168
   registers without spills. The four untouched shapes retain identical code.
   These ablations are not included in the submitted patch:

   | Scatter unroll | Prefill128 instruction slots | Prefill140 instruction slots |
   | --- | ---: | ---: |
   | 16 (final candidate) | 6936 | 7096 |
   | 4 | 6824 | 6984 |
   | 1 | 6760 | 6928 |

   Smaller static code does not mean fewer executed instructions. The rolled
   versions add counter updates, loop-back branches and variable row/swizzle
   calculations; their smaller static store counts are repeatedly executed.
   Keep this as a separate instruction-cache versus loop-overhead comparison.
3. **Decode metadata/address preloading.** Its eight-row scatter loads metadata
   inside the loop. Preloading row addresses could remove repeated dependent
   loads, but would add shuffles that Decode does not currently use. Measure the
   generated dependency chain and register lifetime independently of Prefill.
4. **Direct combine output stores.** The earlier CPU-only scratch already removed
   output shared staging/TMA-store waits without changing input TMA or FMA order.
   Its CUDA 12.9 resource check passed, but it has no GPU evidence. Avoid replacing
   the input pipeline simultaneously: that separate attempt introduced spills.

## Reproducing the inspection

Raw evidence is retained under
`outputs/qwen-fc2-redesign-20261024/cpu-address-check/`: PTX, cubins, disassemblies,
resource logs, include/compiler hashes, captured input headers, the CPU address
check source, and a checksummed manifest. `compile.sh` records the CPU-only
compilation/disassembly commands with no scheduler or GPU invocation.

From this worktree, regenerate instruction summaries with:

```sh
python3 scripts/inspect-fc2-scatter-codegen.py ../cpu-address-check/codegen/baseline/full.sass ../cpu-address-check/codegen/address-hoist/full.sass ../cpu-address-check/inspection
python3 ../cpu-address-check/validate-address-source.py
```

The inspection utility preserves both instruction words when comparing untouched
functions and emits compact source-annotated SASS. The patch remains inactive and
un-pushed. GPU numerical gates, real-routing timing and the final official serving
measurement remain outstanding; the existing user prohibition on new jobs remains
in force.
The owned Genkai CPU scratch directory was removed after recovering the artifacts
and matching every cubin/PTX checksum against the remote originals.
