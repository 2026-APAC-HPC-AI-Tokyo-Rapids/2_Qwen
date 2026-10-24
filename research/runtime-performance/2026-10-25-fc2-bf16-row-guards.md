# FC2 BF16 row-guard hoisting: generated-code comparison

The non-SwapAB BF16 packing path now finishes each row under a single validity
guard, instead of testing the same two row predicates inside every chunk-pair
iteration. The implementation uses an inlined `store_row(row, accum_offset)`
lambda, called with constant offsets 0 and 2. The source becomes 11 lines shorter.
It preserves each FP32 input pair, BF16 conversion, shared-memory swizzle and
destination. The existing warp synchronization remains after both rows.

Apply `patches/deep-gemm-overlay/fc2-bf16-row-guards.patch`
**after** `fc2-scatter-address.patch`, both after the six serving overlays.
The comparison here is address hoisting alone versus address hoisting plus row
guards; scatter unrolling, output layout and combine are unchanged. The patches
remain inactive. No push, workflow dispatch, scheduler job or GPU execution was
performed, per the user's explicit instruction.

## Full-kernel SASS results

Both variants were freshly compiled on Genkai's login node using the pinned
Firmus image's CUDA 13.0.88 and GCC 13.3.1, with identical C++20 / `-O3` /
`-lineinfo` / `compute_90a,sm_90a` flags. The six instantiations come from
`kernels/probes/fc2_output_instantiate.cu`, compiling the full MegaMoE kernel.
These are CPU compile-harness results, not recovered serving JIT cubins or timings.

| Metric | Address hoist only | + BF16 row guards | Reduction |
| --- | ---: | ---: | ---: |
| Prefill packing-region `BSYNC` | 32 | 2 | 30 |
| Prefill128 whole-function `BSSY` | 243 | 213 | 30 |
| Prefill128 whole-function `BSYNC` | 243 | 213 | 30 |
| Prefill128 whole-function `BRA` | 476 | 446 | 30 |
| Prefill128 instruction slots including NOPs | 6936 | 6752 | 184 |
| Prefill128 non-NOP instructions | 6771 | 6584 | 187 |
| Prefill140 whole-function `BSSY` / `BSYNC` (each) | 248 | 218 | 30 |
| Prefill140 whole-function `BRA` | 486 | 456 | 30 |
| Prefill140 instruction slots including NOPs | 7096 | 6928 | 168 |
| Prefill140 non-NOP instructions | 6928 | 6763 | 165 |
| Masked nonswap packing-region `BSYNC` | 8 | 2 | 6 |
| Masked nonswap instruction slots including NOPs | 4520 | 4480 | 40 |
| Masked nonswap non-NOP instructions | 4447 | 4411 | 36 |

The Prefill functions shrink by **2,944 bytes / 2,688 bytes** in this change.
Including the previous address hoist, the total reduction from the original
serving baseline is 4,096 bytes for Prefill128 and 3,840 bytes for Prefill140.
The remaining address-instruction reductions accompany moving the row guards;
they were not a separate address-layout change.

Both Prefill variants retain their original counts of packed BF16 conversions
(`F2FP.BF16.F32.PACK_AB`: 80), shared stores, scatter stores (`ST.E.128`: 16),
warp/CTA barriers, memory fences and TMA phase checks. The removed `BSYNC`s are
compiler-generated branch reconvergence, not the shared-memory or communication
barriers required for correctness. Source line attribution can place a `BSSY`
outside the packing range, so the table reports whole-function `BSSY` totals
and source-annotated packing-region `BSYNC`s separately.

All three SwapAB geometries—independent Decode, shared Decode and partial wave—
have identical 128-bit instruction encodings before and after, including control
bits. The masked nonswap shape also benefits from guard hoisting.

## Registers and spills

All six geometries retain **168 registers**. There is no new stack allocation or
spill in any geometry:

| Geometry | Stack bytes, before → after | Spill store/load bytes, before → after |
| --- | --- | --- |
| Prefill128 | 0 → 0 | 0/0 → 0/0 |
| Prefill140 | 0 → 0 | 0/0 → 0/0 |
| Masked nonswap | 8 → 8 | 0/0 → 0/0 |
| Shared Decode | 8 → 8 | 0/0 → 0/0 |
| Partial wave | 8 → 8 | 0/0 → 0/0 |
| Independent Decode | 16 → 16 | 8/8 → 8/8 |

The independent Decode spill already exists in the baseline. Compiler resource
logs and the unchanged SwapAB instruction encodings independently confirm this.

## Validation and evidence

- Both patches apply cleanly in sequence and reconstruct the compiled header.
- The existing output-probe extractor confirms no edits outside its covered FC2
  output/combine bodies.
- CPU execution of the actual extracted old/new packing bodies passed 7,409,760
  BF16-cell comparisons including sentinel guards. This covers every valid-row
  count 0–128, widths 64/128/256, both warpgroups and all lanes. Distinct symbolic
  labels represent the two inputs to each BF16 conversion; an independent
  element-wise WGMMA mapping checks their destination and padding. This tests
  ownership and source pairs, **not** GPU floating-point execution or races.
- Recovered cubins and PTX match their remote checksums.

Header SHA-256 values:

```text
address only: 2a2227f91e692d23b1f9b6f8f2ec454d378da533122e3fcc50191a485bb28b8e
+ row guards: 36861ed7812629223283b4e5a5b290d8df6140d06b9e8c3a06bb19e378f8843e
```

Raw artifacts live in `outputs/qwen-fc2-redesign-20261024/cpu-bf16-guard-check/`:
input headers, PTX, cubins, full and compact SASS, resource logs, compiler/include
hashes, CPU validation source, and `SHA256SUMS`. `prepare.sh` and `compile.sh`
record the CPU-only extraction/build commands. Re-run the local inspection with:

```sh
python3 scripts/inspect-fc2-scatter-codegen.py ../cpu-bf16-guard-check/codegen/address-hoist/full.sass ../cpu-bf16-guard-check/codegen/bf16-guards/full.sass ../cpu-bf16-guard-check/inspection
python3 ../cpu-bf16-guard-check/check-pack-source.py
```

This establishes fewer static instructions and reconvergence operations without
additional spills. It does not yet establish lower stall time, faster MegaMoE or
higher serving throughput; GPU validation remains intentionally unperformed.
