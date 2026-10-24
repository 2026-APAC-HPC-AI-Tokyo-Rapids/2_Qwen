# MegaMoE FC2 fragment output and combine redesign

Target: improve on the 8854.80 good output tok/s serving run 40071892077 toward 9500+. This is an experimental output-path change, not a measured serving improvement. Work starts from cd8863d and the exact 40064516380 image plus its six serving DeepGEMM overlays. Main and the shared checkout are unchanged.

The FC2 non-SwapAB epilogue currently casts FP32 to BF16, writes a swizzled shared tile, reloads vectors and scatters them into the token owner's top-k slots. Each of the eight expert outputs pays for this transpose. The candidate instead stores register-native uint4 fragments directly into the same symmetric allocation. Within every 32 hidden columns, the two 2-bit BF16-pair coordinates are swapped. The consumer performs exactly the same FP32 FMAs in ascending top-k order on these permuted elements, casts once to BF16, then transposes four packed pairs across each lane quad before its existing output store.

No extra memory, atomic operations, publication counters, rank barriers, readiness protocol, quantization, or reduction reordering is introduced. The total BF16 payload remains unchanged. The intended saving is shared-memory round trips and instructions; NVLink bandwidth is still a possible limit. The existing grid/NVLink release and cleanup protocol remains in place. Every Qwen rank uses the same format regardless of its GEMM geometry, including idle nonswap ranks. SwapAB currently writes the new column permutation through its existing shared-memory transpose, preserving its cross-warpgroup synchronization.

`patches/deep-gemm-bench/fc2-fragment-combine/fc2-fragment-combine.patch` applies after the configured serving overlays. The initial Genkai commit stored it under `patches/deep-gemm-experiments/`; the follow-up one-node Firmus campaign activates it alongside the producer-transpose ablation. The complete kernel must be newly compiled/JITed for this header; the host ABI and container dependencies do not change.

## Validation and resource policy

Per the user's instruction, GPU development tests run on Genkai with **one GPU**. Additional peer-communication testing, if justified, is limited to **one Firmus node**. Multiple nodes are reserved for the final official serving measurement. Jobs go through GitHub Actions and PJM/Slurm; login nodes only compile or inspect.

- CPU ownership tests cover 64/128/256-column tiles, partial rows, the full 4096-channel permutation, and the simultaneous shuffle network.
- `prepare-fc2-output.sh` compiles the exact extracted FC2 epilogue and combine bodies plus both full kernels for six geometries. Extraction rejects modifications outside its covered sections.
- The one-GPU executable checks four producer geometries, Prefill paired/Decode parallel combine, 12 token counts, masked/all-padding slots and changed-weight CUDA Graph replays against both the baseline and an independent scalar FP32-FMA reference. It interleaves A/B timing, and the runner repeats three times.
- Synthetic FP32 accumulators replace GEMM in the output-only probe. This can reject regressions cheaply but does not establish full-kernel speed, NVLink correctness/performance or serving throughput. Full-kernel compiler resource logs are separate from the synthetic probe.
- A successful candidate still needs real routing, mixed-rank geometry, full MegaMoE numerical gate, and final official workload/parser validation before adoption.

## CPU results before GPU submission

CUDA 12.9.1 / GCC 13 on Genkai compiled Prefill128, Prefill140, independent/shared
Decode, nonswap masked geometry and partial-wave SwapAB for both headers. Resource
usage is identical in all six pairs: 168 registers; no spills except the existing
8-byte store/load spill in independent Decode. Prefill has zero stack/spills.
The extracted output/combine probe compiles without spills in all four geometries.
Four layout tests pass. Existing Firmus kernel-bench fixtures pass using GNU tools
on macOS (the initial BSD `stat` invocation is incompatible with Linux fixtures).

One-GPU Actions run: [40105137999](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40105137999),
commit `c647097`. Results remain pending at submission; no Firmus job was launched.


## One-node Firmus campaign

The user selected Firmus one node after Genkai's shared one-GPU pool filled, and
requested packing useful tests into the allocation. The Genkai job remains
queued; it has not been cancelled. No multi-node serving run is launched here.

The second candidate, `fc2-producer-transpose`, uses the same register-packed
BF16 vectors but transposes them across each quad *before* scattering. Four
shuffles restore natural channel order; SwapAB and combine are unchanged. A
ballot before the partial-row branch supplies the shuffle participation mask,
so inactive row quads are excluded without excluding any source lane in a live
quad. This distinguishes shared-memory staging removal from moving the transpose
to the consumer. Both candidates preserve BF16 rounding and FP32 FMA order and
must match the control's output hashes exactly.

One allocation requests four H200 GPUs on one node, with the existing 17-minute
limit. It contains:

1. A single-GPU native output probe for both candidates, three processes each;
   192 independent-reference/bitwise checks per process (1,152 total). It builds
   in the serving container and has a 180-second overall limit, 30 seconds per
   process. Results distinguish local synthetic output timing from full MegaMoE.
2. Current serving-overlay full MegaMoE bench and separate timeline.
3. Fragment-combine EP4 numerical gate, full bench, and separate timeline.
4. Producer-transpose EP4 numerical gate and full bench.
5. Current implementation numerical gate and final repeat bench to expose drift.

The nine EP4 steps plus the bounded native step fit within the existing twelve
75-second provisioning budget units: the native step reserves three units.
The full benches already include balanced/local and observed routing from
layers 43, 62 and 7, with four replicas; gates include mixed geometries, idle
ranks, hot experts, all-idle/live transitions and changing CUDA Graph inputs.
Previously measured bulk-scatter/WG-padding candidates are archived outside the
active campaign, avoiding repeated work from run 40095847738. A successful kernel
campaign still does not establish 9500+ serving throughput.

Local campaign fixtures: 66 passed, 3 skipped (GNU tools on macOS). Producer
ablation compiles in all six full-kernel geometries and the extracted probe.
CUDA 12.9 reports extra Prefill spills for the producer ablation (24-byte store,
32-byte load, 16-byte stack); the fragment-combine candidate retains baseline
resources. This is a known cost to measure, not an assumed winning variant.
The candidate-specific timeline patch only resolves the adjacent declaration
insertion; its instrumentation is otherwise the stock timeline patch.

## First Firmus result: 40107915306

[Run 40107915306](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40107915306)
used commit `e5608e8`, Slurm 25954, one H200 node / four GPUs. All nine EP4
steps succeeded. Both candidates and the final control matched the image's
output hashes in all 19 cases; candidate A's instrumented results also passed.
The overall run failed only because the additional native probe called `nvcc`
through a PATH that did not contain the container compiler (`exit 127`). It never
compiled or executed its single-GPU checks. Use the pinned image's absolute
compiler and Python paths, as the existing EP4 scripts do; retain Slurm's GPU
visibility mask and select logical device zero in the probe.

Raw data were archived with checksums at
`archive-host:/srv/qwen-artifacts/40107915306/1/` and recovered
locally with all SHA256SUMS entries verified. GitHub artifacts are deleted by the
existing workflow only after that archive succeeds. The overall red status does
not invalidate the completed EP4 measurements.

| Case | image ms | fragment/combine ms | producer transpose ms | final control ms |
| --- | ---: | ---: | ---: | ---: |
| decode-balanced-160 | 0.1996 | 0.2006 | 0.2005 | 0.2001 |
| prefill-balanced-4096 | 1.5236 | 1.6179 | 2.0866 | 1.5244 |
| prefill-local-4096 | 1.3020 | 1.2892 | 1.7283 | 1.3051 |
| prefill-real43-4096 | 1.5521 | 1.6657 | 2.1481 | 1.5565 |
| prefill-real43-replica4-4096 | 1.4249 | 1.4792 | 2.0026 | 1.4326 |
| prefill-real62-replica4-4096 | 1.4318 | 1.4728 | 2.0106 | 1.4389 |
| prefill-real7-replica4-4096 | 1.4419 | 1.4876 | 2.0469 | 1.4405 |

This run provides no adoption evidence. A regresses real routing by 2.9–7.3%;
B by 37.9–42.0%. Decode changes are within about 0.7%, with control drift of a
similar scale. A's local-only case is slightly faster, while peer-scatter cases
are slower. The balanced Prefill timeline has comparable median SM clocks
(1.611 vs 1.621 GHz): A's WG0 FC2 epilogue median increases from 235–251 us to
302–309 us across ranks, while combine stays about 86–88 us. Removing shared
staging did not save the critical peer-scatter time. Smaller per-instruction
coalescing groups in the register-native scatter are a hypothesis for this
increase, not a measured transaction-level conclusion. Local timeline clocks
are less comparable and must not be used as a clean speedup measurement.

A repeat with only the probe launcher corrected will complete the missing native
checks and test reproducibility under the same bounded one-node allocation. No
serving throughput or 9500+ tok/s claim follows from these kernel measurements.
