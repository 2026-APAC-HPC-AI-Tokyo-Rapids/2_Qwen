# MegaMoE FC2 fragment output and combine redesign

Target: improve on the 8854.80 good output tok/s serving run 40071892077 toward 9500+. This is an experimental output-path change, not a measured serving improvement. Work starts from cd8863d and the exact 40064516380 image plus its six serving DeepGEMM overlays. Main and the shared checkout are unchanged.

The FC2 non-SwapAB epilogue currently casts FP32 to BF16, writes a swizzled shared tile, reloads vectors and scatters them into the token owner's top-k slots. Each of the eight expert outputs pays for this transpose. The candidate instead stores register-native uint4 fragments directly into the same symmetric allocation. Within every 32 hidden columns, the two 2-bit BF16-pair coordinates are swapped. The consumer performs exactly the same FP32 FMAs in ascending top-k order on these permuted elements, casts once to BF16, then transposes four packed pairs across each lane quad before its existing output store.

No extra memory, atomic operations, publication counters, rank barriers, readiness protocol, quantization, or reduction reordering is introduced. The total BF16 payload remains unchanged. The intended saving is shared-memory round trips and instructions; NVLink bandwidth is still a possible limit. The existing grid/NVLink release and cleanup protocol remains in place. Every Qwen rank uses the same format regardless of its GEMM geometry, including idle nonswap ranks. SwapAB currently writes the new column permutation through its existing shared-memory transpose, preserving its cross-warpgroup synchronization.

`patches/deep-gemm-experiments/fc2-fragment-combine.patch` applies after the configured serving overlays. It is outside the active Firmus campaign so merely adding it does not launch or expand a multi-GPU experiment. The complete kernel must be newly compiled/JITed for this header; the host ABI and container dependencies do not change.

## Validation and resource policy

Per the user's instruction, GPU development tests run on Genkai with **one GPU**. Additional peer-communication testing, if justified, is limited to **one Firmus node**. Multiple nodes are reserved for the final official serving measurement. Jobs go through GitHub Actions and PJM/Slurm; login nodes only compile or inspect.

- CPU ownership tests cover 64/128/256-column tiles, partial rows, the full 4096-channel permutation, and the simultaneous shuffle network.
- `prepare-fc2-output.sh` compiles the exact extracted FC2 epilogue and combine bodies plus both full kernels for six geometries. Extraction rejects modifications outside its covered sections.
- The one-GPU executable checks four producer geometries, Prefill paired/Decode parallel combine, 12 token counts, masked/all-padding slots and changed-weight CUDA Graph replays against both the baseline and an independent scalar FP32-FMA reference. It interleaves A/B timing, and the runner repeats three times.
- Synthetic FP32 accumulators replace GEMM in the output-only probe. This can reject regressions cheaply but does not establish full-kernel speed, NVLink correctness/performance or serving throughput. Full-kernel compiler resource logs are separate from the synthetic probe.
- A successful candidate still needs real routing, mixed-rank geometry, full MegaMoE numerical gate, and final official workload/parser validation before adoption.
