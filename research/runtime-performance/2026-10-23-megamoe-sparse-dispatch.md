# Decode dispatch stage 1: sparse CTA/expert reservations

`decode-sparse-dispatch` implements the first, local part of design A. It removes zero-count reservations from SwapAB Decode's histogram. It does not implement a separate prepare kernel, remove a barrier, or establish a latency improvement. In particular, the earlier 19 → 5–8 µs dispatch target is not a result of this change.

## Source and execution path

The control is image `39899235870`, header SHA256 `ff7a58cca00aa42a90386d12b0437b5d8caf35eb0ad082a8a1a0b4b74001a2a1`, plus the five `deep_gemm_overlay_patches` in `configs/firmus-framework.json` at `b165d1d`. Applying those patches in their listed order produces control header SHA256 `77ee7caea6e5736803b6034f83575f9a846d30c54ce4970ced7c65450a9be497`. The candidate then applies `patches/deep-gemm-bench/decode-sparse-dispatch/sm90-decode-sparse-dispatch.patch`, producing `360d066b26f472acb2747d6ddf2c4248a021bdcf44ab5122f84a617d2a2025e8`.

The patch lives in the existing kernel-bench candidate directory. No `timeline` marker is included: the existing instrumentation patch does not apply to the canonical baseline plus this candidate, and requires a separate rebase before profiling. The shared kernel-bench provisioning update must apply the current five overlays to both control and candidate before applying this patch. Benchmarking against the bare image would use an obsolete control. A fresh JIT build is required; the image supplies the unchanged host ABI and toolchain. No host, workspace layout, or serving config changes are needed.

## Implementation and invariants

The old reservation loop executes one global 64-bit atomic per CTA/expert, including zero counts: 132 × 128 = 16,896 per rank. Each increment has a high word of 1 and a low word containing the local count. The low return word allocates the CTA's source-index segment.

For `kSwapABActive`, the candidate reserves only nonempty CTA/expert pairs and adds only the low-word count. A skipped pair has no source entries to write and never uses its offset. Every nonempty pair gets exactly the same low-word reservation as before under the same interleaving. At 128 tokens and Top-8 there are at most 1,024 such pairs, and zero for all-padding input. Histogram loads, source writes, synchronization, token pull and GEMM still cost time; the atomic count ratio is not a runtime speedup estimate.

After the unchanged grid synchronization, SM0 replaces the local status's high word with `kNumSMs` before remote publication. This preserves the existing scheduler ABI, whose `fetch_expert_recv_count()` waits for high word `kNumSMs * kNumRanks`. The replacement is a register value used by the existing publisher; the local counter itself does not need a new write.

The ordering argument follows the actual image headers:

1. Histogram and offset processing retain their dispatch-thread barriers. Source-index writes retain their position before `comm::grid_sync`.
2. `comm::grid_sync` performs the role's CTA synchronization, then a release atomic and acquire spin on its dedicated grid-sync counter, then another CTA synchronization. SM0's send-count load and high-word normalization remain after this completion point.
3. The only reader of `get_expert_send_count_ptr()` after reservations is this SM0 publisher. Other accesses are the final zeroing stores. The scheduler polls the separate receive-count sum, so it cannot observe the new raw local high word.
4. `comm::nvlink_barrier` uses `get_nvl_barrier_counter_ptr()` and `get_nvl_barrier_signal_ptr()`, plus grid sync; it does not use expert-send counts. Its existing system release/acquire signals and all call sites are unchanged. The publication atomic, remote per-source count store, token/SF pull and L1 arrival notifications are unchanged.
5. Each rank publishes exactly `kNumSMs` in the high word for every expert, even with zero tokens or entirely `-1` padding. Old non-SwapAB ranks publish the same value through their original loop. Any combination of local geometries therefore uses the same wire protocol. This avoids the geometry-dependent barrier divergence documented in the [October 18 tail experiment](2026-10-18-megamoe-decode-tail.md).
6. Source-index lists, pool mapping, all FP instructions and top-k combination order are unchanged. No epoch, persistent new counter, or allocation is introduced. The existing post-GEMM synchronization, workspace zeroing and final rank barrier remain, so graph replay uses the same cleanup contract.

These are source invariants and CPU model checks. They do not replace GPU race, graph replay, numerical or performance gates.

## CPU validation

`scripts/check-megamoe-sparse-dispatch.py BASELINE_HEADER CANDIDATE_HEADER` passed 210 reservation cases. The model covers 0–4,096 tokens, partial warp/CTA boundaries, padded entries, balanced/random/concentrated routes, duplicate entries as an arithmetic stress case, CTA grid-stride traversal, and randomized reservation/write ordering. Candidate and control have identical published counts and source-index lists. It also checks 64 combinations of mixed geometry, idle ranks, all-idle ranks and changed-input replay/reset, including that no receiver reaches its ready high word before the final rank publishes. Header comparison checks that all code outside the reservation and SM0 normalization regions is byte-identical.

The candidate patch passes `git apply --check` on the reconstructed control outside any Git worktree. The checker pins the canonical baseline SHA256 before running its structural checks. Python AST parsing and `git diff --check` pass.

CPU-only compilation used the image's CUDA 13.0.88 `nvcc` on the Firmus login node. No GPU program, scheduler job or Actions run was started by this implementation task. Compile flags were `--cubin --std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -O3 --ptxas-options=-v,-warn-spills,--register-usage-level=10 -gencode arch=compute_90a,code=sm_90a`. The explicit instantiation was:

```cpp
sm90_fp8_mega_moe_impl<4224, 4096, 1536, 128, 8, 32, 64, 128, 128, 141312, 2260992, 8, 64, 64, 256, 132, 4, cute::numeric_limits<float>::infinity(), true, 0, false, false, true, false, SWAP>
```

| Geometry | Control / candidate registers | Spill stores / loads | Stack | Barriers | SASS |
| --- | --- | --- | --- | --- | --- |
| SwapAB Decode (`SWAP=true`) | 168 / 168 | 0 / 0 bytes, both | 8 bytes, both | 16, both | Changed |
| M64/N128 non-SwapAB (`SWAP=false`) | 168 / 168 | 0 / 0 bytes, both | 8 bytes, both | 3, both | Identical |
| M128/N256 Prefill, 2WG | 168 / 168 | 0 / 0 bytes, both | 0 bytes, both | 3, both | Identical |

Normalized `cuobjdump -sass` instruction-line SHA256 values: Decode control `8e0160674daa64256ef054a267c79e20ff44bbf08071cee8683f26678e1306a4`, Decode candidate `b02c9d31ebabb4e4a4c64c61d50d6b82ae83a4ec28852b1351e5cd64f1b6f96f`, both non-SwapAB variants `5842e79afae442ce63eab31f0f20ec8de117ae7547e6cc2f3bc790c05e754ad8`. Compiler warnings were existing unused variables; there were no spills or WGMMA serialization warnings. The Prefill 2WG variants also have identical SASS, SHA256 `f4dfa3b8bba194252571e838604df44885ea652d8bb06b15e8d4d074edcc3f86`. That instantiation changes BLOCK_M/N to 128/256, stages to 3, reuse/arrival to true and fullsync/SwapAB to false. Other Prefill geometries were not compiled.

## Baseline correction

Root review detected a hash mismatch in the initial validation record. The original reconstruction ran `git apply` from a subdirectory inside the experiment worktree. The first three overlays use `diff --git` headers and were silently skipped (exit status 0); the final two plain unified patches applied. `git apply --verbose` reproduced the skip and the old partial baseline SHA `0f18d32a53ebf57df22ae703381efde5ba5d96c75a4b33828ca807579bd04f55`. That baseline omitted the Prefill SF128, swizzle and 2WG epilogue overlays.

The results above replace the initial record: all five patches were applied outside any Git worktree, yielding canonical `77ee7cae…`; candidate application, all 274 CPU cases and six CUDA compilations were rerun. Decode SASS hashes happened to remain unchanged because the missing overlays affect other geometries; the non-SwapAB hash changed and has been corrected. The checker now rejects the partial baseline before testing. The earlier timeline apply check had likewise silently skipped; an actual canonical apply fails, so the timeline marker was removed. No GPU validation or runtime conclusion was based on the erroneous checks.

## Pending GPU acceptance

Run the existing Actions → scheduler kernel-bench route with fresh JIT caches and the complete current overlay on both variants. Required checks are mixed geometry including idle ranks, all-idle/padded ranks, changed-input graph replay, the existing numerical gate, then repeated interleaved timing with actual Decode routing and balanced routing. Keep timeline instrumentation separate from timing. A successful kernel bench must still be followed by the official serving workload, parser and artifact collection before adoption. No performance claim or merge to `main` is justified by these CPU results alone.
