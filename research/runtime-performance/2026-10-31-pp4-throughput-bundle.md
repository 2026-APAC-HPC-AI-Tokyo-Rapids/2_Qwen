# PP4 throughput bundle: 10k / 10.5k / 11k

User instruction: combine meaningful improvements and run them together. This
authorizes a combined serving trial; it does not make a predicted score a result.

Reference: 41005872470 = 9832.74; repeat 41020974704 = 9828.95. Parent
01e68a6 / 41073991312 already pins the fixed FA3 pipe, gates diagonal trimming,
uses gigatoken and removes settled selection work. Its score was pending when
this bundle was prepared. Keep 16 H200s, P3×PP4/EP1, D1×TP4/DP4/EP4, c232,
Decode cap 48/rank, EAGLE3 steps3/width4 and the official workload/parser.

The source, shapes, precision, work accounting and unknown measurements are in
`kernel-theory-model.json:pp4_throughput_bundle_20261031`. Its historical EP4
profile tables are not a current PP4 profile.

## Added together

| Change | Removed work | Required evidence |
|---|---|---|
| Sparse min-new-token penalty | Per-step 48×151936 FP32 accumulated mask and vocabulary-wide add to 192 logits rows; touch only active stop IDs | CPU dense oracle, ragged merge/filter, frozen overlap snapshots, mixed-penalty fallback; four-GPU actual Triton parity, changed-input graph replay and timing |
| FA3 draft-extend graph | Eager launches of the existing draft model operations | Every captured batch checks exact logits/hidden parity against eager and two replays; actual model startup, sanity generation, live serving and score |
| FC1 split unroll 8→16 | Four drains become two per 32-K-block tile; FC2 and numerical order unchanged | Native host and layout checks, zero-spill CUDA compile, exactness on four routings and >0.5% aggregate saving across 12 GPUs, then 600-call endurance |

The sparse path is only for a CUDA overlap batch with min-new-tokens as its sole
prepared penalty and at most 256 stop IDs/request. All other cases keep the
existing dense path. The snapshot owns a new boolean mask, so advancing or
filtering the scheduler cannot change an in-flight worker's decision. Existing
speculative penalty semantics and model precision stay unchanged.

The graph change connects the existing FA3 DRAFT_EXTEND_V2 implementation to the
worker's supported-backend list. It preserves width, DP padding and capture
guards. Capture parity is deliberately a limited check of capture inputs, not
proof of every live page-table/padding configuration.

Unroll16 is supported by the existing mainloop experiment 40993048750 (~1% FC1
gain). This is not the rejected ldmatrix, reciprocal, stagger or U32 candidate.
Only unroll16 and the served split-u8 control are measured. A missing result,
numerical failure, insufficient saving or endurance failure keeps split-u8.
The FA3 and MegaMoE gates run sequentially on each worker to avoid interfering
with timing; the three Prefill workers run in parallel. The sparse gate uses the
otherwise idle four Decode GPUs. No old candidate sweep is restarted.

## Validation and results

CPU checks passed: Torch 2.11.0 / NumPy 2.2.6, PP4 model output and EAGLE captures,
32 tensor cases, 72 relay/KV/release cases, 96 scheduler iterations (40 requests),
18 sparse penalty parity cases plus snapshot/filter/merge/mixed-penalty checks.
Native host/layout: 276 cases; CUDA 13.0.88 (image toolchain): 12 instantiated kernels, zero spills.
Exact source/header/patch hashes are in the JSON ledger. The broad Linux suite
found a missing uv PATH and the stale retired-candidate inventory; both were
fixed and the affected/changed tests passed (79 passed, 1 skipped). Full CI is
required before build/allocation. GPU gates and the serving score remain pending.
Do not merge into main before superiority is measured.

10k requires +1.70% from 9832.74; 10.5k +6.79%; 11k +11.87%. The three added
changes have no measured combined gain yet. They are not evidence that all three
targets are attainable, and no SLA/parser/workload changes are used to claim it.
