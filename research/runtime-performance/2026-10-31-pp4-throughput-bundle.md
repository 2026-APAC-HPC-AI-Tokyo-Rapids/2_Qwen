# PP4 throughput bundle: 10k / 10.5k / 11k

User instruction: combine meaningful improvements and run them together. This
authorizes a combined serving trial; it does not make a predicted score a result.
Additional user constraint: include only working changes. Adoption requires GPU
numerical checks, continuous execution and ordinary serving; CPU compilation and
theoretical savings are insufficient. The validated ~9.8k configuration remains
available; inactive prototypes below are not part of the dispatched image.

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

## Forecast audit after dispatch

Run [41088950539](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41088950539),
commit `03e01f50d609c4d93e88f66bb1e2d5a060ef1966`, passed all three CI jobs.
The image build ran from 2026-10-30 16:31 to 17:05 UTC and succeeded. Image
SHA-256 is `44d16aaa4266ad187ce20590615a3def1eb75232815132bac4f825ff7f8ac781`,
size 12,298,129,408 bytes. The archive step reported an SSH/tar return failure,
but provisioning subsequently verified this image, its patch hashes and the CPU
gate successfully. No rebuild or rerun was needed. Slurm submission completed;
GPU gates and official score remain pending. The dispatch was too early to
describe this as a 10.5k/11k bundle.

Reused the existing PP4 closed-loop model, substituting the R4 measured Prefill
periods 115.3/115.0/118.7 ms, c232 and cap 48. R4's archived Decode log gives
37.29 ms for the `running * acceptance / throughput` proxy at mean running 43.71
(226 samples, 13:42:15–13:43:40 in the log's UTC+8 clock). This proxy mixes a
running-count snapshot and an interval throughput, so it is not a kernel timer.
The sensitivity uses 37.5 ms at 43.4, slopes 0.30/0.55/0.70 ms/request, and 12
simulation seeds. The old model's absolute R4 score is about 0.3% low; only its
change from baseline is added to the measured 9832.74. Seed variation does not
measure hardware or implementation uncertainty.

| Conditional case | Prefill period reduction | Decode step reduction | Model delta placed on R4 |
|---|---:|---:|---:|
| current bundle, low | 0.4 ms | 0.2 ms | 9875–9885 |
| current bundle, central | 1.0 ms | 0.3 ms | 9929–9932 |
| current bundle, upper | 1.5 ms | 0.45 ms | 9974–9978 |
| workload needed for about 10k | 2.3 ms | 0.5 ms | 10010–10043 |
| workload needed for about 10.5k | 8.5 ms | 2.0 ms | 10510–10616 |
| workload needed for about 11k | 13.5 ms | 3.5 ms | 11000–11126 |

All candidate reductions in this table are assumptions, not measured savings.
The last three rows are requirements for a future design, not available changes.
The working forecast is about 9.9k; 10k needs a favorable result. A user-facing
rough range of 9.9–10.1k is not a confidence interval. There is no basis for
claiming 10.5k or 11k from this dispatch.

## Structural candidates considered while the image builds

- Larger Prefill batches alone are already rejected by serving 39379601121:
  8064 tokens/rank became common, but per-token speed fell 1.2% and score fell 0.7%.
  PP2/EP2 with a proportionally larger global batch could preserve pipeline
  latency, but it adds expert communication and new kernel/rank support. It
  cannot be assigned a gain from padding arithmetic alone.
- Dense QKV/O are already near their measured compute floor. The desired
  7–12% total Prefill reduction must substantially change MegaMoE or attention.
- A separate FP32 promotion accumulator can remove the carried-scale
  recurrence in MegaMoE. At the current M128/N256 tile, two partial buffers plus
  the final accumulator already need 256 registers/thread before bookkeeping.
  M64/N256 with two N-split warpgroups fits a smaller register budget, but adds
  operand traffic and a cross-warpgroup FC1 quantization reduction. This is an
  inactive design, not a claim that the no-rescale timing floor is attainable.
  An isolated mainloop prototype compiled with CUDA 13.0.88 but emitted C7512
  WGMMA serialization and 932/1060 bytes of spill stores/loads (168 initial
  registers). It is rejected in this form and is not integrated or dispatched.
- Attention M192 with three consumer warpgroups can reduce logical K/V tile-load
  traffic by about one third and expose more independent warps. Native compilation
  rejects N128: register pressure serializes WGMMA (C7512) and produces 384/376
  bytes of spill stores/loads. N96 compiles without that serialization warning,
  at 40/28 bytes of spill stores/loads versus the control's 84/116. It performs
  twice as many online-softmax/output-rescale updates per query tile as N192.
  Changing the softmax block order requires a numerical reference gate; this is
  not the current diagonal-trim byte-exact candidate. No speedup is assigned
  before GPU timing and serving.

For the Attention prototype at L4001/Q64/KV4/D128, the tile census is:

| Quantity | served M128/N192 | rejected M192/N128 | inactive M192/N96 |
|---|---:|---:|---:|
| M tiles, all KV heads | 2004 | 1336 | 1336 |
| KV block visits | 21924 | 21632 | 28560 |
| K+V tile-load bytes | 1077608448 | 708837376 | 701890560 |
| QK+PV tile FLOPs | 275867762688 | 272193552384 | 269525975040 |
| scalar softmax elements | 538804224 | 531628032 | 526417920 |

Useful QK+PV work is 262340640768 FLOPs for both. Tile-load bytes are logical
L2/global-to-shared traffic, not measured HBM bytes; NVLink traffic is unchanged.
Native compile source is sgl-attn f89bc23 plus the exact bundle FA3 patch chain,
CUTLASS 57e3cfb and CUDA 13.0.88. The prototype uses a separate 8192 variant bit,
changes the fixed scheduler's shape guard and permits M192 in the FP8 epilogue.
Resource-only queries (no GPU execution) report 166,912 shared-memory bytes for
the control and 124,928 for M192/N96. ptxas reports 168 versus 128 initial launch
registers; these are not the dynamic math-warpgroup budgets (M192 uses 160).
The source clone is outside the serving worktree; the shipped candidate table
and image are unchanged. Kernel time, numerical parity and serving gain remain
unknown. The compile commands, logs, SASS and resource result are retained in
`outputs/pp4-throughput-bundle-20261031/fa3-geometry/` outside the repository.

## Terminal result and user-requested retry

Run 41088950539 ended in failure. Slurm 27712 ran for 3:18 and returned 1:0
before model startup or the official benchmark. The first fatal error is in
`logs/logs/sparse-penalty-gate.log`: rank 3 could not read
`~/.triton/cache/.../_apply_min_new_tokens_penalty.json` during Triton JIT
compilation. The archive SHA-256 for that log is
`07fa8aa248bf855ae2de6c95e5c064797b8d5013909368d8da686af9ae1d27f6`, verified
against the archive's SHA256SUMS. The four CPU checks passed; the GPU gate did
not finish. The parent shell propagated its exit code through `wait` with
`set -e`. Missing server-info files are a downstream symptom, not the first
failure. No official score was produced.

Both Prefill selectors retained their served controls: diagonal-trim FA3 lost
its timing gate, and split-unroll16 was below the 0.5% aggregate margin. These
are normal fallback decisions and did not fail the job.

The sparse gate omitted the node-local `TRITON_CACHE_DIR` already used by the
other selectors. Its HOME is bind-mounted from the run directory on shared
storage. The missing metadata is consistent with the prior shared-cache
replacement failure documented in `2026-10-22-claude-takeover-status.md`; the
exact filesystem race for this run has not been reproduced separately.

The user requested a fix and resubmission on 2026-10-31. Set this gate's cache
to `/tmp/qwen-sparse-penalty-${SLURM_JOB_ID}` inside its existing container.
Pin `configs/firmus-prefill-pp4.json` to the already built 41088950539 image,
using the hash, size and source commit verified in its archived sidecar and
runtime config. The previous tracked pin still referenced the round-4 image
because the original fresh-build job updated only its staged runtime config.
Reuse the bundle image without rebuilding. Workload, model, topology, kernels,
correctness gates and their failure propagation are unchanged. Successful
GPU checks and an official serving result remain required for this retry.

### First retry: CPU image provenance gate

Commit `0d69174` started run
[41125598232](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41125598232).
All CI jobs and the existing-image selection/archive passed. Provision verified
the exact bundle image and patch hashes, then the PP4 CPU gate rejected it:
the gate allowed only the historical image or an image whose build commit was
the current `GITHUB_SHA`. A launcher-only retry necessarily has a different
commit from the reused image. This attempt never submitted a Slurm job and
produced no GPU measurements or score; its temporary runtime was cleaned up.

Correct that check to compare reuse with `container_source_commit` from the
configuration. Fresh builds still require `GITHUB_SHA`; sidecar SHA, configured
pin, actual image bytes, embedded source/patch hashes, exact staged overlays
and all CPU behavior checks remain mandatory. Added coverage exercises reuse
through source reconstruction and rejects wrong source commits, wrong hashes,
modified bytes, empty pins, stale successful reports and a fresh build using
an old source pin. Local real-Torch validation: 28 passed; eight real-SquashFS
cases require the Linux CI job. The earlier retry commit's skipped-test note
was imprecise: the one local skip was missing Torch in that pytest environment;
the focused test subsequently passed with `python -m pytest` and real Torch.

### Second retry

Commit `1fa1d36f2fe8abb3d5c3620995454d52a74ceb0c` started
[41138356267](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41138356267)
at 2026-10-31 07:28:30 JST. All CI passed, including 36 Linux PP4 regression
cases with real SquashFS extraction and real Torch, and gigatoken parity.
Provision verified the pinned image hash and the actual PP4 CPU gate passed.

Slurm 27733 was allocated two nodes and failed 1:0 after 6:49. All 149 archive
checksums passed. The sparse penalty gate passed on all four H200s: 150 cases
and four changed-input graph replays per rank; sparse medians 12.23–12.61 us,
dense medians 150.69–151.69 us at batch48/vocab151936/verify-width4. These are
gate timings, not an official serving score. The cache correction is verified
on GPUs.

The next blocker is the new FA3 draft-extend graph. All four Decode ranks
failed its startup hidden-state parity assertion at
`eagle_draft_extend_cuda_graph_runner.py:516`. One reported 122770/163840
mismatches (74.9%), maximum absolute difference 12.3125. Larger capture sizes
passed before the failure near batch10; logits passed the preceding assertion.
This does not establish the internal graph defect, but it rejects this graph
path for adoption. Prefill became ready; Decode did not, so no benchmark or
score was produced. Both Prefill candidates again retained their controls.

For the next retry, keep the unchanged image and sparse penalty implementation,
and set the existing `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1` switch only on
the PP4 Decode worker. This restores its previously served eager draft-extend
path. Target verification and draft-decode graphs remain enabled; the graph
parity assertions are not modified. The original combined forecast includes
the rejected draft-extend graph and does not apply to this reduced bundle.

### Third retry: eager draft-extend

Commit `f619057970614fc8a56d9e6ec1a349cdb0da67f2` started
[41142009481](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41142009481)
at 2026-10-31 08:02:45 JST. All CI passed, including the 36-case real-Torch,
real-SquashFS PP4 regression suite. Provision verified the same bundle image
and the actual image CPU gate passed. Slurm 27739 completed on two nodes in
12:44; serving and archive jobs succeeded. The official result is
9875.188662716093 good output tok/s, eligible and officially valid, with 6234
requests, zero errors and 98.0269% good requests at concurrency 232. It is
0.08% below the parent R8 result: no performance improvement is established,
and the user's 10k objective remains unmet. The startup repairs are successful;
the reduced bundle is not an adopted performance gain. All 176 extracted-file
hashes in the archive's SHA256SUMS passed verification.

The parent R8 score was independently read from its archived `score.json`:
9883.358747435326 good output tok/s, eligible and officially valid, 6241 requests,
zero errors, concurrency 232. Its SHA-256
`288a163650a2aec65eb592bd5bebd518a053d9396b5b510cbe9d57a4613f62b0`
matches that archive's manifest. This is a provisional comparison reference;
a single new run cannot establish a causal performance improvement.

### Follow-up toward the requested 10k result

The user requested an actual result above 10k after the eager retry showed no
gain. The gap from 9875.19 is about 1.264%; the previous bundle's forecast is not
evidence that this gap has been closed. The sparse gate saves about 0.14 ms
per Decode step versus a roughly 37 ms observed step proxy, and does not remove
the Prefill supply limit.

Promote the previously inactive M192/N96 FA3 prototype to a distinct candidate,
`m192n96-direct-o-varlen-pipe`. Preserve the served M128/N192 pipe control,
precision, model, workload, SLA, 16 H200s, concurrency 232 and Decode cap 48.
Keep draft-extend eager for this attempt so the new Prefill candidate is the
major performance factor. Remove the repeated diagonal-trim and split-unroll16
screens, which already lost their timing gates; retain their existing source
for reproducibility.

The measurement question is whether M192/N96 can shorten actual attention and
the Prefill period despite doubling the online softmax/rescale iterations.
Its 34.9% reduction in logical KV tile loads is not a measured HBM reduction
or a throughput prediction. A Prefill period reduction around 1.5 ms would
require roughly 7.5% off the historical 19–20 ms attention share, assuming
Decode does not become the limiter. This is a required saving, not an estimate.

All twelve Prefill GPUs must pass the existing FP32/stock numerical bounds,
changed-input replay checks, paired speed comparisons and 1200 serving-path
calls over six geometries. Because the candidate changes reduction order,
its stable bytes are checked against its own first numerically validated
output; existing candidates keep their original bytewise references. Missing
results, numerical errors, insufficient speed or an endurance failure retain
the served control. An official serving score is still required, and 10k has
not yet been achieved.

The parallel review of draft-extend capture found an invalid parity fixture:
FA3's padding fill is one token, but query/extend width was four. K metadata
came from `seq_lens=1` and Q metadata from `extend_seq_lens=4`; bottom-right
causal alignment masks the first three query rows. This explains why the
roughly 75% full-hidden mismatch can coexist with passing selected-row logits,
but actual GPU parity still needs confirmation. The new fixture uses
width-sized GPU/CPU lengths, distinct free request rows, allocator-owned KV
slots and coherent positions. It restores request tables and buffers and frees
the temporary slots on exit. Full-hidden and logits exact assertions remain.
Seven real-Torch CPU tests exercise the actual fixture's writes, restoration,
page-size 1 and 64 allocation paths, and failure cleanup; the dedicated CI job
now runs these alongside the PP4 CPU gate suite. Draft-extend graphs remain
disabled for the first M192/N96 serving trial.

The integrated candidate and served control both compiled with CUDA 13.0.88
after all 15 declared patches were applied once to a fresh, hash-verified FA3
archive. M192/N96: 94.87 s compile, 525776-byte object, 128 registers, 40/28 B
spill stores/loads; control: 96.03 s, 650736-byte object, 168 registers, 84/116 B
spills. Each emitted a real SM90a kernel and no C7512 warning. Attention and
Firmus image/staging/reference/PP4 regression suites passed locally, as did
Ruff, actionlint, shell syntax and diff checks. Real SquashFS checks still run
in Linux CI before allocation. These are build/CPU checks, not GPU performance.
