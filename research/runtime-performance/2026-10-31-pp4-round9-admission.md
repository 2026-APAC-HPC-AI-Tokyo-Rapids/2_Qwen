# PP4 round 9A: admit stage-0 arrivals before Decode bootstrap

The PP4 direct loop now admits stage-0 requests in bootstrap-queue order when
its local sender polls Bootstrapping or WaitingForInput. WaitingForInput
finalizes immediately; Bootstrapping reserves metadata and joins
`loop.pending`. If metadata is exhausted, the request remains in the bootstrap
queue and is absent from the forwarded admission list. Failed and aborted
requests still follow the failed list. Later stages follow exactly the lists
stage 0 actually produced, irrespective of their own bootstrap state.

`_qpp_result_step` already runs `_qpp_finalize_pending` on every stage, including
stage 0. Its placement stays unchanged: after batch formation, before launch on
the last stage and after launch on the others. A later WaitingForInput poll
initializes the sender and clears `pending_bootstrap`. KV is sent at prefill
result processing, or immediately from the inflight queue if the result was
already processed. Failed polling removes the request from `loop.pending`
without clearing `pending_bootstrap`, aborting or freeing KV. The inflight
release consensus eventually calls `handle_pending_bootstrap`, which aborts
and frees resources on the failing stage at the same release iteration at which
the other stages free their successful transfers.

The existing pending-chunk assertion, admission membership assertion, release
intersection and release lag are retained. This change does not use the
non-PP optimistic retry/requeue path and adds no runtime setting or consensus.

## Evidence and expected effect

User-supplied analysis of run 41073991312 (9883.36 output tok/s): 121 of 6241
requests exceed 3 s TTFT, all in the initial 232-request profiling burst. The
phase opens at 02:41:43.41 server time; the first stage-0 microbatch starts at
+0.29 s with one request, nine bootstrapping and 60 queued; the first pair
starts at +0.51 s. Round 7 with the HF tokenizer also started at +0.26 s.
These facts are inputs to this change, not new measurements.

Removing the local bootstrap admission wait could advance the first pair by
about 0.3–0.5 s. At about 52 requests/s this corresponds to an estimated 15–25
additional burst requests meeting TTFT, or about 0.25–0.4% score. This is an
estimate; no serving run or GPU validation was performed. Steady-state queues
have about one bootstrapping and four waiting requests, so the expected gain
is concentrated at the burst.

## Serving risks

- Earlier admission can retain stage-local KV while Decode is still allocating.
  The 131072-token pool per stage holds only about 32 whole 4032-token requests
  before page rounding and other occupancy: 32 use 129024 tokens, leaving 2048.
  A worker's share of the 232-request burst can exceed this. Normal batch
  formation still gates actual KV allocation; waiting-queue admission does not
  allocate all requests' KV. Slow bootstrap can nevertheless fill the pool,
  stall formation and negate the expected gain.
- Metadata is reserved earlier, including for requests waiting for KV capacity.
  The allocator has twice `max_running_requests` slots. Stage 0 now defers both
  ready and bootstrapping requests when no slot is available. Later stages keep
  the existing reservation/finalization assertions and assume equal capacity,
  admissions and releases. Unequal allocator state remains a fatal invariant
  violation, not a locally chosen admission fallback.
- Pending requests must fit an unchunked prefill. Two approximately 4032-token
  inputs fit the configured 8192-token chunk, but longer inputs or KV pressure
  that causes chunking can trip the existing assertion. Local chunk bootstrap
  checks/requeue would otherwise diverge across stages; the assertion remains.
- Admission order depends on identical request queue order on all stages. Later
  stages scan their queues against stage 0's IDs; the existing assertion checks
  membership counts, not reordered queues. Lost/reordered inputs, inconsistent
  capacity, or premature local release could violate that assumption. This
  change preserves the per-iteration forwarding and common release path.
- A failed bootstrap can consume a full forward and hold KV until every stage
  reaches Success or Failed. A sender stuck Bootstrapping holds resources until
  its existing backend timeout/failure handling acts. There is no new retry or
  timeout. Unexpected local sender states are not a new admission condition.
- CPU fixtures execute extracted scheduler methods, with model, sender,
  transport and allocation fixtures. They cannot establish real NIXL ordering,
  Decode prefix handling, GPU timing, full-model correctness or a score gain.

## Local validation status

All 29 configured SGLang overlays apply from a clean image tree; all 28 Python
files changed by the chain compile. The four-stage CPU source checks pass,
including 32 scheduler-sync cases, 72 output-relay cases and the direct loop
at configured depth 1 with normal and corrupted-copy fixtures (96 iterations,
40 requests and 22 microbatches per scenario). Eleven new pytest cases pass.
The changed Python tests pass Ruff; the reconstructed scheduler also passes
Ruff's standard E4/E7/E9/F checks.

The required full suite returned 1135 passed, 81 skipped, three failed, one
warning and 25 passed subtests. All three failures are outside PP4 admission:
the event-socket and profile-phase socket tests cannot bind AF_UNIX sockets
under the allowed long temporary directory paths. Work stops before commit
because the all-tests-pass condition is unmet. No push, SSH or job submission
was performed, and no performance claim has been validated.
