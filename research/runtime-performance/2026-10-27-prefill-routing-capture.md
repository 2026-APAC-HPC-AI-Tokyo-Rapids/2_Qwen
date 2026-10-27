# Prefill routing capture from a push

This experiment records the physical top-8 expert IDs for eight target forwards
on each of three Prefill workers and their four source EP ranks. It uses the
current Firmus image and Python runtime overlay; it does not need an image
rebuild. Expert weights, placement, arithmetic and normal AIPerf arguments are
unchanged. This is a diagnostic, not a candidate speedup.

The experiment config sets `prefill.routing_capture_samples` to `8`. Provisioning appends
`patches/sglang/prefill-routing-capture.patch` to the staged SGLang overlays.
Absent or zero samples disables capture. Deployment rejects non-profile runs,
unsupported topology, or missing overlay/placement before requesting GPUs.

To submit this configuration through the existing Qwen push workflow, make a
tracked runtime/config change and use the following commit subject:

```
[cluster:firmus] [nodes:2] [profile] Capture source-rank Prefill routing
```

Push that experiment branch. Do not prefix the submission commit with
`[skip ci]` or `[require-build]`. A documentation-only change is insufficient:
the workflow excludes `research/**` and Markdown-only pushes. The workflow
runs CPU validation, reuses the pinned image, submits Slurm, collects results,
checks terminal state and archives artifacts through the existing route.

The ordinary 60/120/30-second client runs first. Only after it succeeds are the
`active` markers created for the separate profile workload. The first eight
target forwards of that diagnostic workload (including its warmup) are sampled;
they are not claimed to represent only the steady profiling interval. There are
no capture tensor operations when the marker is absent or the limit is reached.
All ranks include idle forwards, with zero local rows. Recording performs copies
and host synchronization, so diagnostic timings are not adoption evidence.

Raw files are `results/prefill-routing/prefill-{0,1,2}/rank-R-sample-S.npz`.
Each contains `ids` (94 layers × live tokens × 8, physical IDs) and JSON metadata,
including the actual target forward ID. The loaded placement is archived as
`source/prefill-expert-location.json`. No prompt text or generated text is added
to these routing files.

`scripts/summarize-prefill-routing.py` validates every worker/rank/sample,
matching forward IDs, geometry, live IDs and distinct logical top-k experts.
It writes `results/prefill-routing-summary.json`. Missing evidence fails the
job; it never falls back to the old rank-summed statistics.

The summary reports each layer's source-to-expert-rank matrix and source-rank
logical-expert counts. FC2 output travels in the reverse matrix direction. Each
expert output is 4096 × BF16 = 8192 bytes. Remote output and local combine-read
payloads are kept separate; link overhead and exact transfer time are not
inferred. The cold-expert relocation bound ignores capacity and load balance;
the distinct remote token/destination count is a hypothetical partial-sum bound,
not an implemented or automatically beneficial reduction.

CPU checks cover disabled capture, sample limits, idle ranks, corrupted and
incomplete evidence, rank alignment, traffic direction, byte counts, launch
gating and terminal validation. GPU execution is still required to validate the
capture in the serving environment.
