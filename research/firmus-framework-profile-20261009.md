# Firmus SGLang/vLLM baseline diagnosis — 2026-10-09

## Scope and fixed conditions

This campaign repeats the working reference-07 controls, rather than substituting
an optimized ASPIRE configuration. Model Qwen3-235B-A22B-Instruct-2507-FP8,
revision e156cb4efae43fbee1a1ab073f946a1377e6b969; two Firmus nodes and sixteen
H200 GPUs; four TP2 prefill workers and two TP4/EP4 decode workers; concurrency 8;
ISL 4000 / OSL 200; pinned shared tokenizer and AIPerf 0.12.0; reference timings
20/60/15 seconds. These are not official-duration submissions or accuracy runs.

Pinned server image SHA-256:

- SGLang: 73b2306180f160caf2355425d6c28d79992b0228aa3318850bf692eda604c2c9
- vLLM: 2012b52332deaecdab9a7a397a35412b7e098c40308f93560c7a419519ebfb0e

The vLLM worker homes receive the existing prepared TensorRT-LLM JIT caches;
SGLang uses its existing baseline startup/cache behavior. This asymmetry and the
router difference are part of the reproduced baseline, not controlled away.

## Actions conditions

| Condition | Commit | Actions | Slurm |
| --- | --- | --- | --- |
| SGLang, first reference pass then bounded capture | edc60ba | [37886769411](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37886769411) | 22003 |
| vLLM, first reference pass then bounded capture | e9624ed | [37886786534](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37886786534) | 22005 |
| vLLM, extra profiler-free control, cancelled at user's direction | 3e0a538 | [37886821128](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37886821128) | 22006; cancelled while pending, zero GPU time |
| SGLang, extra profiler-free control, cancelled at user's direction | 13ca2cc | [37886839709](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37886839709) | 22004; cancelled while pending, zero GPU time |
| SGLang, extra first-peer wall timers plus capture, cancelled at user's direction | e3bb75e | [37888308774, attempt 3](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37888308774) | 22044; cancelled while pending, zero GPU time |

All are submitted through Actions. Full Linux validation passed before submission.
The jobs require both nodes, so queueing together does not imply parallel execution.
The extra timing diagnostic passed validation/provisioning but `sbatch` rejected
it with `QOSMaxSubmitJobPerUserLimit`. No scheduler job was created and its runtime
directory was removed. After an account slot opened, attempt 2 failed when the
SSH output-transfer connection closed (`tar: This does not look like a tar
archive`). Its partial runtime was verified and removed; no Slurm job existed.
Attempt 3 successfully submitted job 22044. Other jobs were left untouched.

At approximately 07:35 UTC, SSH output-transfer failures interrupted the original
Actions monitors for jobs 22003, 22005, 22006 and 22044. Their Slurm jobs and
runtime directories survived, still pending. An observed wrapper `tar` status 2
was being treated as fatal even with an exact pending observation. Commit
8e80e60 fixes these two demonstrated cases: pending-without-state plus tar exit,
and explicit SSH disconnect plus tar exit. Other nonzero checker errors remain
fatal. CPU validation 37900350759 passed.

The first recovery Actions 37900458055 at a7d894a used an incorrect job-ID mapping
inherited from the initial status table. Actual Actions `SLURM_JOB_ID` records
and Slurm `Command` paths showed the correct mapping above. That recovery
workflow was cancelled before any scheduler execution; no Slurm jobs were
cancelled by that correction.
[Corrected recovery Actions 37901339985](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37901339985)
at 538fe12 passed CPU validation and checks the scheduler script path as well
as original run/commit provenance before monitoring or collecting. Its matrix
identifies existing jobs only; no scheduler job or condition was added. Recovery
artifacts retain original provenance and are archived under the recovery Actions
ID. Active runtimes are preserved if recovery fails.

The user rejected additional unprofiled controls. Jobs 22004 and 22006 were
verified as `profile=false`, cancelled while pending, and their runtime cleanup
was confirmed. The user also rejected the supplementary first-peer diagnostic;
22044 was cancelled while pending with zero GPU time. The remaining scheduler
jobs are 22003 and 22005, both
originally submitted by their linked Actions workflows. The five-jobs/two-runs
UI discrepancy arose because a single recovery workflow monitored four surviving
Slurm jobs while the other original workflow monitored the fifth.

## Verified existing baseline controls

Actions history was reread, including the original submission and terminal
scheduler/score records, rather than relying only on prior notes:

| Baseline | Actions | Commit | Slurm | Verified outcome |
| --- | --- | --- | --- | --- |
| SGLang, no profiler | [37562274684](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37562274684) | c7bb031 | 21488 | COMPLETED 0:0, 8m34s, 98 measured requests, zero errors, reference validation valid, 84.7% SLA-good/ineligible |
| vLLM, no profiler | [37562605777](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37562605777) | caa006f | 21489 | COMPLETED 0:0, 9m18s, 184 measured requests, zero errors, reference validation valid, 100% SLA-good, 596.370 good tokens/s |

Both workflows and archival jobs succeeded, and their logs explicitly show
`PROFILE=false`. Git diffs of `configs/firmus-framework.json` between each
baseline and its profiling rerun are empty: framework, concurrency 8,
reference-07 workload and both image hashes match. These existing runs supply
the unprofiled controls; additional unprofiled GPU runs are not required.

## Why capture support needed repair

The remote branch already contained Nsight support, although the initial local
checkout was stale. Archived SGLang Actions 37653746232 / Slurm 21641 completed
its reference client but both decoder `/stop_profile` calls exceeded 180 seconds.
The decoder log showed Nsight still exporting its long capture when cleanup killed
the workers. The earlier vLLM Actions 37630835533 / Slurm 21617 had valid kernel
coverage on all sixteen GPUs, but report flushing stalled workers during measured
traffic. Both Actions also failed the summary command because the home archive
shell lacked `python3`.

The repaired flow completes and validates the unchanged reference client before
starting a second identical client under `profile/client/`. It captures the same
ten-second window on both frameworks, five seconds into the second benchmark
phase. First-pass data lives under `results/`; diagnostic client data is separate.
Nsight remains attached during the first pass, so the verified existing
unprofiled runs supply the performance controls. Native reports and per-device/top-kernel
summaries are retained; temporary SQLite exports are removed after validation.
Summary generation now prepares Python and runs on the Actions runner.

## Historical evidence to test

Read-only replay of the prior matched-settings controls on Firmus:

| Metric | SGLang 21399 | vLLM 21412 |
| --- | ---: | ---: |
| Measured requests | 85 | 184 |
| Raw output tokens/s | 270.419 | 597.061 |
| Mean TTFT, ms | 3350.890 | 333.893 |
| Mean ITL, ms | 12.205 | 11.715 |
| Requests with TTFT > 3000 ms | 7 | 0 |
| Mean TTFT among requests <= 3000 ms, ms | 445.839 | 333.893 |
| Mean ITL among requests <= 3000 ms, ms | 12.188 | 11.715 |

The seven slow SGLang requests began 0–17.908 seconds into the measured phase and
had TTFT 23.536–41.123 seconds. They consumed 250.050 request-slot seconds waiting
for first tokens, approximately half of the eight slots × 60 seconds available
during the send phase. All returned 200 output tokens. This is a direct explanation
for much of this short run's throughput deficit, although the mechanism causing
the waits still needs attribution.
For scale, raw throughput was 54.7% lower, while mean ITL was only 4.2% higher.
The 52.1% request-slot occupancy by these waits is consistent with the size of
the throughput loss. This occupancy accounting is not an independent causal
experiment and excludes queue/drain effects.

The SGLang router log declares CacheAware routing; the vLLM proxy explicitly
round-robins prefill and decode clients. In SGLang 21399, prefill-1 did not process
a full-length warmup request and logged its first full-length batch at 20:59:33,
about forty seconds after the measured phase began at 20:58:53. The other prefill
workers processed full-length warmup requests. This supports investigating
incomplete warmup / first-use work and routing rather than assuming a uniform
compute slowdown. It does not by itself identify JIT compilation or NIXL as the
cause. Most logged DeepGEMM precompilation occurred before client warmup.

These controls must not be conflated with older official-duration jobs 19085 and
19221 (917.178 and 1057.144 eligible output tokens/s respectively). Their roughly
15.3% difference and the much larger short-reference difference have different
warmup, duration and configuration contexts.

## First-peer descriptor scale (source-based hypothesis)

Read-only inspection of the pinned SGLang image's
`sglang/srt/disaggregation/nixl/conn.py` shows `_add_remote_peer` calling
`_prepare_payload_xfer` on the prefill bootstrap thread. For TP2→TP4,
`_init_hetero_tp_prep_handle` builds destination addresses for every K/V layer and
all advertised decoder KV slots before calling `agent.prep_xfer_dlist`.
The decoder sends `dst_num_slots = kv_data_lens[0] // dst_kv_item_len`, so it uses
cache capacity rather than the live request's ~4000 tokens.

The historical log reports 1,531,197 decoder slots; the model config has 94 layers
and four KV heads. That implies `2 × 94 × 1,531,197 = 287,865,036` destination
entries per peer, or 6.434 GiB for the uint64 address/length/device matrix alone.
This excludes temporary arrays, source descriptors and NIXL's own structures.
It is a calculation from observed geometry and source, not a measured memory
peak or elapsed time. A first-use peer path of this scale is a concrete candidate
for the 23–41-second TTFT waits. It still needs direct timing or a controlled
cache-capacity intervention before assigning causality.
The pinned vLLM `nixl/base_worker.py` constructs remote descriptors by KV block
(`region_num_blocks`, `block_strides`) rather than individual token slots. Its
source SHA-256 in the pinned image matches the extracted build tree:
`49a2e4faf99bef146b5a1d4d18d48f240b1e561c438252274494b359db96ef52`.
Both implementations cover cache capacity, so block granularity is a relevant
distinction; the actual vLLM block size and setup time have not been measured
here, and this source difference alone does not prove the observed stall cause.

## New measurements and trace attribution

Both requested jobs completed with exit 0:0. SGLang 22003 ran for 14m38s on
2026-10-09 19:36–19:50 JST; vLLM 22005 ran for 14m13s on 2026-10-10
00:14–00:29 JST. Both recovery job rows and the archive job succeeded. The
overall recovery workflow includes expected failures from cancelled auxiliary
jobs; those do not indicate failure of either requested profile.

Both archives contain `run-success`, six native Nsight reports and a valid
`profile/validation.json` with CUDA kernels on all sixteen GPUs. Captures cover
a nominal ten-second second-pass window after warmup. Stop/export took about
47–48 seconds and affected only the diagnostic client, not the preceding normal
reference pass. Runtime directories were cleaned after successful collection.

| Normal reference-pass metric | SGLang 22003 | vLLM 22005 |
| --- | ---: | ---: |
| Requests / SLA-good requests | 144 / 138 | 184 / 184 |
| Errors | 0 | 0 |
| SLA-good fraction | 95.83% | 100% |
| Eligible good output tokens/s | 444.314 | 596.157 |
| Mean TTFT, ms | 900.179 | 336.317 |
| Mean ITL, ms | 12.608 | 11.743 |
| Mean request latency, ms | 3409.078 | 2673.039 |
| Mean TTFT for requests within 3000 ms, ms | 464.665 | 336.317 |
| Mean latency for requests within 3000 ms TTFT, ms | 2976.067 | 2673.039 |

The new vLLM goodput is 34.2% higher; equivalently SGLang is 25.5% lower.
This is a reproduced reference comparison, not an official-duration score or
an estimate of a universal framework advantage. SGLang improved substantially
over its existing 21488 control, whereas vLLM closely repeated its 21489 control.
That variability argues against assigning the entire historical gap to a fixed
per-token GPU cost.

SGLang has six slow first-token requests, starting in the first 14.331 seconds,
with TTFT 4.233–17.950 seconds. Their first-token waiting time sums to 65.502
request-slot seconds, about 13.6% of eight slots × the 60-second send phase.
vLLM has no requests over the 3000-ms TTFT target. The six waits end in a narrow
interval around 18–19 seconds into the phase. These early waits reduce traffic
and lose SLA credit. The ~7.4% mean ITL difference and the remaining fast-request
TTFT difference show that startup stalls alone do not explain the entire gap.

The steady-state trace summaries show different execution paths. SGLang decoder
hotspots include `fused_moe_kernel` (mean 21.6–23.5 microseconds), FP8 group
quantization (7.6–8.4 microseconds for one prominent variant), and fused
all-reduce (10.5–11.4 microseconds). vLLM uses DeepGEMM grouped GEMMs,
TensorRT-LLM scaling/routing and a different fused all-reduce implementation.
Its prominent scaling kernels average about 2.4 microseconds and its fused
all-reduce about 10.1–10.2 microseconds. These are observed hotspots and
optimization candidates, not equivalent-operation speed ratios: batch shapes,
fusion boundaries and call counts differ. Summed kernel time is neither GPU
utilization nor a critical-path attribution.

The supported diagnosis is an early first-token stall component plus a smaller
steady-state latency component. The exact source of the initial stalls remains
unconfirmed: the bounded captures occur in the second pass, after the first-use
events responsible for the first-pass tail. Existing warmup/routing evidence and
the NIXL descriptor-size calculation remain hypotheses; no supplementary
timing experiment was run. No further GPU jobs were submitted for analysis.

Collected evidence:

- [SGLang artifact](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37901339985#artifacts)
- [vLLM artifact](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37901339985#artifacts)
- Canonical archive: `/srv/qwen-artifacts/37901339985/1/`;
  the archive job generated `SHA256SUMS` after copying all collected evidence.

## Evidence ownership

Tracked conditions and change reasons are in Git commits. Execution history is in
Actions/Slurm logs. Canonical raw artifacts, including image/cache provenance and
Nsight reports, are archived under `/srv/qwen-artifacts/<run>/1/`
after collection. This report contains derived findings and links rather than a
second copy of raw request exports or traces. No BigQuery credentials reach the
compute nodes. Only this campaign's temporary artifacts/jobs are cleaned up.
