# Firmus vLLM structural optimization campaign

User authorized continued overnight investigation and focused execution on 2026-10-13, aiming to explain the gap between our verified 4,759.69 good output tokens/s and competing teams' reported 7k on the same Firmus hardware/rules. Roughly 9,519 good tokens/s is a research target, not an expected outcome. The reported 10k remains unverified. Keep all official inputs, checkpoint revisions, precision, correctness and SLA gates. The repository parser validates workload/SLA; it is not the organizer's full accuracy evaluation.

## Scope and queue coordination

The previous two-candidate campaign is complete. This is newly authorized work, with a finite initial budget: at most two distinct official candidates in sequence, and a focused repair only for a demonstrated failure. Reuse existing controls and the already-submitted compatible profile. Do not duplicate a team experiment, add a concurrency sweep, or launch an additional profile without identifying missing decisive evidence. Expand work only when results justify a concrete next step within user authorization; record the scope explicitly.

Hold new GPU submission until the four existing related team allocations are terminal and their decisive logs/results are reviewed. The shared queue is a real constraint. Read-only monitoring must leave their Actions, jobs and staging untouched. Exact provenance and scheduler Command/WorkDir were checked at 2026-10-13T10:25Z; all four were PENDING, zero runtime:

| Experiment | Actions attempt 1 | Slurm | Exact source |
|---|---:|---:|---|
| Katayama decoder SYMM_MEM | 38345930105 | 23084 | 3841c5822a57c1a5d5ccbe60bf1100f80240b435 |
| Katayama decoder M tile32 | 38346456470 | 23086 | 5b96e63069989e61ae97b9eb02aece659f459335 |
| Katayama compatible profile-only | 38350383403 | 23091 | 9815b6103b8068bd5f0a68d71754fdba1305d1eb |
| Komori SGLang DP4/EP4 MegaMoE | 38369214512 | 23120 | 2e7e6f4a29cbb1439ebab88c5a037595330f8f15 |

The first three reuse server image 2012b52332deaecdab9a7a397a35412b7e098c40308f93560c7a419519ebfb0e and client 4b919a7465f1647f4eddc677393cef00f2f390a8544b52a5f6b2456dce940d62. Komori uses new image 67a08ed37d06100e1b130f5d46a1af385346b550893dbfb76b6449c7c2ff1eb1. The profile-only source deliberately skips normal scoring even though its common provenance field says workload official; do not mislabel it as an official throughput result.

Monitoring stays in chat 01a12dff-78cb-7472-98d8-54e53968634e on GPT-6 Luna max every 15 minutes. It reads state and routes meaningful events to original chat 01a11efd-9079-73f3-a189-3dfd01a877fb on GPT-6.1 Sol medium. Luna must not submit or repair. Sol owns source reasoning, candidate selection, Actions submission, terminal verification and reporting. Unchanged checks stay quiet. A verified completion of the team experiments is a dependency event for this new campaign, not permission to modify their jobs.

## Measured client-stage budget

`analyze_stage_budget.py` reads the existing canonical exports and preserves their stored official scores. `stage-budget.json` contains input hashes and diagnostics for the three clean runs. Mean occupancy is computed by intersecting each raw request's before/after-first-token intervals with a common 30-120 second window, including requests crossing window edges. Before-first-token time combines prefill, queues, KV transfer, decoder admission and HTTP overhead. It cannot isolate those components. After-first-token occupancy is a client-visible proxy, not a measured engine batch.

| Configuration | Good output tok/s | Observed input tok/s over scored duration | Stationary before-first-token inflight | Stationary after-first-token inflight | Stationary TTFT ms |
|---|---:|---:|---:|---:|---:|
| c104 / proposals3 | 4710.019 | 95358.761 | 22.658 | 81.237 | 924.786 |
| c104 / proposals2 | 4732.870 | 95553.209 | 29.256 | 74.642 | 1194.128 |
| c112 / proposals2 | 4759.691 | 96412.997 | 35.022 | 76.881 | 1414.471 |

The two modifications did not materially increase sustained input capacity. Higher concurrency mostly increased the before-first-token population. This does not prove that prefill execution alone is the bottleneck: admission and handoff are included. At unchanged workload, 7k requires at least 140k configured input tok/s and 9,519 requires about 190.4k, before rejected requests and overhead. Both stages must sustain the corresponding request rate. Target sensitivity calculations holding current latency/good fraction are arithmetic, not a calibrated simulator.

## Source and history audit

Selected source files were extracted from the actual pinned SquashFS using a short CPU-only diagnostic; the remote temporary extraction directory was removed automatically. `pinned-source-provenance.json` records image identity, byte sizes and exact file hashes. No package, source mirror or image was changed. `audit_pinned_moe.py` refuses nonmatching source hashes before executing a small set of native CPU decision functions, with simulated Hopper/package/platform dependencies. `pinned-moe-audit.json` explicitly states that these checks do not prove GPU execution, CUDA capture, memory capacity, numerical accuracy or speed.

The Hopper block-FP8 EP1 auto selector deliberately prioritizes TRITON. The explicit deep_gemm backend maps to native TritonOrDeepGemmExperts. Its DeepGemmExperts implementation supports the static128-block/dynamic128 activation scheme and has a native fused SiLU/multiply/FP8-quantization path. TP2 expert intermediate width is 768, hidden width4096; representative M128-8192 dimensions pass the native 128 alignment check. Non-E8M0 small shapes can fall back to Triton. The existing Hopper E8M0 oracle can bypass that fallback; do not change the global scale mode, which would also alter dense GEMM and quantization. The BF16 EAGLE3 checkpoint is a one-layer dense LlamaForCausalLMEagle3, not a BF16 MoE that requires this FP8 backend.

The old trace attributes about43.35% of recorded decoder kernel duration and47.7-48.1% of prefill duration to Triton MoE. It is not a critical-path savings estimate. Its unsupported-driver/missing-event warnings prevent treating it as complete activity coverage or proof of hardware saturation. Use Katayama's compatible trace when available to separate target verify, draft, useful versus padded expert work and exposed collectives.

Prior SGLang attention-DP8/EP8 prefill was already measured at about6.9k input tok/s/GPU, slower than TP2, and was rejected. Do not repeat that layout or assume expert sharing guarantees speed. Existing broad SGLang results and generic AISimulate tables do not validate this exact vLLM configuration; the packaged exact H200/QwenFP8/vLLM0.30 simulator support was NO.

## First prepared official candidate: prefill native DeepGEMM

Parent verified control is 769c314 / compute7880056 / Actions38341761493 attempt1: c112, proposals2, P6TP2/D1TP4 EP1, score4759.690607. Change only the target MoE backend on all six prefill workers from auto (actual TRITON) to deep_gemm. Decoder backend/reductions, BF16 draft, images, checkpoints, c112, KV/block64 capacity, graphs, FA3, chunk/context8192, routing and the official60/120/30 workload stay fixed. Server-info assertions require the requested prefill/decode backend in addition to all existing effective configuration checks.

This native path is already present in the pinned image, so no rebuild is required. No [profile] tag belongs on this normal candidate. Preserve all six target/draft58-bucket prefill signatures and decoder53 mapped buckets. Native grouped GEMM workspace, JIT/capture completion and actual backend execution remain GPU checks; backend naming alone does not prove all runtime shapes use DeepGEMM. Do not relabel this candidate as a prediction of 2x speedup.

Decision criteria: first assess successful capture/readiness and official eligibility, then input throughput, the before-first-token interval and decoder service. A larger raw rate with SLA fraction below.90 is a rejected candidate. A small single-run change is inconclusive rather than a confirmed improvement. Use existing c112 and c104 controls; do not infer framework-wide causality. If the team results invalidate this candidate before submission, revise the prepared change rather than spend compute just because it is ready.

## Second conditional candidate: native decoder EP4 with attention TP4

The exact FusedMoEParallelConfig.make function with TP4/DP1/PCP1/SP1 and enable_expert_parallel=true produces MoE TP1/EP4, while attention remainsTP4. Each rank stores32 complete experts with intermediate width1536 rather than128 tensor-sliced experts at width384. Its use_all2all_kernels property is false for this layout: it does not require DP-attention, DeepEP or a new all-to-all dispatch path. The attention/KV partition stays TP4, preserving the existing heterogeneous TP2-to-TP4 connector layout.

Hold the actual decoder MoE implementation at Triton with explicit --moe-backend triton, so the experiment isolates expert partitioning rather than allowing auto to switch to FlashInfer CUTLASS. Preserve memory/admission/graphs/model and proposal count. Verify local expert mapping, capture, correctness and performance; source configuration acceptance does not prove runtime support or savings. This differs materially from Komori's SGLang attentionDP4/EP4/MegaMoE combination. Prepare it only after the first candidate and team results establish the proper control. If useful, retain the first candidate's prefill backend; otherwise return to the verified control. Never combine pending team settings and call the result a single-factor comparison.

## CPU validation and execution status

First candidate full Linux CPU suite passed320tests, one existing skip and22subtests; repository Ruff and all shell syntax checks passed. Exact pinned function audit and new analysis Ruff checks pass. Windows CRLF text in the local validation snapshot was normalized to LF for Linux, and uv/jq/zstd/dependencies were isolated in temporary local WSL directories. Those directories were removed on exit. No GPU job has been submitted for this new campaign.

Automatic approval review rejected uploading a nearly complete validation-source tar to archive-host as sensitive source egress. The rejected command did not execute or transfer source. Validation was completed entirely in local Ubuntu WSL instead. This is resolved, not an outstanding permission request.

Submission uses a new traceable [cluster:firmus] [nodes:2] compute commit only after the dependency gate and CPU validation. Record exact compute/run/attempt and scheduler Command/WorkDir before mapping; never use commit-run PR-filter results as compute evidence. No manual sbatch, blind rerun, extra baseline or sweep. Expected normal run12-13minutes, existing17minute allocation cap; queue delay has no reliable bound.

For each owned terminal run: archive attempt-correct /srv/qwen-artifacts/<run>/<attempt>/ before cleanup; check all hashes, independently replay the repository official-workload parser, inspect raw JSON/JSONL, all seven effective configs and target/draft captures, actual backend logs, provenance, accounting and run-success. Check Actions collection/cleanup and GitHub artifact removal. Identify the first fatal command/error before any minimum repair. Delete only owned exact provenance-verified terminal staging after canonical verification. Preserve all unrelated team work and original checkout's four build-file deletions. Finish the finite campaign with an honest [skip ci] report, cleanup of own helpers/temporary files, and pause monitoring.

## First dependency reviewed: decoder SYMM_MEM

Actions38345930105 attempt1 / Slurm23084 / exact3841c582 ran on g1/g4, COMPLETED0:0 in12m47. Execution115127999202 and archive115506670900 succeeded. Independently checked all64canonical hashes, exact repository parser replay including official-workload validation, raw JSON/JSONL and per-request counts, all seven effective exports and exact graph buckets, target/draft revision and draft weight hash, native backend and dispatcher logs, accounting/run-success. GitHub artifacts are empty and the exact team-owned staging directory is absent. No team resource or archive was modified.

| c104 / proposals2 | Good tok/s | Raw tok/s | Good fraction | TTFT ms | ITL ms |
|---|---:|---:|---:|---:|---:|
| Ordinary decoder reduction control38331343958 |4732.869877|4794.111177|98.719245%|1244.653060|15.234328|
| Decoder SYMM_MEM38345930105 |4734.019108|4795.320452|98.718813%|1227.975353|15.303856|

There were2966measured requests,2928good,38TTFTmisses,0ITLmisses and no errors/mismatches/cancellations. Score changes only+0.02428%, raw+0.02522%, TTFT-1.340%; ITL+0.456%. Single observations at different times establish no meaningful improvement or reproducible superiority. Keep the current control and prepared native-prefill candidate; do not combine this setting based on its score.

The runtime confirms TP dispatch order['SYMM_MEM','PYNCCL'] and disable_custom_all_reduce=true, while compilation still enables fuse_allreduce_rms and initializes a FlashInfer allreduce norm-fusion workspace with backend=mnnvl. This is evidence that the ordinary communicator changed, not proof that all dominant profiled reduction kernels were replaced. Compatible trace/active compiler-path inspection is needed before concluding reduction optimization has no headroom.

Verification retained AIPerf's native goodput field: multiplying it by200 exactly reproduces4734.019107963232. Raw per-request good count and fraction independently match2928/2966. Dividing that count by the separate benchmark_duration123.696518912 would give4734.167179082919 because the exported goodput implies123.700387904seconds. The tiny time-basis difference is recorded rather than substituting a new score or weakening the official validator; its precise internal origin was not established in this review.

The three other dependencies remain pending in the latest monitor snapshot. No structural GPU job has been submitted; the dependency gate remains active. Team-start events alone need not wake the discussion chat; future notices should focus on terminal results, usable profile/collection completion, or required action.

## Second dependency reviewed: decoder tile32

Actions38346456470 attempt1 / Slurm23086 / exact5b96e630 ran on g1/g4, COMPLETED0:0 in12m49. All65canonical hashes passed. In-memory replay of the exact commit's repository parser including official-workload validation exactly reproduced4728.963066412307. Raw2958measured requests independently give2918good,38TTFTmisses,2ITLmisses,0overlap/errors/mismatches/cancellations. All seven effective configuration exports, exact graph modes/buckets and completed signatures, target/draft/image identities, accounting and run-success were independently checked. The archived E128/N384 H200 block-FP8 tuned file has BLOCK_SIZE_M32 at batch256/512; the decoder log explicitly confirms loading that exact runtime source path. Config-file loading does not measure the distribution of selected runtime shapes.

| c104 / proposals2 | Good tok/s | Raw tok/s | Good fraction | TTFT ms | ITL ms |
|---|---:|---:|---:|---:|---:|
| Control38331343958 |4732.869877|4794.111177|98.719245%|1244.653060|15.234328|
| Tile32 38346456470 |4728.963066|4793.698649|98.647735%|1102.871253|15.961312|

Score-0.08254%,raw-0.00860%,TTFT-11.39%,ITL+4.77%. This is a single-run latency tradeoff with essentially unchanged throughput, not evidence for adopting tile32 or rejecting all MoE optimizations. Do not launch a tile sweep or add this setting to the prepared candidate.

GitHub's exact job endpoint and connector both reported execution115129709912 completed/success while retaining stale in_progress/pending step fields. Finalized execution logs reach post-job cleanup at21:28:26UTC; archive job115549308158 completed successfully. Canonical completeness/checksums/run-success, empty GitHub artifact list and absence of exact team-owned staging pd-38346456470-1.Ct8GJGyWSp independently establish execution/collection/cleanup completion despite that metadata inconsistency. No team data or resource was modified.

Two dependencies are now reviewed with no demonstrated throughput gain: generic decoder SYMM_MEM and tile32. Await compatible profile-only38350383403/23091 and SGLang DP4/EP4 MegaMoE38369214512/23120 before the prepared native-prefill DeepGEMM submission. The initial finite structural campaign remains authorized and active; no owned GPU job has yet been submitted.

## Third dependency reviewed: failed compatible profile-only capture

Actions38350383403 attempt1 / exact9815b610 / Slurm23091 ended FAILED134:0 on g1/g4 in13m53. All seven servers became ready after604seconds. This was explicitly profile-only20/30/30 with2048input candidates, not the official60/120/30 pass. Provenance confirms profile=true,measurement_mode=profile-only and the same pinned serving/client images. No score or run-success is expected or present.

First fatal command: the diagnostic srun invoking scripts/official-aiperf.sh. Its AIPerf process printed `Assertion failed: check () (src/msg.cpp:387)` and aborted at shell line19 with exit134 during client shutdown, after processing175valid records. The diagnostic phase had also reached its grace timeout with104requests cancelled; it is not valid performance evidence. No worker fatal was found before the client abort. Subsequent worker SIGKILL/cancellation messages are cleanup consequences, not the initiating failure. The precise internal assertion cause is unproven.

The capture helper started all7endpoints, reported capture_active at21:56:33UTC and requested stop at21:56:38. It never recorded capture_stopped; all7stop response files are empty. The client failure bypassed the normal wait for the capture helper, and cleanup terminated workers while native report finalization was underway. Decoder generation log reached98% before termination. This establishes why finalization cannot be presumed complete; it does not independently prove the client's internal messaging assertion mechanism. Any future owned profile repair should preserve capture finalization/status before teardown even on client failure, with a measured allocation budget; no teammate source/job was changed or rerun.

Collection subsequently completed and archived101files under /srv/qwen-artifacts/38350383403/1/. All101SHA256checks independently passed; source/accounting/sevenserverexports and7native reports are retained. Execution job115142073744 ultimately reportsfailure, archive115563197547 success, GitHub artifact list empty, exact team-owned staging absent. Early archive/staging observations occurred during collection; finalized logs and later canonical checks reconcile them. No own GPU allocation or team cleanup was performed.

Independently CPU-exported all7reports on archive-host with the exact Nsight2026.3.2.476package, SHA256e9bf0ff9c613ff5545095a4177676a9579cb666e78c8621c3792f3559e2c1690. The decoder8945029-byte report fails export with `Section Table Reference magic number mismatch`. Six prefill reports export with SQLite integrity=ok and two recorded GPU ranks each, but include `Not all CUDA events might have been collected` and NVTX warnings. They fail the same strict complete-capture warning gate; warning PID scope was not further localized, so recorded prefill data is not declared complete. There is no valid complete16GPU replacement trace and no trustworthy new decoder critical-path breakdown.

Derived CPU integrity evidence is /home/USER/qwen-analysis/firmus-vllm-structural/38350383403-1/profile-integrity.json on archive-host. The exporter package/extracted tools/SQLite temporary directory was automatically removed. Canonical raw was never modified. This CPU recovery used no GPU and avoided guessing from exit134 or report existence.

The failure is confined to this instrumented diagnostic path; existing clean official runs remain valid controls. Keep the native-prefill candidate prepared, without upgrading old incomplete profile shares into proof of saturation. Await the final original dependency, MegaMoE38369214512/23120, and review its terminal evidence before a single new normal Actions submission. Do not auto-rerun this teammate profile or launch an extra GPU profile merely to replace it.

## Final original dependency reviewed and first candidate released

Komori Actions38369214512 attempt1 / exact2e7e6f4a / Slurm23120 ended FAILED1:0 on g1/g4 in5m58 (2026-10-13T22:47:43-22:53:41UTC). The native SM90 MegaMoE EP4 numerical gate passed: original FP8 scales, late FP32 combine, empty/tail ranks and changing masked graph inputs. All three four-GPU prefill groups and the four-GPU decoder became ready after351seconds. Decoder target-verify and draft-decode graph captures completed on all four DP ranks.

The first underlying fatal command was scripts/check-firmus-server-info.py checking the decoder export: `ValueError: decode state 0: speculative_moe_runner_backend='deep_gemm', expected None`. Exact source expects None for the speculative backend while all four archived decoder DP states export deep_gemm. This is a configuration-validation failure before AIPerf, not evidence that the MegaMoE numerical gate or throughput failed. The precise semantics behind the inherited/exported draft field are not resolved here, and teammate source was not repaired. Later Slurm-step termination was teardown. Workflow monitoring's subsequent `slurm_load_jobs: Invalid job id specified` is not the initiating allocation error.

The attempt-correct canonical archive /srv/qwen-artifacts/38369214512/1/ independently passes all44SHA256checks, including build artifacts. Exact source/run/attempt/image provenance, four server exports containing all16DP states, numerical log and capture/readiness evidence are retained. No AIPerf exports, official score, native profile reports or run-success exists because the configuration gate stopped before measurement. Execution115211147822 failed, archive115583131672 succeeded, GitHub artifacts empty and exact team staging absent. Read-only review modified no team resource.

All four original dependencies are now terminal and reviewed. Neither the ordinary decoder reduction nor tile32 established a meaningful throughput gain; the two other runs supplied no valid scored result. This does not invalidate the prepared native vLLM prefill candidate. The user explicitly requested continuation on2026-10-14 because queues are relatively empty. The latest queue shows only unrelated23235 running; leave it untouched and do not extend the original finite dependency hold to every new team submission.

Release one already CPU-validated candidate through a new [cluster:firmus] [nodes:2] commit, reusing pinned images without [require-build] or [profile]. The only performance change from the verified c112/spec2 control remains six prefill target MoE backends auto to deep_gemm. This submission adds a rationale comment to the runtime script so the normal push path is traceable and is not suppressed by documentation-only path filters; no validated launch setting changes. Track exact compute/Actions attempt and scheduler Command/WorkDir, then complete all official-result/archive/cleanup checks before choosing the conditional decoder EP4 candidate.