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
