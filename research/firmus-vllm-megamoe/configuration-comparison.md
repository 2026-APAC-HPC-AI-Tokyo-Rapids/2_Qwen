# Matched configuration: Komori 6344.96 TPS versus vLLM MegaMoE

Reference: SGLang `e6e295bd`, Actions 38664373784 / Slurm 23571. vLLM attempted baseline: `26b3631`, Actions 38781401340 / Slurm 23872 (failed during decoder graph capture, before readiness or benchmark traffic). Source of actual SGLang settings is the canonical `source/source/server-info-*.json` plus worker startup logs; the prior vLLM launch 38775681156 reached and exported the same serving configuration but sent no benchmark traffic.

The table describes the matched baseline, including the repaired normal trials72688a7/76261bc and profileedd9929. The later prefill-admission candidate deliberately changes P48 to P2 and is described separately below; it was cancelled before execution.

| Dimension | SGLang reference | vLLM reproduction / remaining difference |
|---|---|---|
| Hardware and topology | 2 nodes, 16 H200; three 4-GPU prefill groups and one 4-GPU decode group | Same; two P groups on head, one P and D on tail |
| Attention / experts | Attention TP1, DP4 and EP4 per group | Native TP1/DP4/EP4 with 32 linear experts per rank |
| MoE implementation | Modified SM90 MegaMoE; FP32 late routing-weight combine | Same kernel patch; vLLM method owns dispatch, expert math and combine, bypassing duplicate native communication |
| Target / draft precision | Official published FP8 target; BF16 Nebius EAGLE3 | Same revisions and draft conversion hash; original FP32 block scales; E8M0 disabled |
| Client | AIPerf 0.12.0, concurrency/workers 192, ISL4000/OSL200, 60/120/30, seed2026, first-turn cache bust | Same official command and successful registration/record-processor settings |
| P KV allocation | 131072 tokens per DP rank; page64 | 2048 blocks ×64 per DP rank |
| D KV allocation | 393216 tokens per DP rank; page64 | 6144 blocks ×64 per DP rank |
| Chunk / admission | P4096 tokens and48 requests per rank; D2048 and32 | Same numerical budgets. Scheduler implementations and chunk admission differ |
| Attention / KV | FA3 / FP8 E4M3 | FA3 / FP8 E4M3; framework-specific KV grouping and transfer metadata |
| EAGLE3 | Three steps, topk1, four draft-token rows | Three speculative proposals and up to four target verification tokens; native verification, draft scheduling and acceptance implementation |
| Graphs | P eager; D22 request-size buckets through32 | P target/draft eager; D native FULL_AND_PIECEWISE, same request buckets mapped to four verification tokens through128 |
| Prefix cache | P radix cache enabled; D disaggregated radix cache disabled | P native prefix cache; D prefix cache disabled. Implementations differ; official cache bust preserved |
| Outer P routing | Minimum live load, retain reservation through body completion | Least in-flight with rotating ties, reservation held through body consumption |
| Inner DP routing | SGLang `total_requests` | vLLM native waiting/running/in-flight accounting with KV-pressure penalty; not an identical dispatch policy |
| Prefill handoff | SGLang ready-result fast path before another long forward | vLLM native scheduler/NIXL response path; no literal scheduler-patch translation |
| LM head / dense draft | P `enable_dp_lm_head=false`, TP4 group; D DP-local LM head and dense TP1 | Native TP1 replicates these weights per DP rank; P sharding differs |
| NIXL transfer direction | Prefill initiates UCX WRITE to decode | Decode initiates UCX READ from prefill; native handoff, ownership and metadata lifetime differ |
| NIXL thread configuration | P `num_threads=8`, D `num_threads=0` | Native `num_threads=4` for both roles; this is an agent configuration value, not a measured active-thread count |
| Installed NIXL package | `nixl` and `nixl-cu13` 1.5.0 in reference image38664373784 | 1.4.1 in both reproduction images; native connector compatibility and transport changes need a separate controlled trial |
| UCX device candidates | Every active HCA on each node (`mlx5_0:1` through `mlx5_9:1` in the reference logs) | Same active-HCA enumeration; no link-layer filter in either launch |

The vLLM adapter requires the production 4096-row symmetric workspace, observed as 1608529184 bytes per rank during the prior full-model startup. It is shared across 94 serialized MoE layers. Matching cache token budgets does not imply matching total GPU-memory accounting or identical scheduler overlap.

Performance must be reported from normal official passes only. Profile-only 20/30/30 runs diagnose kernels and communication and cannot supply official throughput. Numerical synthetic tests establish bounded adapter correctness, not the organizers' full-set accuracy score; their stated threshold is 99% of an unchanged official-FP8 reference score.

The user confirmed that no separate organizer accuracy procedure/data is shared for this work. Preserve the published checkpoint, original scales and kernel arithmetic; do not claim the organizer accuracy threshold was measured. The active diagnostic uses the installed native adapter, independent reference and changing-input CUDA Graph replays at the production workspace capacity before full-model startup. Its results must be checked independently of serving readiness.

## Observed kernel configuration

The archived native reports from SGLang 38664373784 and vLLM 38786945340 were independently re-exported with Nsight 2026.5.1. For both `prefill-0` and `decode-0`, the complete demangled MegaMoE template signature and all recorded launch resource fields match exactly (one specialization per role in each capture):

| Recorded field | Prefill, both frameworks | Decode, both frameworks |
|---|---:|---:|
| Grid CTAs | 132 | 132 |
| Threads per CTA | 640 | 384 |
| Registers per thread | 96 | 168 |
| Static shared memory | 0 B | 0 B |
| Dynamic shared memory per CTA | 225600 B | 230672 B |
| Aligned capacity template argument | 4224 | 4224 |

This verifies the selected kernel configuration as well as the shared patched source. It is not a throughput comparison: both traces retain completeness warnings, use different collectors at capture time, and contain instrumented workloads. CUDA/host scheduling, graph execution and actual routed-token distributions can still differ.

The equality above applies to the main `sm90_fp8_mega_moe` kernel. The preceding `sm90_mega_moe_pre_dispatch_kernel<128, kUsePDL>` differs: both inspected SGLang roles use `kUsePDL=true`, while both vLLM roles use `false`. SGLang's pinned `fc275674` wrapper (`deep_gemm_wrapper/entrypoint.py`) calls `set_pdl(True)` when `SGLANG_DEEPGEMM_PDL` is enabled; the private vLLM provider leaves DeepGEMM's default false unchanged. This controls programmatic launch dependency synchronization, not FP8 scales or expert arithmetic. Its performance impact has not been isolated. Matching it is a separate candidate requiring numerical/graph validation and a clean comparison; it is not included in the queued baseline or capture-repair jobs.

Image38798126037 enables private-provider PDL during per-worker workspace initialization, after GPU assignment. CPU imports remain CUDA-free, and the workspace is still cached once per EP group. The actual adapter GPU gate now asserts and records `predispatch_pdl=true` before running the same changing-input numerical/graph cases. The separate actual-adapter gate38801099834 / Slurm23911 passed all64 records with PDL enabled, maximum relative L2 error0.017077382653951645 and maximum scaled error0.01937984488904476. Normal trial38804437745 now completed full-model startup and official traffic; its SLA eligibility failed, as detailed below. The initial dense-kernel isolation below retains the PDL-disabled image.

## Normal decoder startup: controlled dense-kernel change

Clean PDL-off retry 38790949402 / Slurm23890 reproduced the earlier `CUDA error: invalid argument` at about15/22 PIECEWISE captures, before readiness or client traffic. Capture descriptors descend from128 through32 in the first15 entries; the next is28. The actual decoder log selects `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`. Its pinned source switches from ordinary DeepGEMM to FlashInfer swapAB at M<32. The installed FlashInfer `deep_gemm/jit_utils.cuh` emits the exact plain error and calls `exit(1)` from its CUDA driver-check macro; `runtime.cuh` uses that macro for loading/enumerating kernels and unloading the library. These observations identify a concrete path to isolate, but do not yet identify the individual failing driver call.

The clean serving trial72688a7 sets `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0` for decode as already done for prefill. It retains image38781401340, private MegaMoE PDL-off, original FP8 scales, all graph buckets, speculation, caches and topology. Only decoder dense FP8 selection changes to ordinary DeepGEMM. No profiler or debug instrumentation is added to this trial. The separate PDL-on image passed64 actual-adapter numerical/graph records in38801099834 / Slurm23911, but is deliberately not combined with this isolation trial.

That trial (38804183674 / Slurm23917) selected `DeepGemmFp8BlockScaledMMKernel`, completed all decoder graph buckets on all four ranks, and served the complete official workload without request errors. Startup readiness took348 seconds. This isolates a working dense-kernel path; the exact CUDA driver call behind the former failure remains unidentified. Official SLA eligibility failed, so startup repair alone does not establish performance parity.

The following controlled trial pins image38798126037 (SHA256 `06d4f8e0778a94739aaf8230865d5f393a76d6c8192d68d4b5b2b8f24183f35b`) after the PDL-on actual-adapter gate passed. Relative to72688a7, only the serving image changes to enable private-provider pre-dispatch PDL. The same decoder dense-GEMM override is retained. Both releases may queue concurrently, but each has its own commit and immutable run identity. PDL-on normal trial38804437745 / Slurm23918 completed in10m30s with readiness after348 seconds,1952 warmup and3942 measurement requests, and zero request errors/cancellations. The repository validator rejected eligibility:2818/3942 good requests (71.4866%),1122 TTFT misses and3 ITL misses with1 overlap. Measured SLA-passing output throughput is4491.97 TPS (AIPerf goodput×200); the archived local gated-score field is0 because eligibility failed, overall6283.51 TPS, mean TTFT2845.01ms and mean ITL15.7864ms. All61 canonical archive checksums pass and remote staging is removed. One observation per PDL condition does not establish a repeatable performance effect; retain PDL-on for reference fidelity, not as a demonstrated speedup.

## Native transport differences

Pinned SGLang `fc275674` (`python/sglang/srt/disaggregation/nixl/conn.py`, `NixlKVManager.__init__`) explicitly creates its UCX backend with strict thread synchronization, configures eight threads on the producer and zero on the consumer, and submits WRITE transfers. Its default backend-parameter object is empty; the archived reference worker adds no per-NIC backend override. Pinned vLLM0.30.0 (`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py`) uses the native UCX agent with `num_threads=4` and telemetry enabled; `pull_worker.py` submits READ transfers. The current worker supplies no `kv_connector_extra_config` override for these values.

Both launches expose the same ten active HCAs to UCX. That equality does not prove that UCX selects identical transport lanes for READ and WRITE, or that NIC/GID state is stable across runs. Preserve the native transfer protocol in the reproduction. Copying SGLang's thread counts by P/D role without accounting for the reversed transfer initiator would not reproduce its communication behavior. Any network-device or thread-count trial must be tracked separately from the pending dense-kernel/PDL comparison.

Package metadata read directly from the immutable SGLang reference image confirms `nixl==1.5.0` and `nixl-cu13==1.5.0`; the vLLM image checks record1.4.1. The selected vLLM connector source files (`base_worker.py`, `pull_worker.py`, `nixl_utils.py`) have identical SHA256 hashes in both reproduction images and match the pinned v0.30.0 sources. No transport override or package upgrade has been applied.

The next PDL-on profile retains the one-second CUPTI flush and capture stop after request/grace completion. Temporary DEBUG/stack-diagnostic environment settings from the old startup investigation are removed now that clean graph startup succeeds. Before serving, one worker per node records NIC link layer, state, rate and nonzero GIDs; this does not change UCX device selection. A separate read-only inventory step on23917 failed to complete launch and only that extra step was cancelled; embedding the inventory in the profile workers avoids another allocation step. Profile throughput remains diagnostic only.

## Prefill-admission experiment

The first normal workload38804183674 has1069 TTFT misses and only4 ITL misses, all overlapping TTFT misses. Even requests starting30–120 seconds into measurement have648 TTFT misses. Warmup-plus-measurement worker counters report roughly1.12 seconds each of P queue time and scheduler-defined prefill time, zero preemptions, and similar load across all three P groups. These are stage-level observations, not a kernel critical-path measurement.

An earlier, incomplete diagnostic trace38786945340 records `execute_context_2(4096)_generation_0(0)` for the inspected P iterations. The native scheduler can fill a4096-token budget with one prompt and part of the next; its asynchronous pipeline may keep earlier results pending while further work is scheduled. The MP worker also has a separate output-copy thread, so enqueue order alone does not prove a whole-forward handoff delay. Do not use the incomplete trace for a speedup estimate.

Cancelled before GPU execution: candidate7b35dce / Actions38820867351 / Slurm23942 sets `--max-num-seqs 2` on all three P groups, versus48 in normal baseline38804437745. This tests whether restricting admission reduces prompt splitting and TTFT tails without losing material serving throughput. It is an intentional optimization deviation from Komori's admission limit, not a newly matched baseline. Keep client concurrency192, P4096/D2048 token budgets, P/D KV capacities, native asynchronous scheduling, graphs, routing, original scales, and the validated PDL-on modified MegaMoE image. Decode remains32 requests per rank. The result is unproven until the repository validator, complete raw output and the same-image P48 comparison are available; a repeat is needed before a performance conclusion.

## Client-image fidelity control

The archived reference38664373784 used image38664373784 for both SGLang serving and AIPerf. Earlier vLLM trials, including38804183674, baseline38804437745 and candidate38820867351, used client image37886841020. Direct read-only extraction from both immutable images found identical SHA256 hashes for all828 AIPerf files excluding bytecode. Among18 selected package versions, only Pydantic differs: reference `pydantic==2.14.1` / `pydantic_core==2.50.1`, versus earlier client `2.13.5` / `2.46.5`. This does not establish equivalence of all runtime dependencies or demonstrate a performance effect.

A separate normal control restores matched P48 admission and pins the exact reference client image38664373784, SHA256 `8df47c11d7ede66fd19f12c7012a9ab5c97b13387a2655fcbce9f13e1c334110`,12249612288 bytes. Serving remains vLLM image38798126037 with the validated PDL-on modified MegaMoE. Relative to baseline76261bc /38804437745, the only major runtime factor is the client image. No rebuild is needed. The old-client P2 candidate7b35dce and profileedd9929 were cancelled while pending, with zero GPU runtime. Their artifacts were collected and staging removed. Neither supplies a measurement; any future admission trial must use the aligned client and a distinct tracked commit after the complete preflight.

Reference provenance records `profile=true` and `normal_benchmark_before_profile=true`: the official60/120/30 pass precedes a separate20/30/90 profiling pass. Nsight is attached at worker startup even though capture is inactive during the official pass. The6344.96 result is therefore not an active-capture throughput measurement; zero overhead from profiler attachment has not been established. Current normal vLLM controls have no profiler attached.

## Reporting correction: throughput, eligibility and competition scoring

The two completed normal vLLM runs measure4586.79 TPS (PDL-off38804183674) and4491.97 TPS (PDL-on38804437745) from SLA-passing requests. These measured good-output rates must not be reported as0 TPS. Their good-request fractions72.9024% and71.4866% fail the90% eligibility threshold. The stored `score_output_tokens_per_second=0` is this repository parser's local gated-score convention, not organizer-generated throughput. Earlier progress reports calling it an official0 TPS score are superseded by this correction; raw artifacts remain unchanged.

The organizer's current [README Addendum R4](https://github.com/hpcac/2026-APAC-HPC-AI/blob/main/README.md#r4-scoring) assigns the reference deployment's score to a re-run below90% good requests or below that reference. This replaces the older zero-performance-component rule. The numerical reference score has not been established in this investigation, and Komori's6344.96 TPS is not assumed to be that reference. Report measured good-output TPS, overall TPS and eligibility separately from the competition's scoring assignment. The local validator is repository-owned software implementing checks, not an organizer-supplied parser.


## Combined preflight after cancellation of obsolete jobs

The user requires all known repairs and inexpensive checks to be completed before another GPU submission, and authorizes immediate cancellation of obsolete owned jobs. Retain23943 /38821794697: it isolates the exact reference client while keeping the validated PDL-on serving image and P48 settings. Do not resubmit the cancelled old-client jobs.

Reviewed execution path: Actions build/reuse selection, immutable image hashes and provenance, draft/tokenizer staging, native adapter and launch settings, scheduler endpoints, fixed official client, profile start/stop and collector/export version, result validation, summary/analysis, artifact collection and terminal cleanup. No new serving or kernel changes are included in the reporting repair.

| Observed problem | Disposition before the next allocation |
|---|---|
| Decoder FlashInfer dense path fails during small-shape graph capture | Both roles select native dense DeepGEMM; two complete normal runs confirm startup. Modified MegaMoE remains the MoE backend on all16 GPUs. |
| Private predispatch PDL differed from Komori | Serving image38798126037 enables PDL; actual-adapter gate38801099834 passed64 numerical/graph records. |
| AIPerf client dependencies differed | Exact reference client38664373784 pinned; retained23943 isolates this change. |
| Old collector/driver incompatibility and incomplete later captures | Collector and exporter pinned to2026.5.1.161; one-second flush, stop after diagnostic request/grace completion, all-worker strict validation. CPU fixtures pass; a complete GPU capture is still required. |
| Local zero sentinel reported as measured or organizer throughput | Parser and embedded submission emit explicit measured good-output TPS; summaries, Slurm output and stage analysis distinguish it from the legacy local gate and eligibility. Organizer R4 floor is not assigned an invented numerical value. |
| Eligibility rejection skipped ISL/OSL validation | Validate both distributions before the eligibility gate; old input/output checks remain strict. |
| Slurm summary printed a missing good-request count | Read the count from the AIPerf export and print both throughput measures plus latency percentiles. |
| Failed startup profile summary inferred six workers | Read report count from staged endpoints; unknown runtime topology remains unverified. |
| Stage-budget target used twice the gated zero | Use measured goodput; zero good fraction produces unattainable/null current-fraction estimates. |

Combined CPU validation:474 tests, Ruff,46 shell syntax checks, workflow/AWS YAML parsing, actionlint1.7.12.24 with ShellCheck, CloudFormation lint and Git whitespace checks pass. Read-only replay of38804183674,38804437745 and38664373784 preserves every old parser field and validation outcome, verifies ISL/OSL distributions, and produces measured good TPS4586.7894,4491.9727 and6344.9646. Raw archives are unchanged.

The intermittent UCX address-handle failure remains unexplained; two later normal runs have zero transfer failures. Existing profile-only NIC/GID inventory records both nodes without changing transport selection. NIXL version/direction/thread differences and scheduler/LM-head differences remain documented framework differences, not silently bundled speculative fixes. A valid profile is needed before kernel-critical-path claims or a broad optimization batch. The P2 admission hypothesis is unmeasured and has no performance claim.
