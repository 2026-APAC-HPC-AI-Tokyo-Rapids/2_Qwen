# Bounded Qwen optimization assessment, 2026-10-03

## Decision and scope

The evidence supports a **mixed backend**, not a universal DeepGEMM winner. In the fixed H100 TP4/EP1 deployment, auto MoE selects Triton, some dense FP8 GEMMs use DeepGEMM, attention uses FA3, and FlashInfer TRTLLM all-reduce fusion is already active. A from-scratch implementation should specialize for the measured shapes and serving pipeline rather than replace every component with one library.

Prefill-only native CUTLASS is the sole new candidate with positive measured evidence: +3.943% in the first paired comparison. It is not adopted pending a matched repetition. Decode-only DeepGEMM and CUTLASS both lost throughput. CPU screening does not justify further GPU experiments on larger chunks, the intended 4P/12D layout, or a doubled single-decode KV pool. These are bounded decisions, not proofs of global optimality.

Progress and decisions are recorded in [Issue #17](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/issues/17), including the [CPU screening and complete first campaign](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/issues/17#issuecomment-5965381660).

## Comparable first campaign

All five runs reuse image `6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5` and SGLang source `fc275674a97d78553b22597ee72c8fd48f5420f4` on dgx026/dgx030. Control: 16 H100, P2TP4/D2TP4, EP1, EAGLE3 steps3/draft4, BF16 draft, FP8 KV, FA3, round-robin prefill routing, c40, chunk8192, pool131072 per worker, no profiler. Official workload is ISL4000/OSL200 with 60s warmup, 120s measurement and 30s grace.

| Condition | Commit | Actions run / attempt | Good output tok/s | Mean TTFT ms | Mean ITL ms |
|---|---|---|---:|---:|---:|
| Control | `642e97d` | `37091361072/1` | 2472.821915 | 1274.523 | 9.557 |
| Decode DeepGEMM | `c5e1122` | `37091433995/1` | 2386.516231 | 1055.247 | 11.229 |
| Decode native CUTLASS | `e5bda25` | `37091489116/1` | 2421.881838 | 1118.957 | 10.669 |
| P3TP4/D1TP4, same pool | `4ac2cf6` | `37092369218/1` | 1863.594883 | 1675.303 | 12.676 |
| Prefill native CUTLASS | `ac3c70d` | `37093476615/1` | 2570.325310 | 1102.011 | 9.854 |

Actions/PBS terminal success, canonical archive checksums, actual worker arguments, model/image identity, parser replay and per-request/output evidence were checked for every run. The parser is this repository's implementation of published rules, not an independent organizer-supplied accuracy checker. In the prefill-only candidate, both `moe_runner_backend` and inherited `speculative_moe_runner_backend` change from auto to cutlass on prefill; decode arguments remain unchanged. All five satisfy the joint request-level SLA gate. The first CUTLASS-prefill run has 1585 requests, ten TTFT violations, no ITL violations, errors or cancellations, and 99.369% good requests.

The score is good requests/s times 200; it is not raw generated throughput. Each good request must have TTFT <=3000ms and mean ITL <=20ms, and at least90% of requests must jointly qualify. Slightly slower decode streams can be acceptable if concurrency and goodput improve. Here, decode DeepGEMM increased measured client decode occupancy from23.734 to26.921 but still lost3.49% goodput. Conversely, prefill CUTLASS improved goodput despite mean ITL rising from9.557 to9.854ms; measured effective decode concurrency rose from23.734 to25.362 while prefill concurrency fell from15.904 to14.253. This is an observed example of the requested per-stream-latency versus total-throughput tradeoff, pending repetition. Headroom in ITL alone does not establish a profitable tradeoff.

Eight fixed short greedy probes match 8/8 for P3D1, but 3/8 for each alternative backend. Numerical divergence alone is not established task-accuracy degradation. Official target weights and target verification remain intact; obviously accuracy-reducing alternatives are excluded. No full organizer accuracy checker was provided or required by the user.

## Repetition status

Candidate `37093476615/2` is fully audited:2554.714928575 good tok/s, meanTTFT1101.819692ms, meanITL9.923535ms, good fraction99.3650794%,1575 requests, ten TTFT violations, no ITL violations/errors/cancellations. It is0.607331% below candidate attempt1. All eight fixed greedy outputs match candidate attempt1. Image/model/config/actual arguments match except incidental server seeds. PBS215848 completed F/0 in11:43; archive hashes and parser replay pass.

This is a repeated candidate measurement, **not a completed matched repetition**. Control `37091361072` was requeued at04:10UTC but has remained pre-job queued with an empty check suite and run_attempt1 for more than30minutes. No duplicate control has been submitted or run cancelled. A comment-only replacement `ed5ac7e`, parent `642e97d`, is prepared locally with identical runtime bytes; the user explicitly authorized cancelling this redispatch and replacing it. Both direct REST `POST /cancel` and `POST /force-cancel` return HTTP409: `Cannot cancel a workflow re-run that has not yet queued.` CLI cancellation instead describes the run as completed, while GET reports queued and Jobs remains empty. Cancellation has not succeeded; replacement stays unpushed to avoid duplicate GPU work. Luna monitors for a cancellable state. No further cancellation permission is required for this exact run. The [documented API cancellation paths](https://docs.github.com/en/rest/actions/workflow-runs#force-cancel-a-workflow-run) have been attempted; no local runtime defect has been established.

Fourteen derived candidate-repeat audit files are verified at `backend-campaign-audit/repeat-attempt-2/candidate/`, manifest SHA256 `b8abb8f5ba436402db8e5d6cf2a7a6311215f3a195bd48fdbbe560a9bd10759c`.

Local integration `e3a8ad3`, branch `research/aspire-cutlass-integration-20261003-sol`, ports only the needed ASPIRE runtime/configuration onto main.180 tests and static checks pass; executable launch fixtures match the audited candidate and preserve non-ASPIRE behavior. It is unpushed and unadopted. If repeated comparison confirms a gain, the newly assembled orchestration requires its own E2E run, approximately3.2 additional allocated GPU-hours; no integration run has been launched.

## Intended asymmetric layout and batching

The user's intended candidate is **prefill4GPU/decode12GPU: P1TP4/D3TP4**. The previously launched P3D1 trial is the reverse direction and was launched before sufficient simulation. Its failure does not reject P1D3. Major layout changes must be screened before spending GPU credits.

AISimulate uses413 collected whole-forward timing rows, target-only BF16 KV/attention, FP8 weights, synthetic4000/200 requests, c40 and4000 completions. It does not faithfully implement deployed EAGLE3/FP8-KV serving or the actual512-token admission reservation and preallocation queues. Its throughput is not an official AIPerf score.

| Native CPU condition | Output tok/s | Joint SLA fraction | Mean TTFT ms |
|---|---:|---:|---:|
| P2D2, chunk8192/s8 | 2459.50 | 1.0000 | 216.20 |
| P2D2, chunk16384/s4 | 2369.15 | 1.0000 | 341.47 |
| P2D2, chunk16384/s8 | 2350.00 | 1.0000 | 390.68 |
| P2D2, chunk32768/s8 | 2317.08 | 1.0000 | 444.98 |
| Intended P1D3, chunk8192 | 1383.74 | 0.0050 | 3235.74 |
| Intended P1D3, chunk16384/s4 | 1438.72 | 0.0500 | 3007.01 |
| Intended P1D3, chunk32768/s8 | 1464.82 | 0.9935 | 2891.75 |

The Python generic `max_batched_tokens` knob did not change native SGLang's actual chunk budget. Corrected experiments supply existing nested `sglang.chunked_prefill_size` and `sglang.max_prefill_tokens` fields; the native default reproduces control exactly. Every executed configuration is retained. Earlier identical outputs from the inert generic knob are not evidence of absent batch demand. Timing interpolation was not patched and unsupported shapes retain `fallback_policy: deny`.

Measured prefill B1/B2/B4/B8 full-prompt forward latency is154.796/293.657/553.671/1087.219ms. B2 toB8 improves forward input throughput by only about8% while increasing batch latency3.7 times. One TP4 prefill worker would need approximately49,456 input tok/s, or32000 tokens in <=647ms before other overhead, to sustain the measured control's2472.82 good output tok/s. Current B8 timing implies29,433 input tok/s. No available backend measurement establishes the required approximately40.5% latency reduction. This is an optimistic necessary bound assuming every processed request qualifies; servicing SLA-failing requests would require slightly more capacity. This conditional capacity bound explains why more decode GPUs are not automatically better for this4000/200 workload; it is not an EAGLE3 prediction.

The actual P3D1 run shows a sampled preallocation queue of11.50 requests and mean running batch24.91. Logged `#transfer-req` includes bootstrap, prefill and transfer-related waiting; it is not an isolated network-transfer latency measurement. These log samples are not wall-time-weighted measurements of every iteration.

Original-pool unconstrained P3D1 exceeds measured decode KV coverage; doubled-pool P3D1 reaches unmeasured B36. External sequence caps27/29 are labeled admission sensitivities, not a faithful reservation implementation. A separate empirical batch/ITL sensitivity admits a pool win only under optimistic latency assumptions; it is not robust. Prepared pool commit `9440fa6` remains unpushed.

## Kernel and call overhead: what is established

The retained decode trace `36387126341` contains1,601,660 kernels over four ranks;96.18% are CUDA-Graph-associated. It records1176 `cudaGraphLaunch` calls and existing fused all-reduce activity. Large kernel counts therefore do not imply equally many Python/CUDA launch calls on the critical path. Explicitly selecting the existing all-reduce `auto` backend would be a no-op.

Approximately70% of kernels in an endpoint-selected two-second window last under5us, but represent only18% of summed kernel duration. Whole-trace summed categories are approximately50% MoE GEMM,10% all-reduce,10% attention including preparation/combine,6% quantization and6% other DeepGEMM. These are **summed GPU kernel durations, not additive end-to-end critical-path fractions**. Four ranks are not independent repetitions; the selected endpoint is not verified steady state, and the trace records `stop_profile_failed`. Host API durations overlap device work and cannot simply be added. No draft/verify NVTX split was observed, so these category costs cannot be assigned separately to those stages. Utilization, memory activity and power samples do not establish a launch-limited or bandwidth-limited system.

Plausible remaining improvements are shape-specific MoE tiling, reducing intermediate routing/quantization traffic, and targeted fusion. Current Triton uses a compatible tuned H100 configuration despite a version warning; it is not demonstrated to be generic untuned code. Native CUTLASS's128-row tiles are structurally more plausible for the theoretical mean512 routed rows/expert for a full8192-token prefill chunk (top8 routing across128 experts; actual expert skew was not measured) than tiny decode shapes. DeepGEMM padded allocation is a capacity cost, not proof all padding executes as FLOPs.

Supported QK-norm/RoPE fusion remains a smaller candidate, but available category totals do not prove an end-to-end gain or isolate its full scope. The proposed MoE sum/all-reduce option uses BF16 atomic accumulation instead of normal FP32 reduction and does not fuse the inter-GPU collective; it is deprioritized under the accuracy policy. No custom kernel is claimed faster without suitable measured evidence.

## Reproduction, records and next gate

Raw GPU outputs remain on `tomokazu-ubuntu-server` at `/hdd_raid/samba/net_share/qwen-artifacts/<run>/<attempt>/`. Derived audit helpers, pinned parser, comparisons, checksums and provenance are under `/hdd_raid/samba/net_share/qwen-artifacts/research-issue-17/20261003-existing-data-analysis/backend-campaign-audit/`; its manifest SHA256 is `a22f8d9acba3ba87f49e95b3eb6a69d85e2affa4bd2f842541dd0d661642ccb3`.

CPU source/configuration lives on `research/cpu-screen-20261003-sol`, commit `5207e1b8a3010743cf7c0a6e1310e3caa0c43a94`, with per-condition commits. Its README contains exact reproduction commands and model-path identity requirements. FPM SHA256 is `d185ab5430d90031a5b8f6e92a5234e989d278e86a79e3ee3970cf586eaf3215`. CPU raw evidence is archived beside the campaign audit under `cpu-screen/`, with218 verified files/references, no broken symlinks, and manifest SHA256 `aeda11e25c084697d202e4a496ef17c97f0202c758f80966cc83c5b45260c9dc`. Existing canonical FPM/metadata/wheel are referenced without duplicate copies. Reproduction source, dependency snapshot, model config and Git bundle are preserved. Fresh installation is not tested, and the exact collected model path is absent on the archive host; the archive is reproducible evidence, not a claimed turnkey installation.

The only additional paid work selected is one control/prefill-CUTLASS matched repetition, approximately2 x12minutes x16GPUs =6.4 allocated GPU-hours. Keep the fixed image, workload, topology, node pair and runtime settings; change only the prefill MoE backend and record incidental random seeds. Require repeated improvement, acceptable request-level SLA and output evidence. Run IDs/attempts preserve the separate condition commits. No broad layout/pool/batch GPU sweep is justified. Simple monitoring uses Luna; defined implementation and audits use GPT-6.1-Sol; no Astra subagents.
