# Firmus: shared-expert Prefill trials

The user requested the Attention TP4 × DP2 configurations with EP2/MoE TP4 and EP8/MoE TP1. Prefill is changed first because the preceding c96 trials accumulated a Prefill queue while Decode remained underfed. Decode remains two independent TP4/EP1 workers. No source patch or new image is used in this campaign.

## Controlled comparison

Both native candidates use world8, Attention DP2/TP4, MoE DP1, native (`none`) MoE communication, and Triton MoE. Only EP degree and corresponding labels/assertions differ between the candidates. EP2 has64 local experts and intermediate width384; EP8 has16 local experts and width1536. EP8 removes MoE tensor parallelism but retains Attention tensor parallelism and DP/TP boundary collectives.

Common settings: c96, P pool131072, D pool262144, FP8 weights/KV, FA3, EAGLE3 steps3/draft4/topk1, BF16 draft, P max-running96, D256, D graph buckets unchanged, round-robin gateway. P CLI chunk16384 resolves to DP-local8192; max-prefill-tokens16384 remains local. P legacy CUTLASS environment override is explicitly0. Each allocation is2 nodes/16 H200 GPUs with17-minute wall limit and the unchanged60/120/30 official client. Native P graph defaults are retained.

The additional EP8 backend candidate uses DeepEP normal + DeepGEMM with native Triton draft. DeepEP normal disables P CUDA graphs. This is a backend bundle comparison, not an EP-only comparison. DeepEP cannot supply the EP2 control: fixed source overrides EP to world TP8 when DeepEP is selected.

| Candidate | Condition commit | CPU fixture correction / run commit | Actions |
|---|---|---|---|
| EP2, native Triton | 8ce93a8d04e26d84b544085cf1fafa669c68325e | 677d14717db66811a84e03d84f638a73f4926b83 | [37877571478](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37877571478) |
| EP8, native Triton | af9d2461ef890d8ee8fce2ba345b17c26f271217 | ffdc3d4b91c05e844361dd28f255dc8ee034788e | [37877573181](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37877573181) |
| EP8, DeepEP normal / DeepGEMM | 68657a80a52619cd47fb14f47c6d8a3966fe4842 | 41c12d6cb3ec4bea36b34705b133372eb0816a43 | [37877564253](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37877564253) |

All condition commits descend from f0849435, the preceding c96 capacity trial. Image SHA256 is `182fbb6df7f941aa688a393dfc181bb269f3112a6d34c33f6007ddc5d09f0a42`, source fc275674 with the existing four patches. The native CUTLASS EP patch/probe remains an unbuilt separate candidate and is not part of these measurements.

## Preflight and initial failure

Fixed-source review verifies bootstrap advertises Attention TP4 rather than world TP8, maintains both DP rank entries, and uses the same bootstrap-room modulo mapping at P and D. No router DP-aware flag is needed for this path. Static compatibility is not a replacement for runtime NIXL/EAGLE checks.

The first Actions runs37877346628/37877349293/37877457052 stopped in CPU validation because `test_firmus_profile.py` still expected the old summary label `P2 × TP4 / D2 × TP4 EP1`. The summary had been edited after the earlier full test run; only a narrower suite was rerun. Corrected the candidate-specific expected label and reran the entire247-test suite after all changes. Serving settings did not change. No Slurm jobs or GPU charges resulted from those three validation failures.

## Prediction recorded before GPU results

The preceding generic H200 target-only screen at global4×4000 predicts347.676ms for EP2 native Triton and411.807ms for EP8, approximately18.4% slower for EP8. This is a frozen shape-level hypothesis, not an official goodput forecast: real EAGLE, graph selection, batching, routing distribution and collective implementation differ. The generic screen does not predict the DeepEP/DeepGEMM bundle. Compare resolved shapes and queue observations before interpreting agreement or disagreement; do not fit a correction coefficient on these same runs and call it independent validation.

## Acceptance and results

Slurm jobs21977 (EP8 native),21978 (EP2 native),21979 (EP8 DeepEP) were pending resources at03:43 UTC. Both available H200 nodes were occupied; scheduler priorities also placed other pending jobs ahead. Submission order is not a start-time guarantee. No GPU measurement is available yet.

Verify resolved server-info, both P DP ranks and NIXL handoff, unchanged image, scheduler terminal status, official-condition parser replay, per-request SLA, archived hashes and runtime cleanup. Raw throughput is not an eligible score when good-request fraction is below90%. Separate allocation/kernel/graph differences preclude attributing all differences from the old CUTLASS baseline to EP. Repeat a promising configuration before adoption; organizer full-set accuracy remains a separate gate.

## Next candidate preparation: removing Attention TP too (not submitted)

Fixed-source paths exist for world8/EP8/Attention DP8, giving Attention TP1 and MoE TP1. This does not eliminate all tensor parallelism: current candidates leave `enable_dp_lm_head=false`, so the target LM head remains world TP8, including DP hidden gather, world-TP logits gather and DP scatter. Target embedding follows the Attention TP group. Fixed-source references: `server_args.py:1123–1131`, `qwen3_moe.py:963–968`, `qwen2_moe.py:852–855`, `logits_processor.py:352–366,721–738,802–811,839–855`. EAGLE does not automatically enable DP LM head.

Adding `--enable-dp-lm-head` to ADP8 would also make the target head TP1; source validation requires DP Attention and has no discovered EAGLE prohibition, but the combination is untested. The FP8-excluded target BF16 head (151936×4096) would grow from1/8 shard to a replicated1.159GiB per GPU, approximately1.014GiB extra before logits/workspace. Draft has its own64000-vocabulary head and constructs under the Attention TP group, so its TP1 replication is a separate cost (`eagle_worker_v2.py:149–172,283–294`, `llama_eagle3.py:274–294`). No queued configuration was changed.

P-only changes introduce P Attention TP1→D TP4 head slicing in NIXL; the implementation exists, but prior EAGLE heterogeneous-TP transfer failed with an unresolved initial `NIXL_ERR_REMOTE_DISCONNECT`. Changing both roles avoids heterogeneous head slicing while introducing more simultaneous factors. Neither path is validated by the current trials.

Keep aggregate budgets distinct from per-DP settings. Current aggregate chunk16384 becomes local2048 at DP8; keeping local8192 would raise aggregate chunk to65536. A fixed per-DP KV pool allocates four times more KV bytes/rank at Attention TP1 than TP4. This is not an unavoidable penalty at fixed global capacity: P pool32768×DP8 preserves the current131072×DP2 token budget, and D pool65536×DP8 preserves the current262144×two-independent-workers budget. Target KV bytes/rank then remain approximately2.94/5.88 GiB for P/D, respectively, before extra buffers and reservations. DP-local fragmentation, admission reserve and imbalance still require measurement. EAGLE draft TP1 also adds weight replication.

For a prospective DeepEP low-latency Decode candidate, the default dispatch capacity128 is token rows, not requests. EAGLE verify width4 consumes128 rows at local batch32; local64 needs256 before graph padding. Do not copy the existing graph bucket list or request limit without checking this capacity. No extra GPU jobs were submitted for this TP1 preparation.

### Equal-work CPU screen for Attention TP1

A separate H200 target-only screen fixes role resources at8 GPUs and global Prefill input at8 or16 requests×4000, keeping integer requests per DP rank even at DP8. These32k/64k aggregate shapes exceed the current16k aggregate chunk and must not be read as predictions of the queued trials. All20 queries returned with no reported estimator fallback; operator-table extrapolation remains possible.

| Layout | P global32k ms | P global64k ms | D global64 ms | D global128 ms |
|---|---:|---:|---:|---:|
| Independent TP4×2 | 482.157 | 953.404 | 24.042 | 27.301 |
| Attention TP4/DP2, EP2 | 676.040 | 1336.112 | 23.311 | 27.027 |
| Attention TP4/DP2, EP8 | 806.881 | 1597.794 | 22.917 | 26.430 |
| Attention TP2/DP4, EP8 | 757.689 | 1491.662 | 22.728 | 26.365 |
| Attention TP1/DP8, EP8 | 722.503 | 1414.736 | 22.509 | 26.079 |

Within the EP8 family, eliminating Attention TP shortens predicted Prefill latency by10.5–11.5%; this still exceeds the EP2 estimate by6–7%. The experiment should therefore test the benefit without assuming a global optimum. These are AISimulate0.13.0 FP8 Attention/KV, legacy native Triton MoE with default routing alpha1.2, context4100 Decode estimates. They exclude EAGLE and do not predict DeepEP/DeepGEMM, admission, queueing or official goodput. The returned Prefill diagnostics also have no separate LM-head term; they do not validate the real LM-head group or the effect of enabling DP LM head. Queries, full operator diagnostics, wheel/model/MoE-table provenance and six verified raw-file hashes are archived at `/home/USER/qwen-artifacts/research-simulation/20261009-adp8/`.
