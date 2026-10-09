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

Pending execution. Verify resolved server-info, both P DP ranks and NIXL handoff, unchanged image, scheduler terminal status, official-condition parser replay, per-request SLA, archived hashes and runtime cleanup. Raw throughput is not an eligible score when good-request fraction is below90%. Separate allocation/kernel/graph differences preclude attributing all differences from the old CUTLASS baseline to EP. Repeat a promising configuration before adoption; organizer full-set accuracy remains a separate gate.
